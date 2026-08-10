# projects/myfusion/heads/token_head.py
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes

import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class TokenQuery3DHead(BaseModule):
    """A simple DETR-style token head for 3D detection.

    Expects a token memory [B, N, E] and predicts M object queries with
    class logits and 3D box parameters. This head uses a *greedy* matching
    for training (fast baseline); you can later swap it to a Hungarian assigner.

    Args:
        num_classes (int): number of classes.
        embed_dim (int): token/hidden dim E (must match fusion output).
        num_queries (int): number of object queries M.
        num_decoder_layers (int): transformer decoder depth.
        code_size (int): box parameter size, e.g., 7 for (x,y,z,w,l,h,yaw).
        loss_cls (dict): mmengine build dict for classification loss.
        loss_bbox (dict): mmengine build dict for bbox regression loss.
        test_cfg (dict): inference cfg, supports 'score_thr' and 'max_per_img'.
    """
    def __init__(self,
                num_classes: int,
                embed_dim: int = 128,
                num_queries: int = 300,
                num_decoder_layers: int = 3,
                code_size: int = 7,
                loss_cls: dict = dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                loss_bbox: dict = dict(type='mmdet.L1Loss', loss_weight=5.0),
                test_cfg: dict | None = None,
                train_cfg: dict | None = None,
                **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.code_size = code_size
        self.test_cfg = (test_cfg or dict(score_thr=0.05, max_per_img=100))
        self.train_cfg = train_cfg  # usually unused in DETR-style heads

        # Queries + decoder
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Heads
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, code_size)
        )

        # Losses
        self.loss_cls_fn = MODELS.build(loss_cls)
        self.loss_bbox_fn = MODELS.build(loss_bbox)

    # ---------- helpers ----------
    @staticmethod
    def _ensure_tokens(x):
        """Accept [B,N,E], [B,C,H,W] or [tensor] and return [B,N,E]."""
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 4:
            # [B,C,H,W] -> [B,N,E] by flattening spatial dims
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        assert x.dim() == 3, f'Expected [B,N,E], got {tuple(x.shape)}'
        return x

    @torch.no_grad()
    def _greedy_match(self, cls_logits, box_params, gt_boxes, gt_labels):
        """Greedy 1-1 matching: for each GT pick the cheapest unused query.
        Cost = L1(center/size/yaw) + (1 - prob_of_gt_class).
        Returns a list of (pred_idx, gt_idx).
        """
        if gt_boxes.shape[0] == 0:
            return []

        with torch.no_grad():
            B = 1  # called per-batch item
            M = cls_logits.shape[0]
            G = gt_boxes.shape[0]

            # classification cost (use sigmoid prob at gt label)
            probs = cls_logits.sigmoid()  # [M, C]
            cost_cls = 1.0 - probs[:, gt_labels]  # [M, G]

            # bbox L1 cost on first code_size dims
            pred = box_params[:, :gt_boxes.size(-1)]  # [M, K]
            gt = gt_boxes[:, :pred.size(-1)]          # [G, K]
            cost_reg = torch.cdist(pred, gt, p=1)     # [M, G]

            cost = cost_reg + cost_cls  # [M, G]
            matched = []
            used_p = set()

            # simple greedy: iterate gts, pick best available query
            for g in range(G):
                # mask used
                c = cost.clone()
                if used_p:
                    used_mask = torch.tensor([i in used_p for i in range(M)],
                                             device=cost.device)
                    c[used_mask, g] = float('inf')
                p = int(torch.argmin(c[:, g]).item())
                if c[p, g] != float('inf'):
                    used_p.add(p)
                    matched.append((p, g))
        return matched

    # ---------- API ----------
    def forward(self, x):
        """x: [B,N,E] tokens; returns logits and box params."""
        mem = self._ensure_tokens(x)              # [B,N,E]
        B, N, E = mem.shape
        assert E == self.embed_dim, f'embed_dim mismatch: {E} vs {self.embed_dim}'

        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B,M,E]
        dec = self.decoder(tgt=q, memory=mem)                       # [B,M,E]
        cls_logits = self.cls_head(dec)                             # [B,M,C]
        box_params = self.reg_head(dec)                             # [B,M,K]
        return cls_logits, box_params

    def loss(self, x, batch_data_samples):
        """Greedy-matched losses over queries."""
        cls_logits, box_params = self.forward(x)  # [B,M,C], [B,M,K]
        B, M, C = cls_logits.shape
        K = box_params.size(-1)

        total_cls, total_bbox = 0., 0.
        for b in range(B):
            # gather GT
            sample = batch_data_samples[b]
            gtb = sample.gt_instances_3d.bboxes_3d.tensor.to(cls_logits.device)  # [G, 7 or K]
            gtl = sample.gt_instances_3d.labels_3d.to(cls_logits.device)         # [G]

            # handle no-GT case
            if gtb.numel() == 0:
                # all negatives
                target = torch.zeros((M, C), device=cls_logits.device)
                total_cls += self.loss_cls_fn(cls_logits[b], target)
                continue

            # match
            pairs = self._greedy_match(cls_logits[b], box_params[b], gtb, gtl)

            # build cls targets (one-hot at gt label for matched, zeros otherwise)
            t_cls = torch.zeros((M, C), device=cls_logits.device)
            pos_idx = []
            if pairs:
                for p, g in pairs:
                    t_cls[p, gtl[g]] = 1.0
                    pos_idx.append(p)
            total_cls += self.loss_cls_fn(cls_logits[b], t_cls)

            # bbox L1 on positives
            if pos_idx:
                p_idx = torch.tensor(pos_idx, device=cls_logits.device, dtype=torch.long)
                matched_gt = torch.tensor([g for _, g in pairs],
                                          device=cls_logits.device, dtype=torch.long)
                pred_box = box_params[b, p_idx, :K]
                gt_box = gtb[matched_gt, :K]
                total_bbox += self.loss_bbox_fn(pred_box, gt_box)

        losses = dict(
            loss_cls=total_cls / B,
            loss_bbox=total_bbox / max(1, B)
        )
        return losses

    def predict(self, x, batch_data_samples, **kwargs):
        """Return list[InstanceData] with 3D boxes, scores, labels."""
        cls_logits, box_params = self.forward(x)  # [B,M,C], [B,M,K]
        B, M, C = cls_logits.shape
        score_thr = self.test_cfg.get('score_thr', 0.05)
        max_per_img = self.test_cfg.get('max_per_img', 100)

        results = []
        probs = cls_logits.sigmoid()  # [B,M,C]

        for b in range(B):
            # per-query max class
            scores, labels = probs[b].max(dim=-1)  # [M], [M]
            bboxes_3d = LiDARInstance3DBoxes(box_params[b, :, :7])

            # filter
            keep = scores >= score_thr
            bboxes_3d = bboxes_3d[keep]
            scores = scores[keep]
            labels = labels[keep]

            # top-k
            if bboxes_3d.tensor.shape[0] > max_per_img:
                topk = torch.topk(scores, k=max_per_img, sorted=True).indices
                bboxes_3d = bboxes_3d[topk]
                scores = scores[topk]
                labels = labels[topk]

            inst = InstanceData()
            inst.bboxes_3d = bboxes_3d
            inst.scores_3d = scores
            inst.labels_3d = labels
            results.append(inst)

        return results
