from mmengine.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F

def _mbt_mask(n_img: int, n_lid: int, n_bn: int, device):
    """
    Build an MBT mask for a single self-attention op over
    [img (n_img), lidar (n_lid), bottlenecks (n_bn)] tokens.

    True = disallowed attention (per PyTorch docs).
    Allows:
      - img <-> img
      - lidar <-> lidar
      - {img,lidar,bn} <-> bn   (both directions)
    Blocks:
      - img <-> lidar (both directions)
    """
    L = n_img + n_lid + n_bn
    m = torch.zeros(L, L, dtype=torch.bool, device=device)

    img_slice = slice(0, n_img)
    lid_slice = slice(n_img, n_img + n_lid)
    bn_slice  = slice(n_img + n_lid, L)

    # Block direct cross-modal
    m[img_slice, lid_slice] = True
    m[lid_slice, img_slice] = True

    # Everything else stays allowed (False):
    # - within-modality self-attn
    # - {img,lidar} <-> bn in both directions
    return m

class _MBTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(proj_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.drop2 = nn.Dropout(proj_dropout)

    def forward(self, x, attn_mask):
        # x: [B, L, C], attn_mask: [L, L] (bool or float)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)  # masked self-attn (single op)
        x = x + self.drop1(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop2(h)
        return x

@MODELS.register_module()
class TransformerFusionMBT(nn.Module):
    """
    MBT-style masked self-attention fusion.
    - Projects img & lidar to same embed dim, pools to S×S -> sequences N=S*S
    - Adds T learnable bottleneck tokens
    - Concats [img || lidar || bn] and runs masked self-attn so img<->lidar
      communication is ONLY via bn tokens (MBT).
    - Optionally maps T bottleneck tokens to an S×S grid via a learned projector,
      so your bbox head can still treat fused tokens as a spatial map.

    Returns:
      lidar_seq: [B, N, E]
      img_seq:   [B, N, E]
      fused_seq: [B, N, E]    # bn tokens projected to N=S*S positions
    """
    def __init__(self,
                 d_model=256,
                 num_heads=8,
                 mlp_ratio=4,
                 feature_size=32,             # S (pooled H=W)
                 in_img_channels=256,
                 in_lidar_channels=384,
                 num_bottleneck_tokens=32,    # T (small, MBT-style)
                 num_layers=2,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.S = feature_size
        self.N = feature_size * feature_size
        self.E = d_model // 2

        # Projections to common embed dim
        self.img_proj   = nn.Conv2d(in_img_channels,   self.E, kernel_size=1)
        self.lidar_proj = nn.Conv2d(in_lidar_channels, self.E, kernel_size=1)

        # Learnable bottlenecks (shared across batch)
        self.T = num_bottleneck_tokens
        self.bn_tokens = nn.Parameter(torch.randn(self.T, self.E))

        # Stack MBT blocks
        self.blocks = nn.ModuleList([
            _MBTBlock(self.E, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      attn_dropout=attn_dropout, proj_dropout=proj_dropout)
            for _ in range(num_layers)
        ])

        # Map T bottleneck tokens to N=S*S grid so caller can reshape as 2D
        # bn: [B, T, E] -> [B, N, E] via learned mixing
        self.bn_to_grid = nn.Linear(self.T, self.N, bias=False)

    def forward(self, img_feats, lidar_feats):
        B = img_feats.size(0)

        # 1) Project & pool to S×S
        img = F.adaptive_avg_pool2d(self.img_proj(img_feats),   (self.S, self.S))  # [B,E,S,S]
        lid = F.adaptive_avg_pool2d(self.lidar_proj(lidar_feats),(self.S, self.S))  # [B,E,S,S]

        # 2) Flatten to sequences [B, N, E]
        img_seq = img.flatten(2).permute(0, 2, 1)  # [B,N,E]
        lid_seq = lid.flatten(2).permute(0, 2, 1)  # [B,N,E]

        # 3) Init bottlenecks [B, T, E]
        bn = self.bn_tokens.unsqueeze(0).expand(B, -1, -1)

        # 4) Concatenate & build MBT mask
        x = torch.cat([img_seq, lid_seq, bn], dim=1)  # [B, N_img+N_lid+T, E]
        attn_mask = _mbt_mask(self.N, self.N, self.T, device=x.device)  # [L,L]

        # 5) Run masked self-attention blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)

        # 6) Split back
        img_seq_out = x[:, :self.N, :]
        lid_seq_out = x[:, self.N:self.N*2, :]
        bn_out      = x[:, self.N*2:, :]  # [B,T,E]

        # 7) (Optional) map bottlenecks to spatial grid tokens so caller can reshape
        #     [B,T,E] -> [B,E,T] -> Linear(T->N) -> [B,E,N] -> [B,N,E]
        fused_seq = self.bn_to_grid(bn_out.transpose(1, 2)).transpose(1, 2)

        return lid_seq_out, img_seq_out, fused_seq, bn_out
