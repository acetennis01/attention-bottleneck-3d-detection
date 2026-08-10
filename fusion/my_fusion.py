#from mmdet.models.backbones import ResNet
#from projects.BEVFusion.bevfusion import BEVFusion
#from mmdet3d.registry import MODELS
from mmengine.registry import MODELS
from mmdet3d.models import Base3DDetector
#from mmdet3d.models.builder import build_backbone, build_head
#from mmdet3d.registry import MODELS
#from mmengine.model import build_model
#from mmengine.model import build_backbone
#from mmdet3d.models.builder import build_backbone

# Use the registry system to build the backbone
build_backbone = MODELS.build

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@MODELS.register_module()
class MyFusionTransformer(Base3DDetector):
    def __init__(self, img_backbone = None,
                img_neck = None, 
                img_bbox_head=None,
                data_preprocessor =None, 
                lidar_voxel_encoder=None, 
                lidar_middle_encoder=None, 
                lidar_backbone=None, 
                lidar_neck=None, 
                fusion_module=None,
                bbox_head=None, 
                train_cfg=None, 
                test_cfg=None):
        super(MyFusionTransformer, self).__init__()

        # Image and LiDAR backbones
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.img_bbox_head = MODELS.build(img_bbox_head) if img_bbox_head else None
        
        # LiDAR pipeline
        self.data_preprocessor = MODELS.build(data_preprocessor)
        
        self.lidar_voxel_encoder = MODELS.build(lidar_voxel_encoder) if lidar_voxel_encoder else None
        self.lidar_middle_encoder = MODELS.build(lidar_middle_encoder) if lidar_middle_encoder else None
        self.lidar_backbone = MODELS.build(lidar_backbone) if lidar_backbone else None
        self.lidar_neck = MODELS.build(lidar_neck) if lidar_neck else None
        
        # Fusion Transformer Module
        self.fusion_module = MODELS.build(fusion_module)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        # Bounding Box Head
        self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        logger.info(f"✅ Initialized MyFusionTransformer with img_backbone={img_backbone}, "
                    f"img_neck={img_neck}, img_bbox_head={img_bbox_head}, "
                    f"lidar_backbone={lidar_backbone}, fusion_module={fusion_module}, "
                    f"bbox_head={bbox_head}")
        
    def pprint_features(self,name, feats):
        """ Pretty print a list of features (or a single tensor). """
        if isinstance(feats, (list, tuple)):
            print(f"{name} has {len(feats)} levels:")
            for idx, feat in enumerate(feats):
                print(f"  {name}[{idx}] shape: {feat.shape}")
        else:
            print(f"{name} shape: {feats.shape}")

    # Example usage after extracting features:

    def extract_feat(self, img, points):
        """Extract features from image and LiDAR backbones."""
        # Image features
        if img is not None:
            img_feats = self.img_backbone(img)  # [B, 512, H, W]
            #self.pprint_features("Image backbone features", img_feats)
            img_feats = self.img_neck(img_feats )  # [B, 256, H, W]
            #f.pprint_features("Image neck features", img_feats)
            # [B, 256, H, W]
            if isinstance(img_feats, (list, tuple)):
                img_feats = img_feats[-1]  # Take last feature map
        else:
            raise ValueError("Image input is None")

        # LiDAR features
        if points is not None:
            # Pad points for batch processing
            # padded_points = pad_sequence(points, batch_first=True)  # [B, N_max, 4]
            # padded_points = padded_points.numpy()
            # voxels, coors, num_points = self.data_preprocessor(points)
            voxels = points['voxels']
            coors = points['coors'] 
            num_points = points['num_points'] 
            batch_size = int(coors[:, 0].max().item()) + 1  # since batch indices are 0,1,2,3

            voxel_feats = self.lidar_voxel_encoder(voxels, num_points, coors)  # Voxelized features
            canvas_feats = self.lidar_middle_encoder(voxel_feats, coors,  batch_size=batch_size)  # [B, C, H, W]
            backbone_feats = self.lidar_backbone(canvas_feats)  # List of features
            lidar_feats = self.lidar_neck(backbone_feats)
            
            logger.debug(f"voxel shape- {voxels.shape} num points shape- {num_points.shape} coors shape {coors.shape}")
            logger.debug(f"Unique batch indices in coors: {coors[:, 0].unique()}")
            logger.debug(f"voxel features shape: {voxel_feats.shape}")
            logger.debug(f"canvas features shape: {canvas_feats.shape}")
            logger.debug(f"backbone features {len(backbone_feats)} shape: {backbone_feats[0].shape}")
            logger.debug(f"lidar features {len(lidar_feats)} shape: {lidar_feats[0].shape}")
            #self.pprint_features("backbone features", backbone_feats)
            #self.pprint_features("lidar features", lidar_feats)
            # [B, 384, H, W]
            if isinstance(lidar_feats, (list, tuple)):
                lidar_feats = torch.cat(lidar_feats, dim=1)  # Concatenate FPN levels
        else:
            raise ValueError("LiDAR input is None")

        logger.debug(f"Image features shape: {img_feats.shape}")
        logger.debug(f"LiDAR features shape: {lidar_feats.shape}")
        return img_feats, lidar_feats
    
    def prepare_fusedfeature_for_bbhead(self, fused_feats):
        """Prepare fused features for bounding box head."""
        logger.debug(f"prepare Fused features shape: {fused_feats.shape}")
        # Reshape
        B, N, C = fused_feats.shape  # [B, 1024, 128]
        H = W = int(N ** 0.5)  # Assume square grid (32x32)
        fused_feats = fused_feats.permute(0, 2, 1).view(B, C, H, W)  # [B, 128, 32, 32]
        logger.debug(f"reshaped Fused features shape: {fused_feats.shape}")
        return fused_feats
    # in projects/myfusion/fusion/my_fusion.py (inside MyFusionTransformer)

    # A) In _forward: return the token memory (choose one)
    # def _forward(self, batch_inputs, batch_data_samples):
    #     img = batch_inputs.get('imgs', None)
    #     voxels = batch_inputs.get('voxels', None)

    #     img_feats, lidar_feats = self.extract_feat(img, voxels)
    #     # If you modified MBT to also return bn_out:
    #     # lidar_tokens, img_tokens, fused_tokens, bn_tokens = self.fusion_module(img_feats, lidar_feats)
    #     lidar_tokens, img_tokens, fused_tokens = self.fusion_module(img_feats, lidar_feats)

    #     # ---- Choose the memory to feed the token head ----
    #     token_memory = fused_tokens         # [B, N, E]  (simple, uses grid-sized memory)
    #     # token_memory = bn_tokens          # [B, T, E]  (if you returned bn_out and want compact memory)

    #     return token_memory  # (single tensor)

    def _forward(self, batch_inputs, batch_data_samples):
        """Forward pass for training and inference; return token memory for the head."""
        img    = batch_inputs.get('imgs', None)
        voxels = batch_inputs.get('voxels', None)

        img_feats, lidar_feats = self.extract_feat(img, voxels)

        # MBT can return 3 (lid,img,fused) or 4 (lid,img,fused,bn)
        outs = self.fusion_module(img_feats, lidar_feats)
        if not isinstance(outs, (tuple, list)):
            raise RuntimeError(f"fusion_module must return tuple/list, got {type(outs)}")

        if len(outs) == 4:
            lidar_tokens, img_tokens, fused_tokens, bn_tokens = outs
            token_memory = bn_tokens                  # <-- use bottlenecks [B,T,E]
        elif len(outs) == 3:
            lidar_tokens, img_tokens, fused_tokens = outs
            # Fallback if bn not returned: you can still run with fused tokens
            token_memory = fused_tokens               # <-- [B,N,E]
        else:
            raise RuntimeError(f"Unexpected number of outputs from fusion_module: {len(outs)}")

        # Do NOT reshape to [B,C,H,W]; head expects [B, T_or_N, E]
        return token_memory


    # B) In loss(): pass tokens directly
    # def loss(self, batch_inputs, batch_data_samples, **kwargs):
    #     token_memory = self._forward(batch_inputs, batch_data_samples)  # [B,N_or_T,E]
    #     losses = self.bbox_head.loss(token_memory, batch_data_samples)
    #     return losses

    # # C) In predict(): same idea
    # def predict(self, batch_inputs, batch_data_samples, **kwargs):
    #     token_memory = self._forward(batch_inputs, batch_data_samples)
    #     raw_predictions3d = self.bbox_head.predict(token_memory, batch_data_samples)
    #     predictions = self.add_pred_to_datasample(batch_data_samples, raw_predictions3d)
    #     return predictions

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        token_memory = self._forward(batch_inputs, batch_data_samples)   # [B,T,E]
        return self.bbox_head.loss(token_memory, batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        token_memory = self._forward(batch_inputs, batch_data_samples)   # [B,T,E]
        raw = self.bbox_head.predict(token_memory, batch_data_samples)
        return self.add_pred_to_datasample(batch_data_samples, raw)

    # def _forward(self, batch_inputs, batch_data_samples):
    #     """Forward pass for training and inference."""
    #     img = batch_inputs.get('imgs', None)
    #     lidar = batch_inputs.get('points', None)
    #     voxels = batch_inputs.get('voxels', None)

    #     logger.debug(f"img shape: {img.shape}")
    #     logger.debug(f"lidar size {len(lidar)} lidar[0] shape: {lidar[0].shape}")
    #     logger.debug(f"voxels keys {voxels.keys()} ")
    #     logger.debug(f"voxels[voxels] item shape: {voxels['voxels'].shape}")

    #     img_feats, lidar_feats = self.extract_feat(img, voxels)
    #     lidar_feats_flat, img_feats_flat, fused_feats = self.fusion_module(img_feats, lidar_feats)

    #     # reshape to [B,C,S,S]
    #     bbhead_input_lidar  = self.prepare_fusedfeature_for_bbhead(lidar_feats_flat)
    #     bbhead_input_img    = self.prepare_fusedfeature_for_bbhead(img_feats_flat)
    #     bbhead_input_fusion = self.prepare_fusedfeature_for_bbhead(fused_feats)

    #     # concat LiDAR + Fusion along channels -> [B, 256, S, S] (assuming each is 128ch)
    #     bbhead_input_lidar = torch.cat((bbhead_input_lidar, bbhead_input_fusion), dim=1)

    #     # return lists of feature maps (one FPN level)
    #     bbhead_input_lidar = [bbhead_input_lidar]
    #     bbhead_input_img   = [bbhead_input_img]

    #     return bbhead_input_lidar, bbhead_input_img, bbhead_input_fusion


    # def loss(self, batch_inputs, batch_data_samples, **kwargs):
    #     """Computes loss during training."""
    #     logger.debug(f"loss batch input keys{batch_inputs.keys()}")
    #     logger.debug(f"loss data samples {len(batch_data_samples)}")

    #     if 'imgs' not in batch_inputs or 'points' not in batch_inputs:
    #         raise ValueError("batch_inputs must contain 'imgs' and 'points'")
    #     if not batch_data_samples:
    #         raise ValueError("batch_data_samples cannot be empty")

    #     # Get fused LiDAR, fused image, and fusion tokens
    #     bbhead_input_lidar, bbhead_input_img, bbhead_input_fusion = self._forward(
    #         batch_inputs, batch_data_samples
    #     )

    #     # ✅ Already lists of feature maps (levels); pass through as-is
    #     lidar_bbox_head_input = bbhead_input_lidar
    #     # image_bbox_head_input = bbhead_input_img  # if using an image head

    #     # Optional sanity checks
    #     # assert isinstance(lidar_bbox_head_input, (list, tuple)) and lidar_bbox_head_input[0].ndim == 4

    #     losses = self.bbox_head.loss(lidar_bbox_head_input, batch_data_samples)
    #     return losses

    '''
    def _forward(self, batch_inputs, batch_data_samples):
        """Forward pass for training and inference."""
        
        img = batch_inputs.get('imgs', None)#batch_inputs['inputs']['img'].unsqueeze(0)
        lidar = batch_inputs.get('points', None)#batch_inputs['inputs']['points']
        voxels = batch_inputs.get('voxels', None)
     

        logger.debug(f"img shape: {img.shape}")
        logger.debug(f"lidar size {len(lidar)} lidar[0] shape: {lidar[0].shape}")
        logger.debug(f"voxels keys {voxels.keys()} ")
        logger.debug(f"voxels[voxels] item shape: {voxels['voxels'].shape}")
  
        #img_metas = [data.metainfo for data in batch_data_samples]

        img_feats, lidar_feats = self.extract_feat(img, voxels)
        lidar_feats_flat, img_feats_flat, fused_feats = self.fusion_module(img_feats, lidar_feats)
        
        # reshape
        bbhead_input_lidar = self.prepare_fusedfeature_for_bbhead(lidar_feats_flat)
        bbhead_input_img = self.prepare_fusedfeature_for_bbhead(img_feats_flat)
        bbhead_input_fusion = self.prepare_fusedfeature_for_bbhead(fused_feats)

        B, C_lidar, H_lidar, W_lidar = bbhead_input_lidar.shape  
        B_fusion, C_fusion, H_fusion, W_fusion = bbhead_input_fusion.shape
        bbhead_input_lidar = torch.cat((bbhead_input_lidar, bbhead_input_fusion), dim=1)  # LiDAR + Fusion
        # bbhead_input_img = torch.cat((bbhead_input_img, bbhead_input_fusion), dim=1),    # Image + Fusion
      
        # logger.info(f"_forward bbhead_input_lidar shape {bbhead_input_lidar.shape}")
        # logger.debug(f"_forward bbhead_input_img shape {bbhead_input_img.shape}")
        # logger.debug(f"_forward bbhead_input_fusion shape {bbhead_input_fusion.shape}")

        bbhead_input_lidar = [bbhead_input_lidar]
        bbhead_input_img = [bbhead_input_img]

        return bbhead_input_lidar, bbhead_input_img, bbhead_input_fusion #bbox_results,losses
    
    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """Computes loss during training."""
        logger.debug(f"loss batch input keys{batch_inputs.keys()}")
        logger.debug(f"loss data samples {len(batch_data_samples)}")
        # print(batch_data_samples.keys())
        if 'imgs' not in batch_inputs or 'points' not in batch_inputs:
            raise ValueError("batch_inputs must contain 'img' and 'points'")
        if not batch_data_samples:
            raise ValueError("batch_data_samples cannot be empty")

        # Process inputs through _forward
        # Get fused LiDAR, fused image, and fusion tokens from _forward
        bbhead_input_lidar, bbhead_input_img, bbhead_input_fusion = self._forward(batch_inputs, batch_data_samples)

        # Compute loss using bbox_head
        lidar_bbox_head_input = [bbhead_input_lidar]#[bbhead_input_lidar]#, bbhead_input_fusion]
        image_bbox_head_input = [bbhead_input_img]#[bbhead_input_img]#, bbhead_input_fusion]

        # logger.info(f"loss lidar_bbox_head_input shape {bbhead_input_lidar.shape}")
        # logger.info(f"loss lidar_bbox_head_input shape {lidar_bbox_head_input[0].shape}")
        # logger.info(f"loss image_bbox_head_input shape {image_bbox_head_input.shape}")

        # Pass fused LiDAR and fusion tokens to bbox_head
        losses = self.bbox_head.loss(lidar_bbox_head_input, batch_data_samples)        

        # # Compute 2D losses using the image bbox head (if defined)
        # if self.img_bbox_head is not None:
        #     img_losses = self.img_bbox_head.loss(image_bbox_head_input, batch_data_samples)
        #     # Merge 2D losses into the total loss dictionary
        #     losses.update(img_losses)
        return losses
        '''

    # def predict(self, batch_inputs, batch_data_samples, **kwargs):
    #     # Get fused LiDAR, fused image, and fusion tokens from _forward
    #     bbhead_input_lidar, bbhead_input_img, bbhead_input_fusion = self._forward(
    #         batch_inputs, batch_data_samples
    #     )

    #     # Ensure the head receives a list of feature maps (one level)
    #     if not isinstance(bbhead_input_lidar, (list, tuple)):
    #         lidar_bbox_head_input = [bbhead_input_lidar]
    #     else:
    #         lidar_bbox_head_input = bbhead_input_lidar

    #     # Optional sanity check
    #     # assert lidar_bbox_head_input[0].ndim == 4, f"Expected [B,C,H,W], got {lidar_bbox_head_input[0].shape}"

    #     raw_predictions3d = self.bbox_head.predict(lidar_bbox_head_input, batch_data_samples)
    #     predictions = self.add_pred_to_datasample(batch_data_samples, raw_predictions3d)
    #     return predictions


'''
    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        # Process inputs through the model
        #bbhead_input = self._forward(batch_inputs, batch_data_samples)
        # Get fused LiDAR, fused image, and fusion tokens from _forward
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        bbhead_input_lidar, bbhead_input_img, bbhead_input_fusion = self._forward(batch_inputs, batch_data_samples)

        logger.debug(f"predict batch input keys{batch_inputs.keys()}")
        # logger.info(f"predict batch data samples {len(batch_data_samples)}")
     
        lidar_bbox_head_input = bbhead_input_lidar#[bbhead_input_lidar]#, bbhead_input_fusion]
        image_bbox_head_input = bbhead_input_img#[bbhead_input_img]#, bbhead_input_fusion]

        raw_predictions3d = self.bbox_head.predict(lidar_bbox_head_input, batch_data_samples)
        # raw_predictions2d = self.img_bbox_head.predict(image_bbox_head_input, batch_data_samples)

        # logger.info(f"3d predictions: {raw_predictions3d}")
        # # Combine and reformat 2D and 3D predictions into a single list
        # predictions = []
        # for raw_pred_3d, raw_pred_2d, data_sample in zip(raw_predictions3d, raw_predictions2d, batch_data_samples):
        #     logger.debug(f"DataSample keys: {data_sample.keys()}")
        #     logger.debug(f"DataSample metainfo keys: {data_sample.metainfo.keys()}")

        #     # Create a Det3DDataSample object
        #     pred_sample = Det3DDataSample()
        #     pred_sample.set_metainfo(data_sample.metainfo)  # Copy metadata

        #     # Add 3D predictions
        #     pred_sample.pred_instances_3d = InstanceData(
        #         bboxes_3d=raw_pred_3d.bboxes_3d,
        #         scores_3d=raw_pred_3d.scores_3d,
        #         labels_3d=raw_pred_3d.labels_3d
        #     )

        #     # # Add 2D predictions
        #     pred_sample.pred_instances = InstanceData(
        #         # bboxes=raw_pred_2d.bboxes,  # 2D bounding boxes
        #         # scores=raw_pred_2d.scores,  # 2D scores
        #         # labels=raw_pred_2d.labels   # 2D labels
        #     )
        #     # Append the combined prediction to the list
        #     predictions.append(pred_sample)
        #     # logger.debug(f"Prediction {len(predictions)}: {predictions[-1]}")
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                 raw_predictions3d)
        return predictions
'''

@MODELS.register_module()
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, context):
        # query: [B, N_query, dim]
        # context: [B, N_context, dim]
        # key = value = context
        logger.debug(f" attention query shape: {query.shape}") #[4, 768, 256, 20]
        logger.debug(f" attention context shape: {context.shape}") #[4, 768, 256, 20]
       
        out, _ = self.attn(query, context, context)
        return out


@MODELS.register_module()
class TransformerFusionModule(nn.Module):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        channel_dim = d_model//2     # Half the dimension for image and LiDAR features
        self.img_proj = nn.Conv2d(256, channel_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(1*384, channel_dim, kernel_size=1)

        self.feature_size = 64
        # Number of fusion tokens
        num_tokens = self.feature_size * self.feature_size  # Number of fusion tokens, can be adjusted
        # Initialize fusion tokens as learnable parameters
        self.d_model = d_model  # Dimension of the model    
        assert num_tokens > 0, "num_tokens must be greater than 0"          

        embed_dim = channel_dim
        self.embed_dim = d_model
        # Initialize fusion tokens
        self.num_tokens = num_tokens
        self.fusion_tokens = nn.Parameter(torch.randn( num_tokens, embed_dim))  # [1,T, d_model]
        # If num_tokens = 64, and d_model = 256
        # You can reshape to [B, 256, 8, 8].
        self.cross_attn_lidar = CrossAttention(embed_dim, num_heads)
        self.cross_attn_image = CrossAttention(embed_dim, num_heads)
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, img_feats, lidar_feats):
        # Ensure the shapes are compatible
        #[seq_len, batch_size, d_model=128]
        batch_size, channels, height, width = img_feats.shape
        B, C_img, H_img, W_img = img_feats.shape
        B2, C_lidar, H_lidar, W_lidar = lidar_feats.shape
        assert B == B2, "Batch size mismatch between image and LiDAR features"
        
        logger.debug(f"img_feats shape: {img_feats.shape}") #[4, 256, 6, 20]
        logger.debug(f"lidar feature shape: {lidar_feats.shape}") #[4, 768, 256, 20]
        
        # === Step 1: Project both to same channel dimension ===
        img_feats_proj      = self.img_proj(img_feats)
        lidar_feats_proj    = self.lidar_proj(lidar_feats)

        logger.debug(f"Step 1: Project both to same channel dimension")
        logger.debug(f"img_feats_proj shape: {img_feats_proj.shape}")       #[4, d_model/2, 6, 20]
        logger.debug(f"lidar_feats_proj shape: {lidar_feats_proj.shape}")   #[4, d_model/2, 256, 256]
        
        # === Step 2: Downsample (adaptive pooling to smaller spatial size, e.g. 32x32) ===
        img_feats_ds    = F.adaptive_avg_pool2d(img_feats_proj, (self.feature_size, self.feature_size))   # [B, d_model/2, 32, 32]
        lidar_feats_ds  = F.adaptive_avg_pool2d(lidar_feats_proj, (self.feature_size, self.feature_size))  # same shape
        
        logger.debug(f"Step 2: Project both to same channel dimension")
        logger.debug(f"img_feats_downsampled shape: {img_feats_ds.shape}")       #([4, 128, 32, 32])
        logger.debug(f"lidar_feats_downsampled shape: {lidar_feats_ds.shape}")   #([4, 128, 32, 32])
        

        # === Step 3: Flatten features ===
        img_feats_flat = img_feats_ds.flatten(2).permute(0, 2, 1)      # [B, 1024, d_model/2]
        lidar_feats_flat = lidar_feats_ds.flatten(2).permute(0, 2, 1)  # [B, 1024, d_model/2]

        logger.debug(f"Step 3: Flatten features")
        logger.debug(f"img_feats_flat shape: {img_feats_flat.shape}")
        logger.debug(f"lidar_feats_flat shape: {lidar_feats_flat.shape}")

        # === Step 4: Concatenate ===
        # fused_feats = torch.cat([img_feats_ds, lidar_feats_ds], dim=1)  # [B, d_model, 32, 32]
        # Fuse features by concatenation (alternative fusion techniques can be explored)
        ##fused_feats = torch.cat([img_feats_proj, lidar_feats_proj], dim=1)  # (Seq, Batch, Channels) [992, 4, 256]
        
        # logger.debug(f"before encoder fused feature shape: {fused_feats.shape}") #[B, d_model=128, 32, 32]
        # x = fused_feats
        # B, C, H, W = x.shape
        # x = x.view(B, C, H*W)  # [B, C, 32x32]
        # x = x.permute(2, 0, 1)  
        # logger.debug(f"encoder input shape: {x.shape}")# [seq_len=1024=32x32, batch_size=4, d_model=128]
 
        # # Pass through Transformer encoder
        # fused_feats = self.encoder(x)
        # logger.debug(f"after encoder fused feature shape: {x.shape}") # [seq_len=1024=32x32, batch_size=4, d_model=128]

        
        # === Step 5: Initialize fusion tokens ===
        fusion_tokens = self.fusion_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, T, d_model]
        logger.debug(f"Step 5: Initialize fusion tokens")
        logger.debug(f"fusion_tokens shape: {fusion_tokens.shape}") # [B, T=2, d_model=256]
        
        # === Step 6: Fusion tokens attend to features ===
        fusion_tokens = fusion_tokens + self.cross_attn_lidar(fusion_tokens, lidar_feats_flat)
        fusion_tokens = self.norm_tokens(fusion_tokens)
        fusion_tokens = fusion_tokens + self.cross_attn_image(fusion_tokens, img_feats_flat)
        fusion_tokens = self.norm_tokens(fusion_tokens)

        # === Step 7: Feed-forward ===
        fusion_tokens = fusion_tokens + self.ffn(fusion_tokens)
        fusion_tokens = self.norm_ffn(fusion_tokens)
        fused_feats = fusion_tokens
        logger.debug(f"after encoder fused feature shape: {fused_feats.shape}") # [seq_len=1024=32x32, batch_size=4, d_model=128]

        # lidar_feats_flat = lidar_feats_flat + fusion_tokens
        # img_feats_flat = img_feats_flat + fusion_tokens

        return lidar_feats_flat, img_feats_flat, fused_feats
