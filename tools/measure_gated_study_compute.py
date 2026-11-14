"""Matched real-shape camera/fusion FLOPs and fusion latency; no accuracy eval.

Reports supported arithmetic with every unsupported operator listed. Sparse
LiDAR encoding, voxelization and postprocessing are explicitly outside this
component comparison; no complete-detector FLOPs claim is made.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import hashlib
import torch
from torch import nn
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.jit_handles import get_shape
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS, MODELS
from projects.myfusion.tools.gated_study_plan import study_arms
from projects.myfusion.tools.run_gated_studies import ROOT,REFERENCES,sha,write

class Camera(nn.Module):
    def __init__(self,m):super().__init__();self.model=m
    def forward(self,x):return self.model.extract_img_feat(x)

class Fusion(nn.Module):
    def __init__(self,m,metadata):super().__init__();self.module=m;self.metadata=metadata
    def forward(self,image,lidar):return self.module(image,lidar,**self.metadata)['bev_features']

def sdpa(inputs,outputs):
    q,k=get_shape(inputs[0]),get_shape(inputs[1]);batch=1
    for d in q[:-2]:batch*=d
    # Two matrix products, with one fused multiply-add counted as one FLOP.
    return 2*batch*q[-2]*k[-2]*q[-1]

def native_mha(inputs,outputs):
    q,k,v=[get_shape(x) for x in inputs[:3]]
    assert q==k==v, 'Native self-attention shape contract changed'
    batch,n,d=q
    return batch*(4*n*d*d + 2*n*n*d)

def analyze(module,inputs):
    analyzer=FlopAnalyzer(module,inputs)
    analyzer.set_op_handle('aten::scaled_dot_product_attention',sdpa)
    analyzer.set_op_handle('aten::_native_multi_head_attention',native_mha)
    analyzer.unsupported_ops_warnings(False);analyzer.uncalled_modules_warnings(False);analyzer.tracer_warnings('none')
    with torch.no_grad():total=analyzer.total()
    return dict(counted_GFLOPs=total/1e9,by_operator=dict(analyzer.by_operator()),
        unsupported_operators=dict(analyzer.unsupported_ops()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--dataset',choices=['kitti','nuscenes','both'],default='both');args=p.parse_args()
    out=Path(args.output);assert not out.exists(),out
    torch.set_num_threads(4);torch.manual_seed(0)
    if hasattr(torch.backends,'mha'):torch.backends.mha.set_fastpath_enabled(False)
    report=dict(status='running',FLOP_convention='One fused multiply-add = one FLOP; native MHA includes QKV/output projections and both attention matrix products',
        scope='Camera backbone/FPN and fusion module, including computed auxiliary logits. Excludes voxelization, LiDAR encoder, detector head and postprocessing.',
        note='Architecture counts; new arms are not trained accuracy results. Unsupported arithmetic is reported without silently treating it as a complete total.',
        gpu=torch.cuda.get_device_name(),pytorch=torch.__version__,datasets={})
    try:
        for dataset in (['kitti','nuscenes'] if args.dataset=='both' else [args.dataset]):
            cfg=Config.fromfile(REFERENCES[dataset]);m=init_model(cfg,checkpoint=None,device='cuda:0')
            parent=REFERENCES[dataset].parent
            cp=next(parent.glob('best*epoch_18.pth')) if dataset=='kitti' else next(parent.glob('best*iter_110000.pth'))
            actual=sha(cp)
            if dataset=='kitti':
                records=json.loads((ROOT/'work_dirs/part_b_kitti_matched_comparison_20261109/comparison.json').read_text())
                assert actual==next(x for x in records['models'] if x['name']=='gated_b8')['checkpoint_sha256']
            else:assert actual=='691beda2e84cf33c64ebf26fc513b1a20717dfd4d48443c728aed3628e195639'
            m.load_state_dict(torch.load(cp,map_location='cpu')['state_dict'],strict=True);m.eval()
            data=DATASETS.build(cfg.test_dataloader.dataset)
            processed=m.data_preprocessor(pseudo_collate([data[0]]),training=False)
            inputs=processed['inputs'];samples=processed['data_samples'];imgs=inputs['imgs']
            with torch.no_grad():
                im=m.extract_img_feat(imgs)
                if dataset=='kitti':lidar=m.extract_pts_feat(inputs['voxels'],1)
                else:lidar=m.to_sampler_bev(m.pts_neck(m.pts_backbone(m.extract_pts_feat(inputs)))[0])
                metadata=m._projection_metadata(samples,imgs)
            rec=dict(reference_checkpoint=str(cp),checkpoint_sha256=actual,strict_loading=True,
                sample_metainfo={k:str(samples[0].metainfo.get(k)) for k in ['sample_idx','lidar_path','img_path']},
                image_shape=list(imgs.shape),camera_feature_shape=list(im.shape),lidar_feature_shape=list(lidar.shape),
                camera_encoder=analyze(Camera(m),(imgs,)),arms=[])
            variants=[dict(name='gated_b8_reference',changes={})]+study_arms(dataset)
            if dataset=='kitti':variants.append(dict(name='full_attention',changes={}))
            for arm in variants:
                fc=deepcopy(cfg.model.fusion_module)
                if arm['name']=='full_attention':
                    from projects.myfusion.fusion.full_attention_bev import FullAttentionBEVFusion
                    fc=Config.fromfile(ROOT/'work_dirs/part_b_run4_full_attention_20261109/run4_resolved_config.py').model.fusion_module
                else:fc.update(arm['changes'])
                f=MODELS.build(fc).cuda().eval();wrapper=Fusion(f,metadata).eval()
                result=dict(name=arm['name'],changes=arm['changes'],fusion_parameters=sum(p.numel() for p in f.parameters()),**analyze(wrapper,(im,lidar)))
                with torch.no_grad():
                    for _ in range(10):wrapper(im,lidar)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    for _ in range(50):wrapper(im,lidar)
                    torch.cuda.synchronize();result['fusion_latency_ms']=(time.perf_counter()-start)*1000/50
                    result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
                rec['arms'].append(result);report['datasets'][dataset]=rec;write(out,report)
                print(dataset,arm['name'],result['counted_GFLOPs'],result['unsupported_operators'],flush=True)
                del f,wrapper;torch.cuda.empty_cache()
            del m,im,lidar,processed;torch.cuda.empty_cache()
        report['status']='complete'
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:write(out,report)

if __name__=='__main__':main()
