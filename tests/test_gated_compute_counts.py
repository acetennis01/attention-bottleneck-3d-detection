import torch
from torch import nn
from projects.myfusion.tools.measure_gated_study_compute import analyze

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.MultiheadAttention(16,4,batch_first=True)
    def forward(self,q,k):return self.layer(q,k,k,need_weights=False)[0]

def test_fused_self_attention_includes_projections_and_matrix_products():
    m=Attention().eval();q=torch.randn(2,5,16)
    r=analyze(m,(q,q))
    expected=2*(4*5*16*16+2*5*5*16)
    assert abs(r['counted_GFLOPs']*1e9-expected)<.01
    assert 'aten::_native_multi_head_attention' not in r['unsupported_operators']

def test_cross_attention_includes_both_query_and_context_lengths():
    m=Attention().eval();q=torch.randn(2,5,16);k=torch.randn(2,9,16)
    r=analyze(m,(q,k))
    expected=2*((2*5+2*9)*16*16+2*5*9*16)
    assert abs(r['counted_GFLOPs']*1e9-expected)<.01
    assert 'aten::scaled_dot_product_attention' not in r['unsupported_operators']
