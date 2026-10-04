import torch
from llm_from_scratch.model import TinyGPT, CausalSelfAttention

def tiny(): return TinyGPT(vocab_size=259,max_seq_len=16,d_model=32,n_heads=4,n_layers=2,d_ff=64,dropout=0.0)

def test_logits_and_loss_shapes():
    m=tiny(); x=torch.randint(0,256,(2,8)); logits,loss=m(x,x); assert logits.shape==(2,8,259); assert loss.ndim==0

def test_causal_attention_has_zero_future_probability():
    a=CausalSelfAttention(32,4,16,0.0); x=torch.randn(1,6,32); a(x,return_attention=True); w=a.last_attention; future=torch.triu(torch.ones(6,6,dtype=torch.bool),diagonal=1); assert torch.all(w[0,0][future]==0)

def test_generation_extends_sequence():
    m=tiny().eval(); x=torch.tensor([[65,66]]); y=m.generate(x,max_new_tokens=3,temperature=0); assert y.shape[1]==5
