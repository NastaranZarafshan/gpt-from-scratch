import argparse, math, torch
import matplotlib.pyplot as plt
from .data import load_tokens, get_batch
from .generate import load_model
from .tokenizer import ByteTokenizer
from .utils import device

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--data",required=True); p.add_argument("--batches",type=int,default=50); p.add_argument("--plot-attention",action="store_true"); a=p.parse_args()
    dev=device(); tok=ByteTokenizer(); model=load_model(a.checkpoint,dev); tokens=load_tokens(a.data,tok); losses=[]
    for _ in range(a.batches):
        x,y=get_batch(tokens,16,model.max_seq_len,dev); _,loss=model(x,y); losses.append(loss.item())
    loss=sum(losses)/len(losses); print(f"loss={loss:.4f} perplexity={math.exp(min(loss,20)):.2f}")
    if a.plot_attention:
        n=min(32,len(tokens)-1,model.max_seq_len); x=tokens[:n].unsqueeze(0).to(dev); model(x,return_attention=True); att=model.blocks[0].attn.last_attention[0,0].cpu()
        plt.figure(figsize=(7,6)); plt.imshow(att); plt.xlabel("key position"); plt.ylabel("query position"); plt.title("Layer 1, Head 1 causal attention"); plt.colorbar(); plt.tight_layout(); plt.savefig("artifacts/attention.png",dpi=160); plt.close()
if __name__=="__main__": main()
