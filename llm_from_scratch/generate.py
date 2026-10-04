import argparse, yaml, torch
from .model import TinyGPT
from .tokenizer import ByteTokenizer
from .utils import device

def load_model(path, dev):
    ckpt=torch.load(path,map_location=dev); model=TinyGPT(**ckpt["config"]["model"]).to(dev); model.load_state_dict(ckpt["model"]); model.eval(); return model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--prompt",default="To be"); p.add_argument("--max-new-tokens",type=int,default=200); p.add_argument("--temperature",type=float,default=.8); p.add_argument("--top-k",type=int,default=40); a=p.parse_args()
    dev=device(); tok=ByteTokenizer(); model=load_model(a.checkpoint,dev); ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long,device=dev); out=model.generate(ids,a.max_new_tokens,a.temperature,a.top_k); print(tok.decode(out[0].tolist()))
if __name__=="__main__": main()
