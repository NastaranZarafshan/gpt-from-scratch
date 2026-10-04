import argparse, math
from pathlib import Path
import yaml, torch
import matplotlib.pyplot as plt
from .tokenizer import ByteTokenizer
from .data import load_tokens, split_tokens, get_batch
from .model import TinyGPT
from .utils import set_seed, device, count_parameters, save_json

@torch.no_grad()
def estimate(model, train, val, cfg, dev):
    model.eval(); out = {}
    for name, data in [("train", train), ("val", val)]:
        losses=[]
        for _ in range(cfg["training"]["eval_batches"]):
            x,y=get_batch(data,cfg["training"]["batch_size"],cfg["model"]["max_seq_len"],dev)
            _,loss=model(x,y); losses.append(loss.item())
        out[name]=sum(losses)/len(losses)
    model.train(); return out

def lr_for(step, cfg):
    tr=cfg["training"]; warm=tr["warmup_steps"]; total=tr["max_steps"]
    if step < warm: return tr["learning_rate"]*(step+1)/max(1,warm)
    ratio=(step-warm)/max(1,total-warm); coeff=0.5*(1+math.cos(math.pi*ratio))
    return tr["min_learning_rate"]+coeff*(tr["learning_rate"]-tr["min_learning_rate"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/tiny.yaml"); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); set_seed(cfg["seed"]); dev=device()
    tok=ByteTokenizer(); tokens=load_tokens(cfg["data_path"],tok); train,val=split_tokens(tokens,cfg["train_fraction"])
    model=TinyGPT(**cfg["model"]).to(dev); tr=cfg["training"]
    opt=torch.optim.AdamW(model.parameters(),lr=tr["learning_rate"],weight_decay=tr["weight_decay"])
    outdir=Path(cfg["output_dir"]); outdir.mkdir(parents=True,exist_ok=True); history=[]; best=float("inf")
    print(f"device={dev} parameters={count_parameters(model):,} train_tokens={len(train):,} val_tokens={len(val):,}")
    for step in range(tr["max_steps"]):
        lr=lr_for(step,cfg)
        for g in opt.param_groups:g["lr"]=lr
        x,y=get_batch(train,tr["batch_size"],cfg["model"]["max_seq_len"],dev)
        _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),tr["grad_clip"]); opt.step()
        if step % tr["eval_interval"]==0 or step==tr["max_steps"]-1:
            m=estimate(model,train,val,cfg,dev); row={"step":step,"lr":lr,**m,"val_perplexity":math.exp(min(m["val"],20))}; history.append(row)
            print(row)
            state={"model":model.state_dict(),"config":cfg,"step":step,"metrics":row}
            torch.save(state,outdir/"last.pt")
            if m["val"]<best: best=m["val"]; torch.save(state,outdir/"best.pt")
    save_json(history,outdir/"metrics.json")
    plt.figure(figsize=(7,4)); plt.plot([r["step"] for r in history],[r["train"] for r in history],label="train"); plt.plot([r["step"] for r in history],[r["val"] for r in history],label="validation"); plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.legend(); plt.tight_layout(); plt.savefig(outdir/"loss_curve.png",dpi=160); plt.close()

if __name__=="__main__": main()
