import json, math, random
from pathlib import Path
import torch

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_json(obj, path): Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
