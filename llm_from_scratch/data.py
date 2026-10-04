from pathlib import Path
import torch


def load_tokens(path, tokenizer):
    text = Path(path).read_text(encoding="utf-8")
    return torch.tensor(tokenizer.encode(text, add_eos=True), dtype=torch.long)


def split_tokens(tokens, train_fraction=0.9):
    n = int(len(tokens) * train_fraction)
    return tokens[:n], tokens[n:]


def get_batch(tokens, batch_size, seq_len, device):
    if len(tokens) <= seq_len:
        raise ValueError(f"Need more than {seq_len} tokens, got {len(tokens)}")
    starts = torch.randint(0, len(tokens) - seq_len, (batch_size,))
    x = torch.stack([tokens[i:i+seq_len] for i in starts])
    y = torch.stack([tokens[i+1:i+seq_len+1] for i in starts])
    return x.to(device), y.to(device)
