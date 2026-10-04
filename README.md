# LLM From Scratch — Tiny GPT-Style Transformer in PyTorch

A compact decoder-only Transformer implemented almost entirely from first principles with PyTorch primitives. The goal of this project is not to train a large GPT model; it is to demonstrate a clear understanding of the core building blocks behind modern autoregressive language models.

The project includes a byte-level tokenizer, learned positional embeddings, scaled dot-product causal attention, multi-head self-attention, Transformer blocks, training and evaluation loops, checkpointing, autoregressive generation, loss visualization, and attention visualization — without using `nn.MultiheadAttention` or pretrained Transformer libraries.

---

## Results

The model was trained on **Tiny Shakespeare** using a small decoder-only Transformer with **826,496 trainable parameters**.

| Item | Value |
|---|---:|
| Dataset | Tiny Shakespeare |
| Parameters | 826,496 |
| Vocabulary size | 259 |
| Context length | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Feed-forward dimension | 512 |
| Training steps | 3,000 |
| Batch size | 32 |
| Peak learning rate | 3e-4 |
| Full-corpus sampled evaluation loss* | 1.7679 |
| Full-corpus sampled perplexity* | 5.86 |

\*The current `evaluate.py` samples random windows from the complete input text. Therefore this number is useful as a model sanity check, but it should not be interpreted as a strict held-out test score. The training script itself tracks a separate validation split during optimization.

---

## Training Curve

![Training and validation loss](artifacts/loss_curve.png)

Both training and validation cross-entropy decrease consistently during training. The two curves remain close for most of the run, which indicates stable optimization and limited overfitting at this model scale.

The largest improvement occurs early in training, followed by slower convergence as the learning rate decays. Toward the end of the run, validation loss begins to flatten, suggesting that the model is approaching the useful capacity of this configuration on Tiny Shakespeare.

---

## Learned Causal Attention

![Layer 1 Head 1 causal attention](artifacts/attention.png)

The heatmap visualizes the attention matrix of **Layer 1, Head 1** for a short sequence.

The empty upper-right triangle is the most important feature: future tokens receive zero attention probability. For query position `t`, the model can attend only to positions `<= t`.

This is enforced by the causal mask before softmax:

```text
Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + causal_mask) V
```

where entries corresponding to future positions are set to `-inf`.

The non-uniform structure in the visible lower triangle also shows that the learned head is not simply averaging previous positions; it assigns different attention weights to different parts of the context.

---

## Example Generation

Prompt:

```text
ROMEO: O Juliet, my love
```

Generation settings:

```text
temperature = 0.7
top_k       = 30
max_tokens  = 500
```

Sample output:

```text
ROMEO: O Juliet, my love?

COMIO:
The sir, my and well so breford the stone.

KING LICEDWARD II:
I may lord's conterbeds rentlerss the caser;
That word you his row the king.

QUEEN ELIZABETH:
What is a love with and were the at may.

HENRXEN ELIO:
I good more here, whe suke, and how you shall knows to he many
Ow his fenser, in my mean lirde.

DUKE VINCENTIO:
I she the vere be a will and etremer:
The that he poy with hered, and that that the tame thats that
Of him with the befbring not to be carting one
That hath reike
```

The model learns several corpus-level patterns despite its small size: speaker labels, line breaks, Shakespeare-like punctuation, archaic lexical patterns such as `thou`, `hath`, and `lord`, and short-range sentence structure.

At the same time, it still produces invented words and weak long-range semantic coherence. This is expected for an ~0.83M-parameter byte-level language model trained on a relatively small corpus.

---

## Architecture

```text
Raw UTF-8 text
      │
      ▼
Byte-level Tokenizer
      │ token IDs
      ▼
Token Embedding ─────────┐
                         ├──► x_0
Position Embedding ──────┘
          │
          ▼
┌──────────────────────────────────┐
│        Transformer Block × 4     │
│                                  │
│ x ─► LayerNorm ─► Causal MHA     │
│ │                       │        │
│ └──────────────────────► +       │
│                         │        │
│        LayerNorm ─► MLP │        │
│             │           │        │
│             └──────────► +       │
└──────────────────────────────────┘
          │
          ▼
    Final LayerNorm
          │
          ▼
   Linear LM Head
    (weight tied)
          │
          ▼
  Next-token logits
          │
          ▼
 temperature / top-k sampling
          │
          ▼
 append token and repeat
```

This is a **decoder-only, autoregressive Transformer**, following the same high-level modeling principle used by GPT-style language models.

---

## What Is Implemented From Scratch?

- Reversible byte-level tokenizer
- Special tokens and fixed vocabulary
- Token embeddings
- Learned positional embeddings
- Query, key, and value projections
- Scaled dot-product attention
- Causal attention masking
- Multi-head self-attention
- Head splitting and concatenation
- Pre-LayerNorm Transformer blocks
- GELU feed-forward networks
- Residual connections
- Decoder-only language-model head
- Input/output embedding weight tying
- Cross-entropy next-token objective
- AdamW optimization
- Learning-rate warmup
- Cosine learning-rate decay
- Gradient clipping
- Training and validation loops
- Checkpoint save/load
- Perplexity calculation
- Temperature sampling
- Top-k sampling
- Attention-map visualization
- Loss-curve visualization
- Unit tests for model behavior

The implementation intentionally does **not** use `torch.nn.MultiheadAttention`, Hugging Face Transformer models, or pretrained weights.

---

## Model Configuration

The experiment shown in this README uses:

```yaml
model:
  vocab_size: 259
  max_seq_len: 128
  d_model: 128
  n_heads: 4
  n_layers: 4
  d_ff: 512
  dropout: 0.1

training:
  batch_size: 32
  max_steps: 3000
  learning_rate: 0.0003
  min_learning_rate: 0.00003
  warmup_steps: 100
  weight_decay: 0.1
  grad_clip: 1.0
```

---

## Repository Structure

```text
llm-from-scratch/
├── configs/
│   └── tiny.yaml
├── data/
│   ├── sample.txt
│   └── input.txt
├── llm_from_scratch/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── utils.py
├── scripts/
│   └── download_tinyshakespeare.py
├── tests/
├── artifacts/
│   ├── best.pt
│   ├── last.pt
│   ├── loss_curve.png
│   └── attention.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Download Tiny Shakespeare

```powershell
python scripts/download_tinyshakespeare.py
```

The default experiment expects the dataset at:

```text
data/input.txt
```

### 4. Train

```powershell
python -m llm_from_scratch.train --config configs/tiny.yaml
```

### 5. Evaluate

```powershell
python -m llm_from_scratch.evaluate --checkpoint artifacts/best.pt --data data/input.txt --batches 100
```

Example output:

```text
loss=1.7679 perplexity=5.86
```

Because evaluation batches are randomly sampled, repeated runs can differ slightly.

### 6. Visualize Attention

```powershell
python -m llm_from_scratch.evaluate --checkpoint artifacts/best.pt --data data/input.txt --batches 100 --plot-attention
```

This writes:

```text
artifacts/attention.png
```

### 7. Generate Text

```powershell
python -m llm_from_scratch.generate --checkpoint artifacts/best.pt --prompt "ROMEO: O Juliet, my love" --max-new-tokens 500 --temperature 0.7 --top-k 30
```

### 8. Run Tests

```powershell
pytest -v
```

---

## Training Objective

For a token sequence

```text
[x_0, x_1, ..., x_T]
```

the model receives

```text
[x_0, x_1, ..., x_(T-1)]
```

and is trained to predict

```text
[x_1, x_2, ..., x_T]
```

The loss is standard next-token cross entropy:

```text
L = -log p(x_(t+1) | x_0, ..., x_t)
```

Perplexity is computed as:

```text
perplexity = exp(cross_entropy_loss)
```

Lower perplexity means the model assigns higher probability to the observed next tokens.

---

## Multi-Head Self-Attention

Given hidden states `X`, each attention head computes:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

Then:

```text
scores = QK^T / sqrt(d_k)
```

A causal mask removes access to future positions:

```text
scores[i, j] = -inf   when j > i
```

Finally:

```text
weights = softmax(scores)
output  = weights V
```

Multiple heads perform this operation in parallel, their outputs are concatenated, and a final projection mixes information across heads.

---

## Why a Byte-Level Tokenizer?

A byte-level tokenizer keeps this repository self-contained and makes tokenization easy to inspect.

Advantages:

- fixed vocabulary
- no unknown-token problem for UTF-8 input
- fully reversible
- no external tokenizer dependency
- simple enough to implement from scratch

Trade-off:

- byte sequences are usually longer than BPE/SentencePiece sequences
- the model must learn multi-byte and subword structure indirectly
- a small model is therefore more likely to generate malformed or invented words

This trade-off is visible in the generated Shakespeare-like text above.

---

## Design Choices

### Pre-LayerNorm

Layer normalization is applied before the attention and MLP sublayers. This generally provides stable optimization and keeps the residual path simple.

### Learned Positional Embeddings

The model learns one positional vector for each location up to the configured context length. This keeps the implementation transparent, although it does not naturally extrapolate beyond the trained maximum sequence length.

### Weight Tying

The token embedding matrix is reused by the output language-model head. This reduces the total parameter count and links the input and output token representations.

### Top-k + Temperature Sampling

`temperature` controls the sharpness of the next-token distribution, while `top-k` restricts sampling to the `k` most likely tokens.

Lower temperature generally produces safer and more repetitive samples; higher temperature produces more diverse but noisier text.

---

## What the Experiment Demonstrates

This small experiment is intentionally limited in scale, but it demonstrates that the implementation can learn meaningful autoregressive structure from raw text.

The training curve shows successful optimization, the attention map verifies correct causal masking, and generated samples reproduce important structural properties of the training corpus.

The remaining language errors are also informative: they highlight the impact of limited model capacity, byte-level tokenization, short context length, and a small training corpus.

---

## Limitations

This repository is an educational implementation, not a production LLM.

Current limitations include:

- fewer than one million parameters
- byte-level rather than subword tokenization
- only 128 tokens of context
- learned absolute positional embeddings
- no KV cache during generation
- no mixed-precision training pipeline
- no distributed training
- no FlashAttention
- no strict standalone test-set evaluation in the current evaluator
- training on a small single-domain corpus

These limitations are natural directions for future experiments rather than hidden shortcomings.

---

## Possible Extensions

Potential follow-up experiments include:

1. Implement BPE tokenization from scratch and compare it with byte-level tokenization.
2. Replace learned positional embeddings with RoPE.
3. Add KV caching for faster autoregressive inference.
4. Compare context lengths of 64, 128, and 256 tokens.
5. Compare different numbers of heads at approximately fixed parameter count.
6. Add mixed-precision training.
7. Implement knowledge distillation between larger and smaller Transformers.
8. Add pruning and quantization experiments.
9. Measure parameter count, memory usage, throughput, and FLOPs.
10. Visualize and compare attention behavior across layers and heads.

These extensions make the repository a useful starting point for work on **LLMs, model compression, and knowledge distillation**.

---

## Reproducibility

The default configuration uses:

```text
seed = 42
```

Checkpoints are stored in `artifacts/`. `best.pt` contains the best checkpoint selected during training, while `last.pt` stores the final training state.

For reproducible reporting, record the PyTorch version, GPU model, CUDA version, seed, configuration, and exact dataset revision alongside experiment results.

---

## License

MIT License — see `LICENSE`.
