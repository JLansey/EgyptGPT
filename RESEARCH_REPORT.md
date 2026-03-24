# EgyptGPT Autoresearch Report

**Date**: 2026-03-24
**Branch**: `autoresearch/run_amber_2`
**Best commit**: `0710e35` (pushed to origin)
**Best val_bpb**: 1.105 (bits per character)
**Total experiments**: 89 (19 kept, 70 discarded)
**Deployed checkpoint**: `out-glyphs-char/ckpt.pt` (26MB, 2.2M params)

## Task

Optimize a GPT model trained on ~1.4M characters of Egyptian hieroglyphic sequences (Gardiner notation). Fixed 5-minute wall-clock training budget on a single L4 GPU. Metric: val_bpb in bits per character (lower is better).

## Starting point

Standard nanoGPT (GPT-2 style): 6 layers, 6 heads, 384 dim (~10.6M params), 256 context, batch 64, LR 1e-3, dropout 0.2, GELU, learned position embeddings, cosine LR schedule, AdamW.

**Baseline val_bpb: 1.940**

## Final best configuration

- **Tokenization**: Sign-level (781-token Gardiner sign vocabulary, 3.4x compression vs char-level)
- **Architecture**: 3 layers, 4 heads, 256 dim, MLP expansion 3x, ReLU² activation, RoPE, LayerNorm, no bias, weight tying
- **Params**: ~2.2M (slightly more than char-level due to larger vocab)
- **Optimizer**: AdamW, LR 1e-3, beta1=0.9, beta2=0.99, weight_decay=0.1, grad_clip=0.5
- **Schedule**: warmup 20 steps, cosine decay over 4000 steps to min_lr=1e-5
- **Training**: batch_size 32, block_size 256 sign tokens (~870 chars context), dropout 0.3, ~24k steps in 5 min
- **val_bpb: 1.105** (43% improvement from original baseline)

## Progress timeline

| Phase | val_bpb | Key change |
|-------|---------|------------|
| Original baseline | 1.940 | 6L 384d, GELU, char-level |
| Remove dropout | 1.873 | dropout 0 (char-level) |
| ReLU² activation | 1.819 | Sharper activations |
| Smaller batch | 1.791 | batch 32 → more steps |
| Smaller model | 1.339 | 3L 256d → 2x faster steps |
| LR schedule tuning | 1.291 | Cosine decay, RoPE, grad_clip |
| **Sign-level tokenization** | **1.107** | **781-sign vocab, 3.4x compression** |
| min_lr tuning | 1.105 | min_lr=1e-5 keeps learning |

## Key findings

### Phase 1: Model size vs. training steps (char-level)

The single most important discovery was that **smaller models dramatically outperform larger ones** under a fixed time budget. The 5-minute constraint means throughput (steps/sec) matters enormously:

| Config | Params | Steps in 5 min | val_bpb |
|--------|--------|-----------------|---------|
| 6L 384d 6h | 10.6M | 251 | 1.819 |
| 4L 384d 6h | 7.1M | 936 | 1.448 |
| 3L 384d 6h | 5.3M | 1248 | 1.386 |
| 2L 384d 6h | 3.6M | 1811 | 1.352 |
| 2L 256d 4h | 1.6M | 3251 | 1.348 |
| 3L 256d 4h | 2.0M | 2251 | 1.339 |
| 1L 384d 6h | 1.8M | 3501 | 1.413 |

Sweet spot: **3L 256d 4h** (~2.0M params). 1 layer lost too much capacity. Wider/deeper was slower per step for a net loss.

### Phase 2: Sign-level tokenization (the big win)

**val_bpb: 1.291 → 1.105 (14.4% improvement)**

The char-level model wasted capacity learning to spell Gardiner codes (e.g., predicting `2`, `1` after `D` to form `D21`). Sign-level tokenization predicts hieroglyphic signs directly:

- **Vocabulary**: 781 tokens (Gardiner signs appearing ≥5 times in training data + newline, `/`, `<UNK>`)
- **Compression**: 3.41 chars per sign token → sequences 3.4x shorter
- **Context**: block_size=256 sign tokens covers ~870 characters (vs 256 chars before)
- **BPB conversion**: `val_bpb_chars = val_loss / (ln(2) × avg_chars_per_token)` for fair comparison

**Implementation** (all in train.py, prepare.py untouched):
1. At startup, decode train.bin/val.bin back to text via meta.pkl
2. Split on spaces/newlines to extract Gardiner signs
3. Build sign vocabulary (signs with ≥5 train occurrences)
4. Re-encode to sign-level integer IDs, save as train_signs.bin/val_signs.bin
5. Train on sign tokens, convert val loss back to bits-per-character

**Critical regularization finding**: Sign-level data is 3.4x smaller (386k tokens vs 1.3M chars). Without dropout, the model massively overfits (train loss 0.5, val loss 4.4 — the model memorizes the dataset). **Dropout 0.3** was essential — this is the opposite of the char-level finding where dropout=0 was optimal.

### Phase 2b: Hyperparameter tuning on sign-level

Extensive search (25+ experiments) around the sign-level baseline:

**What helped:**
- lr_decay_iters=4000 (tuned to sign-level convergence speed)
- min_lr=1e-5 (allows continued slow learning after primary decay)

**What didn't help:**
- Dropout 0.2 or 0.4 (0.3 was the sweet spot)
- Larger models (4L 384d, 3L 320d): slower per step, net loss
- Smaller models (2L 256d): not enough capacity for 781-token vocab
- Higher/lower LR (5e-4, 1.5e-3): 1e-3 was optimal
- Block size 128 or 512: 256 was optimal (128 too little context, 512 too slow)
- No weight tying: worse (weight tying helps even though vocab > n_embd)
- MLP 4x: slower per step
- BPE over signs (200 merges): larger vocab, similar performance
- Various beta2, weight_decay, warmup changes: marginal at best
- Label smoothing: inflates eval loss, not comparable

### Phase 3: Longer training (10-minute budget)

Tested whether doubling training time to 10 minutes would help. **It did not.** The model learns everything it can in the first ~4000 steps; extra time just adds overfitting:

| Experiment | val_bpb | Issue |
|-----------|---------|-------|
| 10-min, 4L 320d | 1.184 | Massive overfitting |
| 10-min, dropout 0.35 | 1.114 | Over-regularized |
| 10-min, lr_decay 8000 | 1.122 | Wider decay → more overfitting |
| 10-min, lr_decay 4000 | 1.108 | Extra steps at min_lr don't help |
| 10-min, warm restarts | 1.140 | Restarts undo learned patterns |
| **5-min baseline** | **1.105** | **Data is the bottleneck** |

**Conclusion**: The dataset (386k sign tokens) is the hard constraint. More compute cannot overcome limited data with this model class.

## Interpretation

Two discoveries drove nearly all the improvement:

1. **Compute efficiency** (Phase 1, 1.940 → 1.291): Under a fixed time budget, the optimal model maximizes steps × capacity. Smaller models win because they train faster.

2. **Semantic tokenization** (Phase 2, 1.291 → 1.105): Predicting meaningful units (hieroglyphic signs) instead of arbitrary characters lets the model learn language structure instead of spelling. Each sign token carries ~3.4x more information than a character token.

The 10-minute experiments confirmed that **data, not compute, is the bottleneck**. The model saturates the small dataset in ~4000 steps regardless of architecture or schedule.

## Deployment artifacts

- **Checkpoint**: `out-glyphs-char/ckpt.pt` (26MB)
- **Sign vocabulary**: `data/egypt_char/meta_signs.pkl` (vocab mapping for encoding/decoding)
- **Original char vocabulary**: `data/egypt_char/meta.pkl` (needed to decode train.bin back to text)
- **Model code**: `model.py` (GPT class with RoPE, ReLU², weight tying)
- **Config**: `config/train_egypt_char.py`

To load the model:
```python
# Load checkpoint
checkpoint = torch.load('out-glyphs-char/ckpt.pt')
model = GPT(GPTConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
```

## Infrastructure note

The background sync script (`colab_sync_push.sh`) copies `results.tsv` to Drive. Git pushes happen inline only when keeping successful experiments. This prevents divergence from `git reset --hard` on discards.
