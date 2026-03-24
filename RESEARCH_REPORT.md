# EgyptGPT Autoresearch Report

**Date**: 2026-03-23
**Branch**: `autoresearch/run_amber_2`
**Best commit**: `b37c769` (pushed to origin)
**Best val_bpb**: 1.290659
**Total experiments**: 59 (17 kept, 42 discarded)

## Task

Optimize a character-level GPT model trained on ~1.4M characters of Egyptian hieroglyphic sequences (Gardiner notation, 44-char vocab). Fixed 5-minute wall-clock training budget on a single GPU. Metric: val_bpb (lower is better).

## Starting point

Standard nanoGPT (GPT-2 style): 6 layers, 6 heads, 384 dim (~10.6M params), 256 context, batch 64, LR 1e-3, dropout 0.2, GELU, learned position embeddings, cosine LR schedule, AdamW.

**Baseline val_bpb: 1.940**

## Current best configuration

- **Architecture**: 3 layers, 4 heads, 256 dim, MLP expansion 3x, ReLU² activation, RoPE, LayerNorm, no bias
- **Params**: ~2.0M
- **Optimizer**: AdamW, LR 1e-3, beta1=0.9, beta2=0.99, weight_decay=0.1, grad_clip=0.5
- **Schedule**: warmup 20 steps, cosine decay over 2200 steps to min_lr=0
- **Training**: batch_size 32, block_size 256, ~2251 steps in 5 min
- **val_bpb: 1.291** (33% improvement from baseline)

## Key findings

### The dominant insight: model size vs. training steps tradeoff

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

The sweet spot was **3L 256d 4h** (~2.0M params). Going to 1 layer lost too much capacity. Going wider (512d) or deeper (4+ layers) was slower per step for a net loss.

### What worked

1. **Remove dropout** (1.940 → 1.873): With only 5 min of training, dropout wastes capacity. No overfitting risk in this regime.

2. **ReLU² activation** (1.873 → 1.819): Replacing GELU with `relu(x).square()` gave a big boost. This is known to help small models — it creates sharper, sparser activations.

3. **Smaller batch size** (1.819 → 1.791 → 1.653): Reducing batch from 64→32 doubled training steps. The gradient noise from smaller batches was acceptable; batch 16 was too noisy, batch 24 was marginal. 32 was the sweet spot.

4. **Smaller model** (1.653 → 1.339): As described above, the biggest lever. Went from 10.6M to 2.0M params.

5. **MLP 3x expansion** (1.339 → 1.332): Reducing MLP from 4x to 3x gave a slight gain — faster steps with minimal capacity loss. 2x was too small.

6. **Proper cosine LR decay** (1.332 → 1.319): The original config had lr_decay_iters=5000 (later 50000) but only ~2500 actual steps, so the LR barely decayed — essentially constant. Setting lr_decay_iters=2500 to match actual training length made the cosine schedule actually meaningful.

7. **LR 1e-3 with decay** (1.319 → 1.318): With proper cosine decay, LR 1e-3 slightly beat 5e-4. The decay prevents late-training instability that flat LR caused.

8. **RoPE** (1.318 → 1.318): Rotary positional embeddings replacing learned position embeddings. Marginal gain but also removes the wpe parameter table — a simplification win.

9. **Grad clip 0.5** (1.318 → 1.304): Tighter clipping (from default 1.0) stabilized training noticeably. 0.3 was equivalent — 0.5 kept as simpler.

10. **min_lr = 0** (1.304 → 1.297): Decaying LR all the way to zero (instead of LR/10) helped. Common in modern training recipes.

11. **lr_decay_iters = 2200** (1.297 → 1.291): Slightly shorter than the ~2500 actual steps, so the LR reaches zero before training ends. The last ~300 steps at near-zero LR act as fine-tuning. 1800 was too aggressive (not enough high-LR learning).

### What didn't work

- **RMSNorm** (1.872 vs 1.819): Hurt significantly. LayerNorm's mean-subtraction may matter at this scale.
- **QK-Norm** (1.901 vs 1.819): Added overhead without benefit. Likely unnecessary at this small scale.
- **Softcap logits** (1.842 vs 1.819): Slight hurt. The 44-char vocab doesn't produce extreme logits.
- **SwiGLU** (1.363 vs 1.339): Extra linear layer added params/compute. ReLU² is simpler and faster.
- **Larger batch / grad accum** (always worse): Fewer optimization steps always dominated any gradient quality improvement.
- **Larger context (512)** (2.582 vs 1.819): Catastrophic — doubled VRAM, halved throughput.
- **Disabling torch.compile** (1.891 vs 1.819): Compile overhead was already amortized; the compiled model was faster per step.
- **Lower beta2 (0.95)** (1.354 vs 1.332): Adam's second moment benefits from high beta2 with small batch.
- **Lower beta1 (0.8)** (1.325 vs 1.304): Standard momentum (0.9) was better.
- **Weight decay 0.01 or 0.05** (both worse): Default 0.1 was optimal — some regularization still helps.
- **8 heads at 256d** (1.325 vs 1.304): Head dim of 32 was too small. 4 heads (dim 64) was better.
- **Wider models** (320d, 512d): Always slower per step for a net loss at this time budget.

## Interpretation

Under a fixed time budget, this is fundamentally a **compute-efficiency** problem, not a capacity problem. The optimal model is the smallest one that can still represent the data well, trained for as many steps as possible. With ~2.0M params and 2200+ steps, we're well past the point where the model "fits" the data distribution — we're fine-tuning a converged-enough model.

The LR schedule was the second most impactful finding: getting the cosine decay to actually play out over the training window (rather than being truncated at 10% of the planned schedule) gave a significant boost.

## Important note on faster GPU

When moving to a faster GPU, the step count will increase significantly. This changes the optimal tradeoffs:
- **Model size should increase** — more steps means larger models can be trained more fully
- **lr_decay_iters must be re-tuned** — it should match the new expected step count
- **Batch size may need adjustment** — more VRAM available, but step count still matters
- **The baseline should be re-established** on the new hardware first

## Untested ideas for next session

- **Parallel attn+MLP (GPT-J style)** — removes one LN per block, computes attn and MLP in parallel. Could be faster per step.
- **Muon optimizer** for matrix parameters — could converge faster than AdamW
- **Higher LR for embeddings** — token embeddings might benefit from 10x higher LR
- **Label smoothing** in the cross-entropy loss
- **Different init scales** — 0.02 std might not be optimal for this small model
- **Try n_embd=288, n_head=4** (head_dim=72) — slightly wider might be in the sweet spot
- **Reduce eval_iters** to e.g. 100 — less accurate eval but more training time
- **Stochastic weight averaging** in the final steps

## Infrastructure note

The background sync script (`colab_sync_push.sh`) was modified to only copy `results.tsv` to Drive — it no longer does `git push`. Pushes happen inline in the experiment loop only when keeping a successful experiment. This prevents divergence from `git reset --hard` on discarded experiments.
