# EgyptGPT autoresearch

Autonomous research agent for optimizing a character-level GPT model trained on Egyptian hieroglyphic sequences (Gardiner notation).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `train.py` — training loop, data loading, evaluation. You may modify this.
   - `model.py` — GPT model architecture. You may modify this.
   - `config/train_egypt_char.py` — hyperparameters for Egypt training. You may modify this.
   - `data/egypt_char/prepare.py` — data preparation. **Read-only, do not modify.**
4. **Verify data exists**: Check that `data/egypt_char/train.bin` and `data/egypt_char/val.bin` exist. If not, tell the human to run `python data/egypt_char/prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context

This is a character-level language model trained on Egyptian hieroglyphic sequences in Gardiner notation. The data:
- ~1.4M characters of cleaned hieroglyphic text
- 44-character vocabulary (letters A-Z, digits 0-9, and a few punctuation/special chars)
- 90/10 train/val split stored as numpy memmap files (train.bin, val.bin)
- Character-to-integer mapping in meta.pkl

The baseline model is standard nanoGPT (GPT-2 style):
- 6 layers, 6 heads, 384 embedding dim (~10M params)
- 256 token context window
- LayerNorm, GELU activation, learned position embeddings
- AdamW optimizer with cosine LR schedule
- Dropout 0.2

## Experimentation

Each experiment runs on a single GPU. Training runs for a **fixed time budget of 5 minutes** (wall clock training time). Launch with:

```
python train.py config/train_egypt_char.py
```

**What you CAN modify:**

- `train.py` — training loop, optimizer, data loading, evaluation
- `model.py` — model architecture (attention, MLP, normalization, embeddings)
- `config/train_egypt_char.py` — all hyperparameters

**What you CANNOT modify:**

- `data/egypt_char/prepare.py` — data preparation is fixed
- Do not install new packages or add dependencies

**The goal is simple: get the lowest val_bpb.** Since this is a character-level model with single-byte ASCII characters, val_bpb = val_loss / ln(2). The time budget is fixed at 5 minutes, so you don't need to worry about training time — it's always 5 minutes. Everything is fair game: architecture, optimizer, hyperparameters, batch size, model size, context length.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline — run the training script as-is.

## Ideas to explore

These are techniques from state-of-the-art small LLM research (from Karpathy's nanochat/autoresearch and related work). The agent is encouraged to try them, but should exercise judgment about what's likely to help given this specific small, character-level task.

### Architecture changes

- **RMSNorm instead of LayerNorm**: Simpler normalization — no mean subtraction. Replace `F.layer_norm` with `F.rms_norm(x, (x.size(-1),))` or implement as `x * rsqrt(mean(x²) + eps)`. Often faster with no quality loss.
- **ReLU² activation instead of GELU**: In the MLP, replace `GELU(x)` with `ReLU(x).square()`. Can be more effective for small models. The autoresearch baseline uses this.
- **Rotary positional embeddings (RoPE)** instead of learned position embeddings: Removes the `wpe` embedding table entirely. Better length generalization. Precompute cos/sin frequencies, apply rotation to Q and K before attention. This is what modern LLMs use.
- **QK-Norm**: Apply RMSNorm to Q and K after projection, before computing attention scores. Stabilizes training, especially at higher learning rates.
- **Value residuals (ResFormer)**: Compute value embeddings directly from token IDs at each layer (via an embedding table), and mix them into the attention values with a learnable gate. Helps gradient flow through deep networks.
- **Sliding window attention**: Use shorter attention windows for early/middle layers, full attention only for the last layer. Pattern like "SSSL" (Short, Short, Short, Long). Saves compute in early layers which tend to be more local anyway.
- **Softcap logits**: Cap the final logits with `softcap * tanh(logits / softcap)` to prevent extreme values. Try softcap=15 or 30. Gemma 2 uses this.
- **Remove bias**: Set bias=False in all Linear and LayerNorm layers. Slightly faster and often no quality loss. The current config supports this (`bias = False`).
- **Per-layer residual scaling**: Learn per-layer scalars for residual connections: `x = lambda_resid[i] * x + lambda_x0[i] * x0` where x0 is the initial embedding. Helps with training stability and gradient flow.
- **Aspect ratio tuning**: Explore the depth vs width tradeoff. Try `model_dim = depth * aspect_ratio` (e.g. aspect_ratio=64) and vary depth. A deeper, narrower model might outperform a shallow, wide one (or vice versa).
- **Grouped query attention (GQA)**: Use fewer KV heads than Q heads (e.g. 6 Q heads but only 2-3 KV heads). Saves memory and compute, sometimes improves quality.
- **Weight tying**: Already present (wte ↔ lm_head). Confirm this stays in place, especially if changing embedding dimensions.

### Optimizer changes

- **Muon optimizer** for matrix (2D) parameters: Uses Newton-Schulz orthogonalization to find steepest-descent directions on the manifold of matrices. Can converge significantly faster than Adam for weight matrices. Keep AdamW for embeddings and 1D parameters (biases, layernorm scales). This is what autoresearch uses.
- **Higher learning rates for embeddings**: Token embeddings can often tolerate 10-100x higher LR than matrix weights. autoresearch uses embedding_lr=0.6 vs matrix_lr=0.04.
- **Separate LR per parameter type**: Different learning rates for: embeddings (high), unembeddings/lm_head (low), matrix weights (medium), scalar parameters (high). This gives each parameter type its optimal learning speed.
- **Cosine warmdown schedule**: Instead of pure cosine decay, use: (1) linear warmup, (2) flat LR period, (3) linear decay in the final 50% of training down to 0. The autoresearch baseline uses warmup_ratio=0.0, warmdown_ratio=0.5.
- **Cautious weight decay**: Only apply weight decay to a parameter when the gradient and the parameter point in the same direction. Mask: `(grad * param) >= 0`.

### Training changes

- **Increase batch size**: Larger effective batch sizes (via gradient accumulation) often help. Try total_batch_size of 2^17 to 2^19 tokens per step.
- **Increase context length**: The current 256 might be limiting for hieroglyphic sequences. Try 512 or 1024 if memory allows. Hieroglyphic inscriptions can be long.
- **Remove or reduce dropout**: For short training runs (5 min), dropout can hurt more than help since we're not at risk of overfitting within the time budget. Try dropout=0.0.
- **Gradient clipping**: Experiment with the clip value, or remove clipping if using QK-norm and other stabilization techniques.
- **Warmup duration**: Try shorter warmup (or none — autoresearch uses warmup_ratio=0.0) since training runs are short.
- **Disable torch.compile if it's slow to start**: On some GPUs, compilation overhead eats into the 5-minute budget. Profile whether compile helps net of startup cost.

## Output format

The training script prints a summary at the end:

```
---
val_bpb: 1.234567
val_loss: 0.855432
training_seconds: 300.1
peak_vram_mb: 2048.3
num_steps: 5000
num_params_M: 10.6
```

Extract the key metric:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 2.1 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.892300	2.0	keep	baseline
b2c3d4e	1.876100	2.1	keep	increase LR to 2e-3
c3d4e5f	1.901000	2.0	discard	switch to ReLU squared
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

Do not commit results.tsv — it must stay untracked to survive git resets on discarded experiments. If running on Colab, also copy it to Google Drive after each experiment for persistence.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `train.py`, `model.py`, and/or `config/train_egypt_char.py` with an experimental idea.
3. git commit
4. Run the experiment: `python train.py config/train_egypt_char.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in results.tsv (do NOT commit results.tsv — it must survive git resets)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
