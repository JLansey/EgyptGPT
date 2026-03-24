# EgyptGPT Deployment Guide

## Model artifacts

### On Google Drive (`/content/drive/MyDrive/EgyptGPT_autoresearch/`)

| File | Size | Description |
|------|------|-------------|
| `out-glyphs-char/ckpt.pt` | 26MB | Model checkpoint (weights + config) |
| `meta_signs.pkl` | 16KB | Sign-level vocabulary (781 Gardiner sign tokens) |
| `meta.pkl` | 2KB | Original char-level vocabulary (44 chars) |
| `results.tsv` | 5KB | Full experiment log (89 experiments) |
| `RESEARCH_REPORT.md` | 8KB | Detailed research findings |

### In the repo (`/content/EgyptGPT/`)

| File | Description |
|------|-------------|
| `model.py` | GPT model definition (required to load checkpoint) |
| `sample_signs.py` | Generation script for sign-level model |
| `sample.py` | Original char-level generation script |
| `config/train_egypt_char.py` | Training config (for retraining) |
| `train.py` | Training script (for retraining) |
| `data/egypt_char/train_signs.bin` | Sign-level training data |
| `data/egypt_char/val_signs.bin` | Sign-level validation data |
| `data/egypt_char/meta_signs.pkl` | Sign vocabulary mapping |

## Generate hieroglyphic text

### Quick start

```bash
cd /content/EgyptGPT

# Generate 10 inscriptions (default settings)
python sample_signs.py

# Generate 50 inscriptions, save to file
python sample_signs.py --num_samples 50 --out_file generated.txt

# Lower temperature = more conservative/common patterns
python sample_signs.py --temperature 0.5 --num_samples 20

# Higher temperature = more creative/diverse
python sample_signs.py --temperature 1.0 --num_samples 20

# Run on CPU (no GPU needed for inference)
python sample_signs.py --device cpu
```

### From Google Drive checkpoint

```bash
# If loading from Drive backup
python sample_signs.py \
  --checkpoint /content/drive/MyDrive/EgyptGPT_autoresearch/out-glyphs-char/ckpt.pt \
  --sign_meta /content/drive/MyDrive/EgyptGPT_autoresearch/meta_signs.pkl
```

### Python API

```python
import torch, pickle
from model import GPTConfig, GPT

# Load model
checkpoint = torch.load('out-glyphs-char/ckpt.pt', map_location='cuda')
model = GPT(GPTConfig(**checkpoint['model_args']))
state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval().to('cuda')

# Load sign vocabulary
with open('data/egypt_char/meta_signs.pkl', 'rb') as f:
    sign_meta = pickle.load(f)
sign_itos = sign_meta['itos']

# Generate
start = torch.tensor([[sign_meta['stoi']['\n']]], dtype=torch.long, device='cuda')
with torch.no_grad():
    tokens = model.generate(start, max_new_tokens=200, temperature=0.8, top_k=200)
    signs = [sign_itos[t] for t in tokens[0].tolist()]
    text = ' '.join(s for s in signs if s != '\n')
    print(text)
```

## Output format

The model generates **Gardiner sign codes** — the standard notation for Egyptian hieroglyphs. Example output:

```
M23 X1 D21 N35 G17 I9 S29 Z7 D36 A1 / N35 X1 D2 Z1
```

Each code (e.g., `M23`, `D21`, `X1`) represents one hieroglyphic sign. Codes are separated by spaces, `/` marks line breaks within an inscription, and newlines separate inscriptions.

### Common signs in output

| Code | Category | Meaning/usage |
|------|----------|---------------|
| X1 | bread loaf | Common phonogram (t) |
| N35 | water | Common phonogram (n) |
| D21 | mouth | Common phonogram (r) |
| M17 | reed | Common phonogram (i) |
| G17 | owl | Common phonogram (m) |
| Z1 | stroke | Determinative/number marker |
| A1 | seated man | Determinative |
| / | — | Line break within inscription |

## Model details

- **Architecture**: 3-layer GPT, 4 heads, 256 embedding dim, ReLU², RoPE, 2.2M params
- **Tokenization**: 781 Gardiner sign tokens (signs appearing ≥5 times in training)
- **Training data**: ~370K sign tokens from ~17K Egyptian inscriptions
- **Performance**: 1.105 bits per character (val set)
