"""
Generate hieroglyphic text from the sign-level EgyptGPT model.

Usage:
    python sample_signs.py                          # defaults
    python sample_signs.py --num_samples 50         # more samples
    python sample_signs.py --temperature 0.7        # less random
    python sample_signs.py --out_file output.txt    # save to file
    python sample_signs.py --device cpu             # run on CPU
"""
import os
import pickle
import argparse
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# --- Config ---
CHECKPOINT = 'out-glyphs-char/ckpt.pt'
SIGN_META = 'data/egypt_char/meta_signs.pkl'

parser = argparse.ArgumentParser(description='Generate hieroglyphic sign sequences')
parser.add_argument('--num_samples', type=int, default=10, help='Number of inscriptions to generate')
parser.add_argument('--max_new_tokens', type=int, default=200, help='Max sign tokens per sample')
parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (lower=more deterministic)')
parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling (0=disabled)')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--out_file', type=str, default=None, help='Output file (default: print to stdout)')
parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='Path to checkpoint')
parser.add_argument('--sign_meta', type=str, default=SIGN_META, help='Path to sign vocabulary')
args = parser.parse_args()

# --- Setup ---
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed(args.seed)
device_type = 'cuda' if 'cuda' in args.device else 'cpu'
ptdtype = torch.bfloat16 if device_type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Load model ---
checkpoint = torch.load(args.checkpoint, map_location=args.device)
model = GPT(GPTConfig(**checkpoint['model_args']))
state_dict = checkpoint['model']
# strip torch.compile prefix if present
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(args.device)
print(f"Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# --- Load sign vocabulary ---
with open(args.sign_meta, 'rb') as f:
    sign_meta = pickle.load(f)
sign_itos = sign_meta['itos']  # int -> sign string
sign_stoi = sign_meta['stoi']  # sign string -> int
print(f"Sign vocabulary: {sign_meta['vocab_size']} tokens")

def decode_signs(token_ids):
    """Convert sign token IDs back to readable Gardiner notation."""
    parts = []
    for tid in token_ids:
        sign = sign_itos.get(tid, '<UNK>')
        if sign == '\n':
            parts.append('\n')
        else:
            parts.append(sign)
    # Join with spaces (signs are space-separated in Gardiner notation)
    result = []
    for i, part in enumerate(parts):
        if part == '\n':
            result.append('\n')
        else:
            if result and result[-1] != '\n':
                result.append(' ')
            result.append(part)
    return ''.join(result)

# --- Generate ---
# Start with newline token (inscription boundary)
newline_id = sign_stoi.get('\n', 0)
start_ids = [newline_id]
x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

output_lines = []
with torch.no_grad():
    with ctx:
        for k in range(args.num_samples):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k if args.top_k > 0 else None)
            tokens = y[0].tolist()
            text = decode_signs(tokens)
            # Split into individual inscriptions and take the first complete one
            inscriptions = [line.strip() for line in text.split('\n') if line.strip()]
            if inscriptions:
                output_lines.append(inscriptions[0])
                print(f"[{k+1}/{args.num_samples}] {inscriptions[0][:100]}{'...' if len(inscriptions[0]) > 100 else ''}")

# --- Output ---
output_text = '\n'.join(output_lines) + '\n'
if args.out_file:
    with open(args.out_file, 'w') as f:
        f.write(output_text)
    print(f"\nSaved {len(output_lines)} inscriptions to {args.out_file}")
else:
    print(f"\n--- Generated {len(output_lines)} inscriptions ---\n")
    print(output_text)
