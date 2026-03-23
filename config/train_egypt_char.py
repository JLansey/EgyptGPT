# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-glyphs-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # disabled for autoresearch
wandb_project = 'glyph-char'
wandb_run_name = 'mini-gpt'

dataset = 'egypt_char'
sign_level = True  # sign-level tokenization
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256 # context of up to 256 sign tokens (~870 chars)

# model sized for sign-level vocab (~781 tokens)
n_layer = 3
n_head = 4
n_embd = 256
dropout = 0.3  # regularize against overfitting on smaller sign-level dataset

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 4000  # shorter decay
min_lr = 0
beta2 = 0.99

warmup_iters = 20
grad_clip = 0.5

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
