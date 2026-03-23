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
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256 # context of up to 256 previous characters

# smaller model for more steps
n_layer = 3
n_head = 4
n_embd = 256
dropout = 0.0

learning_rate = 5e-4 # lower LR
max_iters = 50000
lr_decay_iters = 50000 # time budget is binding constraint
min_lr = 5e-5 # learning_rate / 10
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 20 # short warmup for stable LR

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
