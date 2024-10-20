"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

from hiero_transformer import utils
import matviz
from tqdm import tqdm

# download the tiny shakespeare dataset
cur_path = os.path.dirname(__file__)
input_file_path = os.path.join(cur_path, 'cleaned_graphics.txt')
print(f"input path is: {input_file_path}")


if True: # not os.path.exists(input_file_path):
    json_path = os.path.join(cur_path, 'validation_data.json')
    print(json_path)

    if not os.path.exists(json_path):
        raise Exception("Please run setup.sh to get the training data from the egypt repo")
    else:
        print("CLEANING THE DATA")
        graphics_data = utils.load_data_from_folder(cur_path)
        # clean the hieroglyphics data
        all_sources = []
        for item in tqdm(graphics_data):
            if item['source']:
                cur_item = utils.clean_graphics(item['source'])
                all_sources.append(cur_item)

        # remove duplicates   
        all_sources = list(set(all_sources))

        # print the file
        cleaned_graphics = "\n".join(all_sources)
        matviz.etl.write_string(input_file_path, cleaned_graphics)

        print("\nCLEANING COMPLETE.")



with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)