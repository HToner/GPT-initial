"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # progress bar

# ------------------------------------------
local_dir = "edu_fineweb10B" #where to store the data
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True) #create the file if it doesnt exist alr

# download the dataset
fw_small = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train[:2]") #which dataset we use

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"])) #encodes it according to enc and adds to tokens
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16) 
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool: #distribute the downloading amongst all the CPUs
    #creates instance of pythons multiprocessing Pool
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16) #space in each shard
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw_small, chunksize=16): 
#grab each item and parallel tokenize it then return it in the same order
        # is there enough space in the current shard for the new tokens?
        #so it takes the tokenized result of each doc out of 16
        if token_count + len(tokens) < shard_size: #if current count + length of this batch  is less
            #than what space is left inshard, add them
            all_tokens_np[token_count:token_count+len(tokens)] = tokens #this is where they get appended
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else: #this is where the tokens actually all get added to the shard
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            #train or val shard
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count #remainder obv
            progress_bar.update(remainder)
            #remember all chars have been tokenized. so we just need to fill the remaining space.
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            #now its full, so we can write it.
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            #makes remaining tokens at beginning of all_tokens_np
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])