"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
import unicodedata
import math
from collections import Counter
from pathlib import Path
import json
from datasets import load_dataset



"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None)->dict: #given list of ints, return dictionary with counts of consec pairs.

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx): #lst of tokens, pair we want to merge, new token id.
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids




# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        self.merges = {} # (int, int) -> int
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False): 
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text) #regex--> list of chunks

        ids = [list(ch.encode("utf-8")) for ch in text_chunks] #list of ID lists

        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges): #we get exactly as many merges as we want
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats) #stats assigned with count now
            pair = max(stats, key=stats.get) #max returns item from thing we loop over. key= tells max how to rank.
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx #assign that merge token ID
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] #put that token ID into bytes
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges 
        self.vocab = vocab 
    def _build_vocab(self): #we save our merges dictionary, NOT the byte assignments.
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1] #builds from the merges
        for special, idx in self.special_tokens.items(): #special tokens
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        file_prefix = str(file_prefix)
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab = self._build_vocab()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids


# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class
def text_prep(target_byte_count_train=100000,target_byte_count_test=100000): #streaming=True means dont download the whole dataset first, gives iterable instead
    ds = load_dataset("HuggingFaceFW/fineweb",name="sample-10BT",split="train",streaming=True)
    docs_train = []
    docs_test = []
    train_byte_count = 0
    test_byte_count = 0
    for doc in ds:
        text = doc["text"] #key in each iterable
        text_bytes = len(text.encode("utf-8"))

        if train_byte_count < target_byte_count_train:
            docs_train.append(text)
            train_byte_count += text_bytes
        else:
            docs_test.append(text)
            test_byte_count += text_bytes
        if train_byte_count >= target_byte_count_train and test_byte_count >= target_byte_count_test:
            break
    train_text = "\n\n".join(docs_train)
    test_text = "\n\n".join(docs_test)

    return train_text,test_text
#loading:
#tok = RegexTokenizer()
#tok.load("tokenizers/regex1000model")
#tok1 = {"regex1000":tok} 
def analytics(testing, tokenizers, top_n=20, check_roundtrip=False, from_files=False):
    """
    Compare tokenizers on one or more large text samples.

    testing can be a string, a Path, an iterable, or a dict of name -> text/path.
    tokenizers can be a single tokenizer, a list/tuple of tokenizers, or a
    dict of name -> tokenizer.

    Set from_files=True when testing contains file paths instead of raw text.
    Roundtrip checks are off by default because they decode the full corpus.
    """
    def load_sample(sample):
        if from_files: #if its not a file path, assume its alr text.
            return Path(sample).read_text(encoding="utf-8")
        return sample

    if isinstance(testing, dict): #deconstruct the dict into tuples
        texts = [(name, load_sample(sample)) for name, sample in testing.items()] 
    elif isinstance(testing, (str, Path)): #str or path
        texts = [("sample_0", load_sample(testing))]
    else: #string like thing
        texts = [
            (f"sample_{idx}", load_sample(sample))
            for idx, sample in enumerate(testing)
        ]

    if isinstance(tokenizers, dict): #multiple tokenizers
        tokenizer_items = list(tokenizers.items())
    else: #single tokenizer
        tokenizer_items = [
            (getattr(tokenizers, "__class__", type(tokenizers)).__name__, tokenizers)
        ]

    results = []
    for text_name, text in texts:
        text_bytes = len(text.encode("utf-8")) #natural baseline
        word_count = len(re.findall(r"\S+", text))

        for tokenizer_name, tokenizer in tokenizer_items:
            ids = tokenizer.encode(text)
            token_count = len(ids)
            counts = Counter(ids)
            unique_tokens = len(counts)
            vocab_size = getattr(tokenizer, "n_vocab", None)
            vocab_size = vocab_size if vocab_size is not None else len(getattr(tokenizer, "vocab", {})) 
            #rn we dont have vocab_size attr exposed
            top_counts = counts.most_common(top_n)

            if token_count:
                probabilities = [count / token_count for count in counts.values()]
                entropy = -sum(p * math.log2(p) for p in probabilities)
                top_token_share = top_counts[0][1] / token_count if top_counts else 0
                top_n_share = sum(count for _, count in top_counts) / token_count #avg exp surprisal per token
            else:
                entropy = 0
                top_token_share = 0
                top_n_share = 0
            compression_ratio = text_bytes / token_count
            token_reduction_pct = 1 - (token_count / text_bytes)
            byte_token_count = sum(count for token_id, count in counts.items() if token_id < 256)
            merge_token_count = token_count - byte_token_count
            decoded = tokenizer.decode(ids) if check_roundtrip and hasattr(tokenizer, "decode") else None

            results.append({
                "sample": text_name,
                "tokenizer": tokenizer_name,
                "bytes": text_bytes,
                "words": word_count,
                "tokens": token_count,
                "compression_ratio": compression_ratio,
                "compression_pct":token_reduction_pct,
                "bytes_per_token": text_bytes / token_count if token_count else 0,
                "tokens_per_word": token_count / word_count if word_count else 0,
                "tokens_per_kb": token_count / (text_bytes / 1024) if text_bytes else 0,
                "unique_tokens": unique_tokens,
                "vocab_size": vocab_size,
                "vocab_coverage": unique_tokens / vocab_size if vocab_size else None,
                "entropy_bits_per_token": entropy,
                "effective_vocab_size": 2 ** entropy if entropy else 0,  #reversing entropy
                "token_reuse_rate": 1 - (unique_tokens / token_count) if token_count else 0,
                "most_common_token_share": top_token_share,
                f"top_{top_n}_token_share": top_n_share,
                "byte_token_share": byte_token_count / token_count if token_count else 0,
                "merge_token_share": merge_token_count / token_count if token_count else 0,
                "top_tokens": top_counts,
                "roundtrip": decoded == text if decoded is not None else None,
            })

    return results

#lowkey include base case for graphing purposes.
def train_tokenizers(text,vocab_sizes,pattern=None,save_dir="tokenizers"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizers = {}
    for vocab_size in vocab_sizes:
        name = f"regex_vocab_{vocab_size}"
        tokenizer = RegexTokenizer(pattern=pattern)
        tokenizer.train(text,vocab_size)
        tokenizer.save(Path(save_dir) / name)
        tokenizers[name] = tokenizer

        print(f"finished training tokenizer: {vocab_size}")
    return tokenizers

def save_analytics(save_dir,text):

    save_dir = Path(save_dir)
    tokenizers = {}
    for model_file in save_dir.glob("*.model"):
        name = model_file.stem
        tok = RegexTokenizer()
        tok.load(str(model_file))
        tokenizers[name] = tok
    results = analytics(testing=text,tokenizers=tokenizers,from_files=False)
    with open(save_dir / "analytics_summaries.json","w",encoding="utf-8") as f:
        json.dump(results,f,indent=2)
    return results



def run(vocab_sizes:list):
    train,test = text_prep() #add size params
    train_tokenizers(train,vocab_sizes)  #creates the objects, saves them
    save_analytics("tokenizers",test)  #folder where train_tokenizers saves them
    
    
