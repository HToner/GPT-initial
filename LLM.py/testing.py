import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
import pandas as pd
import copy
import subprocess
#from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------
@dataclass
class TrainResults:
    config: GPTConfig
    train_logs: list[dict]
    val_logs: list[dict]
    best_val_loss: float
    best_val_step: int
    best_model_path: str
    stopped_due_to_time: bool
    elapsed_seconds: float

@dataclass #decorator that creates init, so we don't have to write it 
class GPTConfig:
    batch_size: int = 4
    total_batch_size: int = 100
    max_steps: int = 10000
    max_lr: float = 6e-6
    min_lr_ratio: float = 0.5
    warmup_steps: int = 50
    block_size: int = 256 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    #messy number--need to increase to pwr 2
    n_layer: int = 3 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 256 # embedding dimension
    seed: int = 1337
    max_time_seconds: float | None = None

def train_gpt(config:GPTConfig,run_name:str):
    run_dir = os.path.join("runs",run_name)
    os.makedirs(run_dir,exist_ok=True)
    best_model_path = os.path.join(run_dir,"best_model.pt")

    train_logs = []
    val_logs = []



    total_batch_size = config.total_batch_size       #524288 # 2**19, ~0.5M, in number of tokens 
    B = config.batch_size     #4 # micro batch size  MAKE SURE THAT TOTAL BATCH IS DIVISIBLE BY B*T
    T = config.block_size #1024 # sequence length
    max_lr = config.max_lr  #6e-4
    min_lr = max_lr * config.min_lr_ratio
    warmup_steps = config.warmup_steps #715
    max_steps = config.max_steps # 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    testing_prompt = "the best thing about living alone"
    class CausalSelfAttention(nn.Module):

        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            # key, query, value projections for all heads, but in a batch
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            # output projection
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1
            #where calling this from?
            # regularization
            self.n_head = config.n_head
            self.n_embd = config.n_embd

        def forward(self, x):
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
            # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
            #this line replaces the softmax to do it all in GPU memory
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            # output projection
            y = self.c_proj(y) #mix heads back together (refers to linear)
            return y

    class MLP(nn.Module): #multi layer perceptron

        def __init__(self, config):
            super().__init__()
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) #widen (standard)
            self.gelu    = nn.GELU(approximate='tanh') #smoother RELU
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1  #called before resid updates-->make sure weights havent
            #blown up (deep net)

        def forward(self, x):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            return x

    class Block(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
    #--------------------------------------------------------------------
    #indiv transformer

    class GPT(nn.Module):

        def __init__(self, config):
            super().__init__()
            self.config = config

            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd), #token embedding
                wpe = nn.Embedding(config.block_size, config.n_embd), #position embedding
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #n_layer transformer layers
                #Block(config) makes a block object using config (recall config is dict w params)
                ln_f = nn.LayerNorm(config.n_embd),  
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            # weight sharing scheme
            self.transformer.wte.weight = self.lm_head.weight

            # init params
            self.apply(self._init_weights)

        def _init_weights(self, module): #everys single module gets checked
            if isinstance(module, nn.Linear): #checks if the layer in the module is linear
                std = 0.02
                if hasattr(module, 'NANOGPT_SCALE_INIT'): #attack NANO to give it the attr
                    std *= (2 * self.config.n_layer) ** -0.5 #shrinks std even more (why?)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std) #normal mean 0 and std calc applied
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding): #checks if layer is embedding
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            # idx is of shape (B, T)
            B, T = idx.size() #how many sequences we have and the length of them
            assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
            # forward the token and posisition embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
            x = tok_emb + pos_emb
            # forward the blocks of the transformer
            for block in self.transformer.h: 
                #transformer init as the whole transformer process
                #Recall :h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
                #so self.transformer.h calls all n_layer transformers, and block grabs each one
                x = block(x)
                #applies each of the layers in block
            # forward the final layernorm and the classifier
            x = self.transformer.ln_f(x) #final layer norm
            logits = self.lm_head(x) # (B, T, vocab_size) back to individual scores for each token
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        


        def configure_optimizers(self, weight_decay, learning_rate, device_type):
            # start with all of the candidate parameters (that require grad)
            param_dict = {pn: p for pn, p in self.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            if master_process:
                print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"
            if master_process:
                print(f"using fused AdamW: {use_fused}")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
            return optimizer

    # -----------------------------------------------------------------------------

    def load_tokens(filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    class DataLoaderLite:
        def __init__(self, B, T, process_rank, num_processes, split): #single GPU so process_rank,num_processes=0
            self.B = B
            self.T = T
            self.process_rank = process_rank
            self.num_processes = num_processes
            assert split in {'train', 'val'}

            # get the shard filenames
            data_root = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")
            shards = os.listdir(data_root) #shards is now the name of each file in the folder
            shards = [s for s in shards if split in s] #filters train or val
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            if master_process:
                print(f"found {len(shards)} shards for split {split}")
            self.reset()

        def reset(self):
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        def next_batch(self): #goal: return the next x,y from the current shard
            B, T = self.B, self.T #readability
            buf = self.tokens[self.current_position : self.current_position+B*T+1] #from where we are to what we need
            x = (buf[:-1]).view(B, T) # inputs
            y = (buf[1:]).view(B, T) # targets
            # advance the position in the tensor
            self.current_position += B * T * self.num_processes
            # if loading the next batch would be out of bounds, advance to next shard
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards) #makes the loading cycle continuous
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
            return x, y
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    ddp = int(os.environ.get('RANK', -1)) != -1 #obv false bc we have no gpu
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
        # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337) #random weight initialization, dropout masks
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")


    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    torch.set_float32_matmul_precision('high')

    # create model
    model = GPT(config) #GPTconfig with default params, pass that into GPT!
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)


    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimize!
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

    def get_torch_gpu_stats(device="cuda"):

        return {
            "mem_allocation":torch.cuda.memory_allocated(device) / 1024**2,
            "gpu_mem_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            "gpu_peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
    def get_nvidia_smi_stats():
        try:
            result = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ], encoding="utf-8")

            gpu_util, power_draw, mem_used, mem_total = result.strip().split(",")

            return {
                "gpu_utilization_pct": float(gpu_util),
                "gpu_power_watts": float(power_draw),
                "gpu_memory_used_mb": float(mem_used),
                "gpu_memory_total_mb": float(mem_total),
            }

        except Exception:
            return {
                "gpu_utilization_pct": None,
                "gpu_power_watts": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
            }
    def get_gpu_stats(device="cuda"):
        stats = {}
        stats.update(get_torch_gpu_stats(device))
        stats.update(get_nvidia_smi_stats())
        return stats
    best_val_loss = float("inf")
    best_val_step = None
    best_model_state = None
    train_start_time = time.monotonic()
    stopped_due_to_time = False
    for step in range(max_steps):
        elapsed_seconds = time.monotonic() - train_start_time
        if config.max_time_seconds is not None and elapsed_seconds >= config.max_time_seconds:
            stopped_due_to_time = True
            if master_process:
                print(f"time budget reached before step {step}; stopping training")
            break

        t0 = time.monotonic()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                val_loss = val_loss_accum.item()
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                    best_model_state = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        "step":step,"model_state_dict":model.state_dict(),"val_loss":best_val_loss,"config":config
                    }
                    torch.save(checkpoint,best_model_path)
            val_logs.append({
                "step":step,"val_loss":val_loss_accum.item(),"val_perplexity":math.exp(val_loss_accum),**get_gpu_stats(device)
            })

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        train_logs.append({
            "step":step,
            "train_loss":loss_accum.item(),
            "lr":lr,
            "grad_norm":norm.item(),
            "tps":tokens_per_sec,
            "elapsed_seconds":time.monotonic() - train_start_time,
            **get_gpu_stats(device)
        })
        train_df = pd.DataFrame(train_logs)
        val_df = pd.DataFrame(val_logs)

        train_df.to_csv(os.path.join(run_dir, "train_logs.csv"), index=False)
        val_df.to_csv(os.path.join(run_dir, "val_logs.csv"), index=False)
    return TrainResults(
        config=GPTConfig,
        train_logs=train_logs,
        val_logs=val_logs,
        best_val_loss=best_val_loss,
        best_val_step=best_val_step,
        best_model_path=best_model_path,
        stopped_due_to_time=stopped_due_to_time,
        elapsed_seconds=time.monotonic() - train_start_time)






"""
class GPTConfig:
    batch_size: int = 4
    total_batch_size: int = 100
    max_steps: int = 10000
    max_lr: float = 6e-6
    min_lr_ratio: float = 0.5
    warmup_steps: int = 50
    block_size: int = 256 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    #messy number--need to increase to pwr 2
    n_layer: int = 3 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 256 # embedding dimension
    seed: int = 1337
    max_time_seconds: float | None = None

"""
tests = {"layers":GPTConfig(n_layer=3),"heads,layers":GPTConfig(n_layer=3,n_head=2)}
for key in tests:
    train_gpt(tests[key],key)
