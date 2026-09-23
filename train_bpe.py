import json
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datetime import datetime
import shutil
import os
from pathlib import Path

# Usage:
# python train_bpe.py data.txt out_model
if len(sys.argv) != 3:
    print("Usage: python train_bpe.py <data.txt> <out_dir>")
    sys.exit(1)
def log(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")

data_file = sys.argv[1]
out_dir = sys.argv[2]
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

out_dir = models_dir / out_dir
out_dir.mkdir(exist_ok=True)
# ------------------
# Config
# ------------------

batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 50
learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

vocab_size_target = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Train BPE tokenizer
# ------------------

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size_target,
    special_tokens=["[UNK]"]
)

tokenizer.train([data_file], trainer)
tokenizer.save(f"{out_dir}/tokenizer.json")

# ------------------
# Encode data
# ------------------

text = open(data_file, "r", encoding="utf-8").read()
ids = tokenizer.encode(text).ids

vocab_size = tokenizer.get_vocab_size()
data = torch.tensor(ids, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

log(f"Number of tokens: {len(ids)}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ------------------
# Model
# ------------------

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ------------------
# Training
# ------------------

for step in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        log(f"Step {step}/{max_iters}: loss {loss.item():.4f}")

# ------------------
# Save
# ------------------

torch.save(model.state_dict(), f"{out_dir}/model.pt")

config = {
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "vocab_size": vocab_size,
}

with open(f"{out_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)

shutil.copy2(
    data_file,
    os.path.join(
        out_dir,
        os.path.basename(data_file)
    )
)

log("Saved:")
log(f"{out_dir}/model.pt")
log(f"{out_dir}/tokenizer.json")
log(f"{out_dir}/config.json")
log(f"{out_dir}/{os.path.basename(data_file)}")

