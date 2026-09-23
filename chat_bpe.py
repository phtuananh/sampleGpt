import json
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# Usage:
# python chat_bpe.py out_model

if len(sys.argv) != 2:
    print("Usage: python chat_bpe.py <model_dir>")
    sys.exit(1)

model_dir = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Load config/tokenizer
# ------------------

with open(f"{model_dir}/config.json", "r") as f:
    cfg = json.load(f)

block_size = cfg["block_size"]
n_embd = cfg["n_embd"]
n_head = cfg["n_head"]
n_layer = cfg["n_layer"]
dropout = cfg["dropout"]
vocab_size = cfg["vocab_size"]

tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
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

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ------------------
# Load model
# ------------------

model = GPT().to(device)
model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
model.eval()

print(f"Loaded model from: {model_dir}")
print(f"Device: {device}")
print("Type 'exit' to quit.")

# ------------------
# Chat loop
# ------------------

while True:
    prompt = input("\nQuestion > ")

    if prompt.lower() == "exit":
        break

    input_ids = tokenizer.encode(prompt).ids

    if len(input_ids) == 0:
        print("Empty prompt.")
        continue

    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        x,
        max_new_tokens=200,
        temperature=0.8
    )[0].tolist()

    text = tokenizer.decode(output_ids)

    print("\Question >")
    print(text)

