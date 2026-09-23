import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================
# CONFIG (must match training)
# ==========================

block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# LOAD TOKENIZER
# ==========================

text = open("data.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join([itos[i] for i in l])

# ==========================
# MODEL
# ==========================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)

        v = self.value(x)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        return self.proj(out)


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

        self.sa = MultiHeadAttention(
            n_head,
            n_embd // n_head
        )

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

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding(idx)

        pos = self.position_embedding(
            torch.arange(T, device=device)
        )

        x = tok + pos

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=1.0
    ):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

        return idx


# ==========================
# LOAD MODEL
# ==========================

model = GPT().to(device)

model.load_state_dict(
    torch.load(
        "nanogpt.pt",
        map_location=device
    )
)

model.eval()

print("Model loaded.")
print("Type 'exit' to quit.")
print()

# ==========================
# CHAT LOOP
# ==========================

while True:

    prompt = input("You > ")

    if prompt.lower() == "exit":
        break

    tokens = encode(prompt)

    if len(tokens) == 0:
        print("Unknown characters.")
        continue

    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        x,
        max_new_tokens=200,
        temperature=0.8
    )

    answer = decode(
        output[0].tolist()
    )

    print()
    print("Model >")
    print(answer)
    print()

