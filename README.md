# SampleGPT

Minimal GPT implementation for learning how a decoder-only Transformer works.

This project is intentionally small:
- no Hugging Face `transformers`
- no optimization framework
- no distributed training
- no production serving layer

Goal: understand the core mechanics of GPT training and inference from first principles. For a more useful chat, consider this [repo](https://github.com/phtuananh/jokesGpt)

## Project structure

```text
miniGPT/
├── requirements.txt
├── train_bpe.py
├── chat_bpe.py
├── data/data_shakesspear.txt
└── models/shakespeare
    ├── model.pt
    ├── tokenizer.json
    ├── config.json
    └── data.txt
```

## Requirements

```txt
torch
tokenizers
```

## Install

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install torch tokenizers
```

## Prepare data

The training file is a plain text file. Use data/data_sharespeare.txt for sample. The data/data_jokes.txt will not work well, for an useful jokes chatbot, consider this [repo](https://github.com/phtuananh/jokesGpt)

Example:

```text
data/data_sharespeare.txt
```

You can use any text corpus: stories, jokes, books, notes, code, etc.

For a quick test:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data.txt
```

For better output, use a larger text file.

Recommended minimum:

```text
100k+ tokens
```

Very small files will fail or produce bad output.

## Train

Command:

```bash
python train_bpe.py <data_file_path> <output_folder_name>
```

Example:

```bash
python train_bpe.py data.txt output_folder_name
```

This creates:

```text
models/<output_folder_name>
├── model.pt
├── tokenizer.json
├── config.json
└── data.txt
```

### Training time
For a Jokes dataset of 230k jokes on an old PC without GPU, it take around 25'

## Run chat/reference inference
Command:

```bash
python chat_bpe.py <output_model_path>
```

Example:

```bash
python chat_bpe.py model_file_path
```

Then type a prompt:

```text
Question > Once upon a time
```

Exit:

```text
exit
```

## What is saved

### `model.pt`

PyTorch model weights.

Contains learned tensors:

```text
token_embedding.weight
position_embedding.weight
attention weights
feed-forward weights
lm_head.weight
```

### `tokenizer.json`

BPE tokenizer.

Required to convert text to token IDs and token IDs back to text.

### `config.json`

Model architecture config.

The chat script needs it to rebuild the same GPT architecture before loading weights.

### `data.txt`

Copy of the training text.

Not required for inference, but useful for reproducibility and debugging.

## Important rule

The chat script must use the same:

```text
model.pt
tokenizer.json
config.json
```

from the same training run.

Do not mix files from different model folders.

## Expected quality

This is a learning GPT, not a real assistant.

Expected behavior:

| Dataset size | Result |
|---|---|
| tiny file | error or bad output |
| 1 MB | learns style weakly |
| 10 MB | recognizable text style |
| 100 MB+ | better mini-GPT behavior |

## Core idea

Training:

```text
text
→ BPE tokenizer
→ token IDs
→ GPT
→ predict next token
→ loss
→ backpropagation
→ model.pt
```

Inference:

```text
prompt
→ tokenizer.json
→ token IDs
→ model.pt
→ next-token generation
→ decoded text
```
## Training parameters

Currently configured inside `train_bpe.py`:

```python
batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

vocab_size_target = 2000
```

Meaning:

| Parameter | Meaning |
|---|---|
| `batch_size` | number of training sequences per step |
| `block_size` | context length in tokens |
| `max_iters` | number of training steps |
| `eval_interval` | print loss every N steps |
| `learning_rate` | optimizer step size |
| `n_embd` | hidden dimension |
| `n_head` | number of attention heads |
| `n_layer` | number of Transformer blocks |
| `dropout` | regularization |
| `vocab_size_target` | target BPE vocabulary size |

If your dataset is small, reduce:

```python
block_size = 32
```

or:

```python
block_size = 16
```

## Minimal philosophy

This project keeps only the essentials:

```text
PyTorch
BPE tokenizer
Transformer blocks
Training loop
Save/load
Interactive generation
```

No optimization. No abstraction. No magic.

## Acknowledgements

This project is heavily inspired by the educational work of Andrew Karpathy.

The Transformer architecture, training loop structure, attention implementation, GPT block organization, and overall learning approach are based on concepts presented in:

* https://github.com/karpathy/minGPT
* https://github.com/karpathy/nanoGPT
* https://github.com/karpathy/ng-video-lecture

Karpathy's repositories are among the best resources for understanding GPT models from first principles.

This project intentionally keeps the same educational philosophy:

* minimal code
* minimal dependencies
* transparent implementation
* learning-focused design

Differences from minGPT/nanoGPT:

* uses a tokenizer Hugging Face 
* stores tokenizer and configuration separately
* includes a simple interactive chat program
* simplified folder structure for experimentation
* designed as a learning project rather than a production training framework

Credit goes to Andrew Karpathy for the original educational GPT implementations and teaching material and to ChatGPT 5.5.
