# tinyGPT

A mini transformer language model platform for rapid NLP experimentation on your laptop. Supports both character-level and BPE-tokenized GPT models—optimized for running and training even on Apple Silicon (M1/M2/M3).

---

## Features

- **Train & generate:** both character-level and BPE-based GPT models
- **Unified CLI** for training, text generation, and interactive chat with your models
- **Interactive chat mode:** Pick your model type and chat like a simple AI assistant
- **Sample datasets:** Use Tiny Stories or Tiny Shakespeare out of the box
- **Apple Silicon/MPS support** (and works on CPU)
- **Easily configurable:** Customize model size, batch/config settings, sampling, and more

## Quick-start

### 1. **Install requirements**

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 2. **Train a Model**

#### BPE Model

```bash
python cli.py train-bpe
```

#### Char-level Model

```bash
python cli.py train-char
```

### 3. **Generate Text**

#### From BPE Model

```cmd
python cli.py generate-bpe --prompt "Once upon a time" --num_tokens 120
```

#### From Char-level Model

```cmd
python cli.py generate-char --prompt "ROMEO:" --num_tokens 100
```


### 4. **Chat Mode**

```cmd
python cli.py chat
```

- Pick between Char/BPE model.
- Chat interactively with your trained model.
- Type `exit` to quit.

---

## Dataset Format

- Provide a plain text file with one story or passage per line.
- Examples included: `tiny_stories.txt`, `tiny_shakespeare.txt`

---

## Model Configuration Example

_For best results on an 8GB M1/M2 MacBook Air (“medium” BPE model):_

DEFAULT -- Config as below

- vocab_size: 1024–2048 (with pre-trained BPE)
- block_size: 128
- n_layers: 4
- n_heads: 4
- n_embd: 128
- batch_size: 128–256 (if RAM allows)
- epochs: 2–4
- learning_rate: 3e-4

---

## Develop & Extend

- Change `SURPRISE_PROMPTS` in `cli.py` to tune your favorite story starters.
- **Training:** Adjust batch/model epochs for more/less speed vs. coherence.
- **Sampling:** Experiment with `--temperature` for more creative or conservative outputs.

---

## Apple Silicon Support

Apple Silicon users: All training/generation will use the Metal/MPS GPU automatically if available (otherwise falls back to CPU).

Check MPS support:
