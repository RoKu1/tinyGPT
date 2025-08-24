# from pathlib import Path
import numpy as np

def get_train_val_splits(encoded, train_frac=0.9):
    idx = int(len(encoded) * train_frac)
    return encoded[:idx], encoded[idx:]

def get_batch(data, batch_size, block_size):
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([np.array(data[i:i+block_size]) for i in ix])
    y = np.stack([np.array(data[i+1:i+1+block_size]) for i in ix])
    return x, y

def load_shakespeare(path="tiny_shakespeare.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def build_vocab(text):
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return vocab, stoi, itos

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(ids, itos):
    return ''.join([itos[i] for i in ids])


if __name__ == "__main__":
    # test: run `python tinygpt/data.py` from project root.
    text = load_shakespeare()
    vocab, stoi, itos = build_vocab(text)
    print(f"Vocab size: {len(vocab)}. Example vocab: {vocab[:10]}")
    sample_text = "To be, or not to be"
    ids = encode(sample_text, stoi)
    print("Encoded:", ids)
    decoded = decode(ids, itos)
    print("Decoded:", decoded)

    encoded = encode(text, stoi)
    train_ids, val_ids = get_train_val_splits(encoded)
    print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")

    # Try out a batch
    x, y = get_batch(train_ids, batch_size=4, block_size=8)
    print("x shape:", x.shape)
    print("Sample x (as text):")
    for row in x:
        print(decode(row, itos))
    print("Sample y (next tokens for x):")
    for row in y:
        print(decode(row, itos))
