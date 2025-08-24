from tokenizers import ByteLevelBPETokenizer

# Lazy singleton to load tokenizer only once
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = ByteLevelBPETokenizer(
            "tinygpt_bpe/bpe_tokenizer/vocab.json",
            "tinygpt_bpe/bpe_tokenizer/merges.txt"
        )
    return _tokenizer

def load_stories(path="tiny_stories.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(tokenizer):
    vocab = tokenizer.get_vocab()
    stoi = vocab
    itos = {v: k for k, v in vocab.items()}
    return vocab, stoi, itos

def encode(text, tokenizer):
    return tokenizer.encode(text).ids

def decode(ids, tokenizer):
    return tokenizer.decode(ids)

def get_train_val_splits(encoded, train_frac=0.9):
    idx = int(len(encoded) * train_frac)
    return encoded[:idx], encoded[idx:]

# Example batching function to create training minibatches
import torch
import numpy as np
def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = torch.tensor([data[i:i+block_size] for i in ix], dtype=torch.long)
    y = torch.tensor([data[i+1:i+block_size+1] for i in ix], dtype=torch.long)
    return x, y



if __name__ == "__main__":
    tokenizer = get_tokenizer()
    vocab, stoi, itos = build_vocab(tokenizer)
    print(f"Vocab size: {len(vocab)}")
    text = load_stories("tiny_stories.txt")
    ids = encode(text[:300], tokenizer)
    print("Encoded IDs:", ids[:20])
    print("Decoded:", decode(ids[:20], tokenizer))
