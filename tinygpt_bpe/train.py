import time
import torch
import torch.nn.functional as F
from tinygpt_bpe.config import get_training_configs
from tinygpt_bpe.data import get_tokenizer, build_vocab, load_stories, encode, get_train_val_splits, get_batch
from tinygpt_bpe.model import GPTConfig, TinyGPT



def train():
    train_config = get_training_configs()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare data
    # Load text and tokenizer
    tokenizer = get_tokenizer()
    vocab, stoi, itos = build_vocab(tokenizer)
    vocab_size = len(vocab)
    print(f"Using vocab size: {vocab_size}")

    # Load stories text
    text = load_stories("tiny_stories.txt")
    ids = encode(text, tokenizer)
    train_data, val_data = get_train_val_splits(ids)
    print(
        f"Vocab size: {len(vocab)}. Train size: {len(train_data)}, Val size: {len(val_data)}"
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=train_config.block_size,
        n_embd=train_config.n_embd,
        n_layers=train_config.n_layers,
        n_heads=train_config.n_heads,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

    steps = int(train_config.epochs * (len(train_data) // train_config.batch_size))
    print(f"Training for {steps} steps...")

    for step in range(steps):
        step_time = time.time()
        xb, yb = get_batch(train_data, train_config.batch_size, train_config.block_size)
        xb = xb.to(device, dtype=torch.long)
        yb = yb.to(device, dtype=torch.long)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Simple validation
        if step % train_config.eval_interval == 0 or step == 1:
            val_x, val_y = get_batch(val_data, train_config.batch_size, train_config.block_size)
            val_x = val_x.to(device, dtype=torch.long)
            val_y = val_y.to(device, dtype=torch.long)
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)), val_y.view(-1)
                )
            print(f"Time Spent {time.time() - step_time:.2f}s")
            print(
                f"Step {step:4d}/{steps}: train loss {loss.item():.4f} | val loss {val_loss.item():.4f}"
            )

    # Save model, vocab, config
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "stoi": stoi,
            "itos": itos,
            "config": config.__dict__,
        },
        "tinygpt_bpe/checkpoint.pt",
    )
    print("Training complete. Model saved to tinygpt_bpe/checkpoint.pt")


if __name__ == "__main__":
    train()
