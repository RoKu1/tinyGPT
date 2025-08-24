import time
import torch
import torch.nn.functional as F
from tinygpt.data import load_shakespeare, build_vocab, encode, get_train_val_splits, get_batch
from tinygpt.model import GPTConfig, TinyGPT

def train(
    epochs=1, batch_size=128, block_size=64, n_embd=128, n_layers=4, n_heads=4, lr=3e-4, eval_interval=200
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare data
    text = load_shakespeare()
    vocab, stoi, itos = build_vocab(text)
    encoded = encode(text, stoi)
    train_ids, val_ids = get_train_val_splits(encoded)
    print(f"Vocab size: {len(vocab)}. Train size: {len(train_ids)}, Val size: {len(val_ids)}")

    config = GPTConfig(vocab_size=len(vocab), block_size=block_size, n_embd=n_embd, n_layers=n_layers, n_heads=n_heads)
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    steps = epochs * (len(train_ids) // batch_size)
    print(f"Training for {steps} steps...")

    for step in range(steps):
        step_time = time.time()
        xb, yb = get_batch(train_ids, batch_size, block_size)
        xb = torch.tensor(xb, dtype=torch.long, device=device)
        yb = torch.tensor(yb, dtype=torch.long, device=device)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Simple validation
        if step % eval_interval == 0 or step == 1:
            val_x, val_y = get_batch(val_ids, batch_size, block_size)
            val_x = torch.tensor(val_x, dtype=torch.long, device=device)
            val_y = torch.tensor(val_y, dtype=torch.long, device=device)
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
            print(f"Time Spent {time.time() - step_time:.2f}s")
            print(f"Step {step:4d}/{steps}: train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")

    # Save model, vocab, config
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "stoi": stoi,
            "itos": itos,
            "config": config.__dict__,
        },
        "tinygpt/checkpoint.pt"
    )
    print("Training complete. Model saved to tinygpt/checkpoint.pt")

if __name__ == "__main__":
    train()
