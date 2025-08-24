import torch
from tinygpt.model import GPTConfig, TinyGPT
from tinygpt.data import decode


def sample(
    model, idx, num_tokens, stoi, itos, temperature=1.0, top_k=None, device="cpu"
):
    model.eval()
    idx = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(num_tokens):
        idx_cond = idx[:, -model.pos_emb.num_embeddings :]  # last block_size chars
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # last time step
        if top_k is not None:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[..., -1, None]] = -float("Inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx[0].tolist()


def load_model(checkpoint_path="tinygpt/checkpoint.pt", device="mps"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint["stoi"], checkpoint["itos"]


if __name__ == "__main__":
    model, stoi, itos = load_model()
    prompt = "ROMEO:"
    idx = [stoi.get(c, 0) for c in prompt]
    out = sample(model, idx, num_tokens=200, stoi=stoi, itos=itos, device="mps")
    print(decode(out, itos))
