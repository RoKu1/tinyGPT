import torch
from tinygpt_bpe.model import GPTConfig, TinyGPT
from tinygpt_bpe.data import get_tokenizer, encode, decode

def load_model(checkpoint_path="tinygpt_bpe/checkpoint.pt", device="mps"):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = TinyGPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, get_tokenizer()


def sample(model, tokenizer, prompt, steps=100, block_size=64, temperature=1.0, device="mps"):
    ids = encode(prompt, tokenizer)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(steps):
        if input_ids.size(1) > block_size:
            input_ids = input_ids[:, -block_size:]
        with torch.no_grad():
            logits = model(input_ids)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    return decode(input_ids[0].tolist(), tokenizer)

if __name__ == "__main__":
    prompt = "Once upon a time"
    model, tokenizer = load_model()
    generated = sample(model, tokenizer, prompt)
    print(f"Prompt: {prompt}\nGenerated:\n{generated}")
