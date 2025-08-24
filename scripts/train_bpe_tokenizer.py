from tokenizers import ByteLevelBPETokenizer
import os

os.makedirs("tinygpt_bpe/bpe_tokenizer", exist_ok=True)
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["tiny_stories.txt"],
    vocab_size=1024,  # or higher for more complexity; 1000 is great for small models
    min_frequency=2,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
)
tokenizer.save_model("tinygpt_bpe/bpe_tokenizer/")
print("BPE tokenizer trained and saved to tinygpt_bpe/bpe_tokenizer/")
