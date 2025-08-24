from datasets import load_dataset

# Download TinyStories and save as a plain text file (entire set)
dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Total stories: {len(dataset)}")

with open("full_tiny_stories.txt", "w", encoding="utf-8") as fa:
    for example in dataset:
        fa.write(example["text"].strip().replace("\n", " ") + "\n")
print("TinyStories saved as full_tiny_stories.txt")

# Downsampling to 50,000 stories
sample = 50000
subset = dataset[:sample]
with open("tiny_stories.txt", "w", encoding="utf-8") as fe:
    for story in subset["text"]:  # subset is a dict of lists
        fe.write(story.strip().replace("\n", " ") + "\n")
print("TinyStories downsampled to 50K stories and saved as tiny_stories.txt")
