import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = "tiny_shakespeare.txt"


def main():
    r = requests.get(URL)
    r.raise_for_status()
    with open(DEST, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"Downloaded to {DEST}")


if __name__ == "__main__":
    main()
