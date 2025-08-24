import click
import torch
import random
import sys
from tinygpt_char.generate import load_model as load_model_char, sample as sample_char
from tinygpt_char.data import decode as char_decode
from tinygpt_char.train import train as train_char_model
from tinygpt_bpe.generate import sample as sample_bpe, load_model as load_model_bpe
from tinygpt_bpe.train import train as train_bpe_model

SURPRISE_PROMPTS = [
    "Once upon a time",
    "The robot said",
    "In the distant future",
    "She opened the door and",
    "The jungle was silent",
    "Suddenly, the lights went out",
    "My grandmother once told me",
    "On the edge of the city",
]

@click.group()
def cli():
    """tinyGPT CLI: Generate text with your character-level or BPE transformer models."""
    pass

@cli.command()
def hello():
    """Say hello from your CLI app."""
    click.echo("👋 Hello from tinyGPT CLI!")


# ============================================================================================================
# Char based
# ============================================================================================================


@cli.command("train-char")
@click.confirmation_option(prompt="About to retrain and overwrite model—continue?")
def train_char():
    """Train the char-level TinyGPT model."""
    train_char_model()
    click.secho(f"Char model trained and saved to tinygpt_char/checkpoint.pt", fg="green")


@cli.command("generate-char")
@click.option("--prompt", default="ROMEO:", help="Prompt to begin char-level generation. Use 'surprise' for a random one.")
@click.option("--num_tokens", default=100, help="How many new tokens/chrs to generate.")
@click.option("--temperature", default=0.6, show_default=True, help="Sampling temperature (higher = more random).")
@click.option("--top_k", default=None, type=int, help="Limit sampling to top_k candidates.")
@click.option("--out", type=click.Path(), help="Write generated text to file.")
def generate_char(prompt, num_tokens, temperature, top_k, out):
    """Generate text using the trained TinyGPT character-level model."""
    if prompt == "surprise" or not prompt.strip():
        prompt = random.choice(SURPRISE_PROMPTS)
    temperature = max(0.1, min(temperature, 2.0))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model, stoi, itos = load_model_char(device=device)
    except Exception as e:
        click.echo(f"Error loading char model: {e}")
        raise click.Abort()

    idx = [stoi.get(c, 0) for c in prompt]
    out_text = sample_char(
        model,
        idx,
        num_tokens=num_tokens,
        stoi=stoi,
        itos=itos,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    out_text = char_decode(out_text, itos)

    click.echo(f"\nPROMPT: \"{prompt}\"")
    click.echo("MODEL: Char | Tokens: %d | Temp: %.2f\n" % (num_tokens, temperature))
    click.echo("-" * 40 + "\nGENERATED TEXT:\n")
    click.echo(out_text)
    if out:
        with open(out, "w") as f:
            f.write(out_text)
        click.echo(f"\nOutput written to {out}")


# ============================================================================================================
# BPE based
# ============================================================================================================


@cli.command("train-bpe")
@click.confirmation_option(prompt="About to retrain and overwrite model—continue?")
def train_bpe():
    """Train the BPE TinyGPT model."""
    train_bpe_model()
    click.secho(f"BPE model trained and saved to tinygpt_bpe/checkpoint.pt", fg="green")


@cli.command("generate-bpe")
@click.option("--prompt", default="Once upon a time", help="Prompt to begin BPE generation. Use 'surprise' for a random one.")
@click.option("--num_tokens", default=100, help="How many BPE tokens to generate.")
@click.option("--temperature", default=1.0, show_default=True, help="Sampling temperature.")
@click.option("--block_size", default=64, help="Block size (context window).")
@click.option("--checkpoint", default="tinygpt_bpe/checkpoint.pt", help="Path to BPE model checkpoint.")
@click.option("--out", type=click.Path(), help="Write generated text to file.")
def generate_bpe(prompt, num_tokens, temperature, block_size, checkpoint, out):
    """Generate text using the trained TinyGPT BPE model."""
    if prompt == "surprise" or not prompt.strip():
        prompt = random.choice(SURPRISE_PROMPTS)
    temperature = max(0.1, min(temperature, 2.0))
    block_size = max(8, min(int(block_size), 512))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model, tokenizer = load_model_bpe(checkpoint, device)
    except FileNotFoundError:
        click.echo("Error: BPE model checkpoint not found! Please train your model first.")
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error loading BPE model: {e}")
        raise click.Abort()
    try:
        out_text = sample_bpe(
            model,
            tokenizer,
            prompt,
            steps=num_tokens,
            block_size=block_size,
            temperature=temperature,
        )
    except Exception as e:
        click.echo(f"Error during generation: {e}")
        raise click.Abort()

    ckpt_info = getattr(model, "config", None)
    click.echo(f"\nPROMPT: \"{prompt}\"")
    if ckpt_info:
        click.echo(f"MODEL: BPE | Tokens: {num_tokens} | Temp: {temperature} | Layers: {ckpt_info.n_layers} | Embedding: {ckpt_info.n_embd}")
    else:
        click.echo("MODEL: BPE | Tokens: %d | Temp: %.2f" % (num_tokens, temperature))
    click.echo("-" * 40 + "\nGENERATED TEXT:\n")
    click.echo(out_text)
    if out:
        with open(out, "w") as f:
            f.write(out_text)
        click.echo(f"\nOutput written to {out}")



@cli.command("chat")
@click.option("--char_ckpt", default="tinygpt_char/checkpoint.pt", help="Path to char model checkpoint.")
@click.option("--bpe_ckpt", default="tinygpt_bpe/checkpoint.pt", help="Path to BPE model checkpoint.")
@click.option("--block_size", default=64, help="Block size (BPE context window).")
def chat(char_ckpt, bpe_ckpt, block_size):
    """Start interactive chat mode!"""
    click.echo("tinyGPT Chat Mode 🤖")
    click.echo("Select model: [1] Char-level  [2] BPE")
    model_type = None
    while model_type not in ('1', '2'):
        model_type = input("Enter model (1 or 2): ").strip()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model once
    if model_type == '1':
        click.echo("\nLoading char model...")
        try:
            model, stoi, itos = load_model_char(char_ckpt, device)
            click.echo("Loaded char model 👍")
        except Exception as e:
            click.secho(f"Error loading char model: {e}", fg="red")
            sys.exit(1)
        history = ""
    else:
        click.echo("\nLoading BPE model...")
        try:
            model, tokenizer = load_model_bpe(bpe_ckpt, device)
            click.echo("Loaded BPE model 👍")
        except Exception as e:
            click.secho(f"Error loading BPE model: {e}", fg="red")
            sys.exit(1)
        block_size = int(block_size)
        history = ""

    # Chat loop
    click.secho("\nType your prompt below. Enter 'exit' or 'quit' to end.\n", fg="green")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                click.echo("\nBye! 👋")
                break

            # (Optionally) Append turn to context/history
            if model_type == '1':
                prompt = (history + user_input)[-block_size:]  # truncate to context size if char
                idx = [stoi.get(c, 0) for c in prompt]
                out = sample_char(
                    model,
                    idx,
                    num_tokens=80,
                    stoi=stoi,
                    itos=itos,
                    temperature=0.6,
                    top_k=None,
                    device=device,
                )
                response = char_decode(out, itos).replace(prompt, "", 1).strip()
                history = (prompt + response)[-block_size*2:] # keep rolling context
            else:
                prompt = (history + user_input)[-block_size:]
                out = sample_bpe(
                    model,
                    tokenizer,
                    prompt,
                    steps=80,
                    block_size=block_size,
                    temperature=1.0,
                )
                response = out.replace(prompt, "", 1).strip()
                history = (prompt + response)[-block_size*2:]

            click.secho(f"\ntinyGPT: {response}\n", fg="yellow")
        except (KeyboardInterrupt, EOFError):
            click.echo("\nBye! 👋")
            break
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")



if __name__ == "__main__":
    cli()
