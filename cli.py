import click
import torch
from tinygpt.generate import load_model, sample
from tinygpt.data import decode


@click.group()
def cli():
    """tinyGPT - a mini transformer language model (char-level)"""
    pass


@cli.command()
def hello():
    """Say hello from your CLI app."""
    click.echo("👋 Hello from tinyGPT CLI!")


@click.command()
@click.option("--prompt", default="ROMEO:", help="Prompt string to begin generation.")
@click.option(
    "--num_tokens", default=100, help="How many new tokens(chrs) to generate."
)
@click.option(
    "--temperature", default=0.6, help="Sampling temperature (higher = more random)."
)
@click.option(
    "--top_k", default=None, type=int, help="Limit sampling to top_k candidates."
)
def generate(prompt, num_tokens, temperature, top_k):
    """Generate text using the trained TinyGPT model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, stoi, itos = load_model(device=device)
    idx = [stoi.get(c, 0) for c in prompt]
    out = sample(
        model,
        idx,
        num_tokens=num_tokens,
        stoi=stoi,
        itos=itos,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    print(decode(out, itos))


cli.add_command(generate)


if __name__ == "__main__":
    cli()
