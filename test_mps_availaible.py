import torch

print("Torch version:", torch.version)

mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()

print("MPS available:", mps_ok)

device = torch.device("mps" if mps_ok else "cpu")

x = torch.randn(1024, 1024, device=device)

y = x @ x.t()

print("Device used:", device)

print("y mean:", y.mean().item())

