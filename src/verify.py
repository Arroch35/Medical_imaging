import torch

#verfy that i'm using cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")