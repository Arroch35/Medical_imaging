from torch import nn as nn

class Embedder(nn.Module):
    """
    Maps latent vectors → contrastive embedding space
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64)   # final embedding
        )

    def forward(self, x):
        return self.net(x)