import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator network for PlaceEmotion-GAN.
    Maps latent variables to cultural-emotional feature representations.
    """
    
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Minimal GAN discriminator."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
