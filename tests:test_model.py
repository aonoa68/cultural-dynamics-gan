import torch
from placemotiongan.model import Generator, Discriminator

def test_generator_output_dim():
    G = Generator(latent_dim=8, output_dim=4)
    z = torch.randn(2, 8)
    out = G(z)
    assert out.shape == (2, 4)

def test_discriminator_output_dim():
    D = Discriminator(input_dim=4)
    x = torch.randn(2, 4)
    out = D(x)
    assert out.shape == (2, 1)
