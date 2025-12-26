"""
placemotiongan: GAN-based model for cultural dynamics and emotional niche construction.
"""

from .model import Generator, Discriminator
from .scheduler import LambdaScheduler, LambdaConfig, make_scheduler
from .train import simulate_training

__all__ = [
    "Generator",
    "Discriminator",
    "LambdaScheduler",
    "LambdaConfig",
    "make_scheduler",
    "simulate_training",
]
