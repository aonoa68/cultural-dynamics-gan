"""
placemotiongan: GAN-based model for cultural dynamics and emotional niche construction.
"""

from .model import Generator, Discriminator
from .scheduler import LambdaScheduler
from .train import Trainer

__all__ = ["Generator", "Discriminator", "LambdaScheduler", "Trainer"]
__version__ = "1.0.0"
