from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4
    latent_dim: int = 64
    device: str = "cuda"
    lambda_schedule: str = "linear"  # "linear" | "logistic" | "delayed"
