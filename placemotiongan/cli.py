import argparse
import torch
import pandas as pd

from .config import TrainingConfig
from .scheduler import LambdaScheduler
from .train import Trainer


def main():
    parser = argparse.ArgumentParser(
        prog="placemotiongan",
        description="Train and run PlaceEmotion-GAN for cultural dynamics.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file containing training data (rows=samples, columns=features).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-schedule", type=str, default="linear",
                        choices=["linear", "logistic", "delayed"])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    df = pd.read_csv(args.data)
    x = torch.tensor(df.values, dtype=torch.float32)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_schedule=args.lambda_schedule,
        device=args.device,
    )
    trainer = Trainer(config, x, LambdaScheduler(config.epochs * (len(x) // config.batch_size + 1),
                                                 mode=config.lambda_schedule))
    state = trainer.train()

    print(f"Training finished. G loss (last): {state.g_losses[-1]:.4f}")
