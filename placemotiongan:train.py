from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .model import Generator, Discriminator
from .scheduler import LambdaScheduler
from .config import TrainingConfig

@dataclass
class TrainingState:
    g_losses: list
    d_losses: list
    lambdas: list


class Trainer:
    """High-level training loop for PlaceEmotion-GAN."""

    def __init__(
        self,
        config: TrainingConfig,
        data: torch.Tensor,
        lambda_scheduler: Optional[LambdaScheduler] = None,
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.dataset = TensorDataset(data)
        self.loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)

        input_dim = data.shape[1]
        self.G = Generator(config.latent_dim, input_dim).to(self.device)
        self.D = Discriminator(input_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=config.lr_generator, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.lr_discriminator, betas=(0.5, 0.999))

        total_steps = config.epochs * len(self.loader)
        self.lambda_scheduler = lambda_scheduler or LambdaScheduler(total_steps, config.lambda_schedule)
        self.bce = torch.nn.BCELoss()

    def train(self) -> TrainingState:
        step = 0
        g_losses, d_losses, lambdas = [], [], []

        for epoch in range(self.config.epochs):
            for (x_real,) in self.loader:
                x_real = x_real.to(self.device)
                bsz = x_real.size(0)

                # 1. Update D
                self.opt_D.zero_grad()
                z = torch.randn(bsz, self.config.latent_dim, device=self.device)
                x_fake = self.G(z).detach()

                y_real = torch.ones(bsz, 1, device=self.device)
                y_fake = torch.zeros(bsz, 1, device=self.device)

                d_real = self.D(x_real)
                d_fake = self.D(x_fake)

                loss_D = self.bce(d_real, y_real) + self.bce(d_fake, y_fake)
                loss_D.backward()
                self.opt_D.step()

                # 2. Update G
                self.opt_G.zero_grad()
                z = torch.randn(bsz, self.config.latent_dim, device=self.device)
                x_fake = self.G(z)
                d_fake = self.D(x_fake)

                lam = float(self.lambda_scheduler(step))
                loss_G = self.bce(d_fake, y_real)  # ここに lam を絡めた正則化項などを追加
                loss_G.backward()
                self.opt_G.step()

                g_losses.append(loss_G.item())
                d_losses.append(loss_D.item())
                lambdas.append(lam)

                step += 1

        return TrainingState(g_losses=g_losses, d_losses=d_losses, lambdas=lambdas)
