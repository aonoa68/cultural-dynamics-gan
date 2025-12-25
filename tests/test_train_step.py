import torch
from placemotiongan.config import TrainingConfig
from placemotiongan.train import Trainer
from placemotiongan.scheduler import LambdaScheduler

def test_single_epoch_trains():
    x = torch.randn(16, 4)
    cfg = TrainingConfig(epochs=1, batch_size=4, latent_dim=8, device="cpu")
    trainer = Trainer(cfg, x, LambdaScheduler(total_steps=4, mode="linear"))
    state = trainer.train()
    assert len(state.g_losses) > 0
