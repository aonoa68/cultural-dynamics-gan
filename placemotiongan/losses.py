# losses.py
# -*- coding: utf-8 -*-
"""
Loss helpers for the PlaceEmotion-GAN simulation.
Implements the paper-consistent trade-off:
    total_loss = (1 - λ) * adv_loss - λ * β * feature_reward
Note:
- adv_loss: "小さいほど良い" ものとして扱う（例：識別器のloss的イメージ）
- feature_reward: "大きいほど良い"（理論駆動の望ましい特徴に近いことへの報酬）
- β: スケール調整パラメータ
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LossConfig:
    beta: float = 0.10  # 特徴報酬のスケール（論文の強度感に合わせ調整可能）

def tradeoff_loss(adv_loss: float, feature_reward: float, lam: float, cfg: LossConfig) -> float:
    """
    Compute total loss under the trade-off schedule.
    Smaller is better.
    """
    lam = max(0.0, min(1.0, lam))
    return (1.0 - lam) * adv_loss - lam * cfg.beta * feature_reward
