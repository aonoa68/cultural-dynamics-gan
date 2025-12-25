# lambda_scheduler.py
# -*- coding: utf-8 -*-
"""
Lambda scheduling utilities for PlaceEmotion-GAN (simulation version).
Implements 3 shapes: linear / logistic / delayed, with 30% reach-to-1.0 default.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class LambdaConfig:
    total_steps: int = 400
    reach_portion: float = 0.30     # λ≈1.0 に到達させたい割合（Tの何割か）
    kind: str = "linear"            # "linear" | "logistic" | "delayed"
    window_portion: float = 0.05    # 立ち上がり幅（delayed/logistic）
    sharpness: float = 12.0         # logisticの鋭さ（大きいほど急）

class LambdaScheduler:
    """
    Scheduler that returns λ(t) in [0,1] for a given step t.
    - linear: 0→1 を reach_portion*T までに線形到達
    - logistic: シグモイド（中心 ~ t_star - w/2）
    - delayed: しばらく0→短い窓幅wで線形→1
    """
    def __init__(self, cfg: LambdaConfig):
        self.cfg = cfg
        self.T = int(max(1, cfg.total_steps))
        self.t_star = max(1, int(self.T * cfg.reach_portion))
        self.w = max(1, int(self.T * cfg.window_portion))
        # logistic用：鋭さをt_starで相対スケーリング
        self.k = cfg.sharpness / float(self.t_star)

    def __call__(self, t: int) -> float:
        t = max(0, min(t, self.T - 1))
        if self.cfg.kind == "linear":
            return min(1.0, (t + 0.0) / self.t_star)

        elif self.cfg.kind == "logistic":
            t0 = self.t_star - self.w / 2.0
            return 1.0 / (1.0 + math.exp(- self.k * (t - t0)))

        elif self.cfg.kind == "delayed":
            if t < (self.t_star - self.w):
                return 0.0
            elif t < self.t_star:
                return (t - (self.t_star - self.w)) / float(self.w)
            else:
                return 1.0
        else:
            raise ValueError(f"Unknown kind: {self.cfg.kind}")

def make_scheduler(
    total_steps: int = 400,
    reach_portion: float = 0.30,
    kind: str = "linear",
    window_portion: float = 0.05,
    sharpness: float = 12.0
) -> LambdaScheduler:
    return LambdaScheduler(LambdaConfig(
        total_steps=total_steps,
        reach_portion=reach_portion,
        kind=kind.lower(),
        window_portion=window_portion,
        sharpness=sharpness
    ))
