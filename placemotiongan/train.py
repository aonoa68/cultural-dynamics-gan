# train.py
# -*- coding: utf-8 -*-
"""
Minimal-yet-solid training loop (simulation) for PlaceEmotion-GAN JOSS package.
- Reproduces λ scheduling dynamics over T steps.
- Generates dummy adv_loss and feature_reward with controlled stochasticity.
- Logs to CSV; optionally saves PNG curves if matplotlib is available.

CLI Examples:
    python train.py --lambda-kind linear
    python train.py --lambda-kind logistic --lambda-sharpness 12 --lambda-window-portion 0.05
    python train.py --lambda-kind delayed  --total-steps 400 --reach-portion 0.30
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

from lambda_scheduler import make_scheduler
from losses import tradeoff_loss, LossConfig


def simulate_training(
    total_steps: int = 400,
    lambda_kind: str = "linear",
    reach_portion: float = 0.30,
    lambda_window_portion: float = 0.05,
    lambda_sharpness: float = 12.0,
    seed: int = 42,
    outdir: str = "results/run_linear",
    beta: float = 0.10
):
    """
    Simulate GAN learning dynamics with a paper-consistent loss formulation.
    Returns a dict of arrays for plotting/analysis and writes CSV logs.
    """
    rng = np.random.default_rng(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # λ scheduler
    sched = make_scheduler(
        total_steps=total_steps,
        reach_portion=reach_portion,
        kind=lambda_kind,
        window_portion=lambda_window_portion,
        sharpness=lambda_sharpness
    )

    # Initialize “latent” process states for adv_loss & feature_reward
    # adv_loss starts higher and tends to decrease (with noise)
    adv = 0.80 + 0.05 * rng.standard_normal()
    # feature_reward starts low and tends to increase (with noise)
    rew = 0.10 + 0.03 * rng.standard_normal()

    # Time constants (heuristics to yield pleasant curves)
    adv_decay = 0.0035
    rew_rise = 0.0040

    logs = []
    loss_cfg = LossConfig(beta=beta)

    for t in range(total_steps):
        lam = sched(t)

        # Stochastic dynamics:
        # - adv_loss decays but with noise & small rebounds
        adv_noise = 0.02 * rng.standard_normal()
        adv = max(0.0, adv * (1.0 - adv_decay) + adv_noise)
        # - feature reward grows then saturates a bit
        rew_noise = 0.02 * rng.standard_normal()
        rew = max(0.0, rew + rew_rise * (1.0 - min(1.0, rew)) + rew_noise)

        total = tradeoff_loss(adv, rew, lam, loss_cfg)

        logs.append({
            "step": t,
            "lambda": lam,
            "adv_loss": adv,
            "feature_reward": rew,
            "total_loss": total
        })

    # CSV save
    csv_path = out / f"training_{lambda_kind}_T{total_steps}_p{int(100*reach_portion)}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lambda", "adv_loss", "feature_reward", "total_loss"])
        w.writeheader()
        w.writerows(logs)

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        xs = [r["step"] for r in logs]
        ys_lambda = [r["lambda"] for r in logs]
        ys_adv = [r["adv_loss"] for r in logs]
        ys_rew = [r["feature_reward"] for r in logs]
        ys_total = [r["total_loss"] for r in logs]

        # λ curve
        plt.figure()
        plt.plot(xs, ys_lambda, label="lambda")
        plt.title(f"Lambda ({lambda_kind})")
        plt.xlabel("step")
        plt.ylabel("lambda")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"lambda_{lambda_kind}.png", dpi=160)
        plt.close()

        # Loss/Reward curves
        plt.figure()
        plt.plot(xs, ys_total, label="total_loss")
        plt.plot(xs, ys_adv, label="adv_loss")
        plt.plot(xs, ys_rew, label="feature_reward")
        plt.title(f"Training dynamics ({lambda_kind})")
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"dynamics_{lambda_kind}.png", dpi=160)
        plt.close()
    except Exception as e:
        print("[WARN] matplotlib not available or plotting failed:", e)

    print(f"[{lambda_kind}] T={total_steps}, reach={reach_portion} → CSV: {csv_path}")
    return logs, str(csv_path)


def main():
    ap = argparse.ArgumentParser(description="PlaceEmotion-GAN λ-scheduling simulation")
    ap.add_argument("--total-steps", type=int, default=400)
    ap.add_argument("--reach-portion", type=float, default=0.30)
    ap.add_argument("--lambda-kind", type=str, default="linear", choices=["linear", "logistic", "delayed"])
    ap.add_argument("--lambda-window-portion", type=float, default=0.05)
    ap.add_argument("--lambda-sharpness", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--outdir", type=str, default="results/run_linear")
    args = ap.parse_args()

    simulate_training(
        total_steps=args.total_steps,
        lambda_kind=args.lambda_kind,
        reach_portion=args.reach_portion,
        lambda_window_portion=args.lambda_window_portion,
        lambda_sharpness=args.lambda_sharpness,
        seed=args.seed,
        outdir=args.outdir,
        beta=args.beta
    )

if __name__ == "__main__":
    main()
