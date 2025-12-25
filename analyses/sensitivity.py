# sensitivity.py
# -*- coding: utf-8 -*-
"""
Run 3-condition sensitivity analysis for lambda scheduling:
  - linear / logistic / delayed
Outputs:
  - Combined CSV: results/sensitivity/lambda_sensitivity.csv
  - Per-condition CSV/PNG via train.simulate_training
  - Optional overlay PNG for λ curves and total_loss
"""

from __future__ import annotations
import csv
from pathlib import Path
from train import simulate_training

def run_sensitivity(
    total_steps: int = 400,
    reach_portion: float = 0.30,
    window_portion: float = 0.05,
    sharpness: float = 12.0,
    seed: int = 42,
    beta: float = 0.10,
    outdir: str = "results/sensitivity"
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    per_run_dirs = {
        "linear":   out / "linear",
        "logistic": out / "logistic",
        "delayed":  out / "delayed",
    }

    for kind, odir in per_run_dirs.items():
        logs, csv_path = simulate_training(
            total_steps=total_steps,
            lambda_kind=kind,
            reach_portion=reach_portion,
            lambda_window_portion=window_portion,
            lambda_sharpness=sharpness,
            seed=seed,
            outdir=str(odir),
            beta=beta
        )
        for r in logs:
            rows.append({
                "step": r["step"],
                "lambda_kind": kind,
                "lambda": r["lambda"],
                "adv_loss": r["adv_loss"],
                "feature_reward": r["feature_reward"],
                "total_loss": r["total_loss"],
            })

    # Save combined CSV
    comb_csv = out / f"lambda_sensitivity_T{total_steps}_p{int(100*reach_portion)}.csv"
    with comb_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lambda_kind", "lambda", "adv_loss", "feature_reward", "total_loss"])
        w.writeheader()
        w.writerows(rows)

    # Optional overlay plots
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(rows)

        # λ overlay
        plt.figure()
        for k in ["linear", "logistic", "delayed"]:
            sub = df[df["lambda_kind"] == k]
            plt.plot(sub["step"], sub["lambda"], label=k)
        plt.title("Lambda schedules (overlay)")
        plt.xlabel("step")
        plt.ylabel("lambda")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "overlay_lambda.png", dpi=160)
        plt.close()

        # total_loss overlay
        plt.figure()
        for k in ["linear", "logistic", "delayed"]:
            sub = df[df["lambda_kind"] == k]
            plt.plot(sub["step"], sub["total_loss"], label=k)
        plt.title("Total loss (overlay)")
        plt.xlabel("step")
        plt.ylabel("total_loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "overlay_total_loss.png", dpi=160)
        plt.close()
    except Exception as e:
        print("[WARN] plotting overlay skipped:", e)

    print(f"[sensitivity] combined CSV: {comb_csv}")
    return str(comb_csv)

if __name__ == "__main__":
    run_sensitivity()
