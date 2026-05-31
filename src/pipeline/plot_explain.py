"""Explainability plots: importance bars, ablation comparison, SHAP summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_top_importances(
    imp_df: pd.DataFrame,
    out_path: str | Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top features by permutation importance."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = imp_df.head(top_n).iloc[::-1]  # reverse for bottom-to-top

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    y_pos = range(len(df))
    ax.barh(y_pos, df["importance_mean"], xerr=df["importance_std"],
            color="#4c72b0", alpha=0.85, capsize=3, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlabel("Permutation Importance (MAE increase)")
    ax.set_title(f"Top {len(df)} Features by Permutation Importance")
    ax.axvline(0, color="k", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_mae(
    results: dict,
    out_path: str | Path,
) -> None:
    """Bar chart comparing MAE across ablation experiments."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    maes = [results[n].get("mae_mean", float("nan")) for n in names]
    stds = [results[n].get("mae_std", 0) for n in names]

    valid = [(n, m, s) for n, m, s in zip(names, maes, stds) if not np.isnan(m)]
    if not valid:
        return

    names, maes, stds = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    x = range(len(names))
    colors = ["#55a868" if n == "full_model" else "#4c72b0" for n in names]
    ax.bar(x, maes, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE (years)")
    ax.set_title("Ablation Study: MAE by Feature Set")
    for i, (m, s) in enumerate(zip(maes, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shap_summary(
    shap_df: pd.DataFrame,
    out_path: str | Path,
    top_n: int = 20,
) -> None:
    """Simple bar chart of mean |SHAP| per feature."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    df = mean_abs.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    y_pos = range(len(df))
    ax.barh(y_pos, df.values, color="#dd8452", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {len(df)} Features by SHAP Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
