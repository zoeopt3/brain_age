"""Robustness check plots."""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_eo_vs_ec(results: dict, out_path: str | Path) -> None:
    """Grouped bar chart comparing EO, EC, and combined MAE."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels, maes, stds = [], [], []
    for cond in ["EO", "EC", "combined"]:
        if cond in results and "mae_mean" in results[cond]:
            labels.append(cond)
            maes.append(results[cond]["mae_mean"])
            stds.append(results[cond]["mae_std"])

    if not labels:
        return

    colors = {"EO": "#4c72b0", "EC": "#dd8452", "combined": "#55a868"}
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(labels))
    ax.bar(x, maes, yerr=stds, capsize=5, color=[colors.get(l, "#999") for l in labels],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (years)")
    ax.set_title("EO vs EC vs Combined Performance")
    for i, (m, s) in enumerate(zip(maes, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_family(results: dict, out_path: str | Path) -> None:
    """Bar chart comparing MAE across feature subsets."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = [(k, v) for k, v in results.items() if isinstance(v, dict) and "mae_mean" in v]
    if not items:
        return

    names = [k.replace("_", " ").title() for k, _ in items]
    maes = [v["mae_mean"] for _, v in items]
    stds = [v["mae_std"] for _, v in items]
    n_feat = [v.get("n_features", "?") for _, v in items]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 4.5))
    x = range(len(names))
    colors = ["#55a868" if "all" in names[i].lower() else "#4c72b0" for i in x]
    ax.bar(x, maes, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({nf} feat)" for n, nf in zip(names, n_feat)],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Feature Family Sensitivity")
    for i, (m, s) in enumerate(zip(maes, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
