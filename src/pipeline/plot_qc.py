"""QC summary plots: distributions of key quality metrics.

Generates lightweight PNGs for visual review before proceeding to
feature extraction.  All plots use matplotlib with no interactive
backend so they work in headless / CI environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_qc_plots(
    qc_df: pd.DataFrame,
    out_dir: str | Path,
    seed: int = 1337,
) -> list[str]:
    """Generate all QC summary figures.

    Args:
        qc_df: DataFrame from qc_summary.parquet.
        out_dir: Directory to write PNGs.
        seed: Random seed for reproducible example selection.

    Returns:
        List of paths to generated figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    if qc_df.empty:
        return created

    # 1. Usable duration histogram
    if "usable_duration_sec" in qc_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = qc_df["usable_duration_sec"].dropna()
        ax.hist(vals, bins=20, color="#4c72b0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Usable Duration (sec)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Usable Duration After Artifact Rejection")
        ax.axvline(vals.median(), color="#c44e52", ls="--", label=f"median = {vals.median():.0f}s")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "usable_duration_hist.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p))

    # 2. Rejection fraction histogram
    if "rejection_fraction" in qc_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = qc_df["rejection_fraction"].dropna()
        ax.hist(vals, bins=20, color="#dd8452", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Rejection Fraction")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Epoch Rejection Fraction")
        ax.axvline(vals.median(), color="#c44e52", ls="--", label=f"median = {vals.median():.2f}")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "rejection_fraction_hist.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p))

    # 3. Line-noise ratio histogram
    if "line_noise_ratio" in qc_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = qc_df["line_noise_ratio"].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=20, color="#55a868", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Line-Noise Ratio (power at line freq / neighbours)")
            ax.set_ylabel("Count")
            ax.set_title("Line-Noise Proxy Distribution")
            ax.axvline(1.0, color="gray", ls=":", label="ratio = 1 (no excess noise)")
            ax.legend()
            fig.tight_layout()
            p = out_dir / "line_noise_ratio_hist.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))
        else:
            plt.close(fig)

    # 4. Age vs usable duration scatter
    if "age" in qc_df.columns and "usable_duration_sec" in qc_df.columns:
        valid = qc_df.dropna(subset=["age", "usable_duration_sec"])
        if len(valid) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(valid["age"], valid["usable_duration_sec"],
                       alpha=0.6, s=30, color="#8172b2", edgecolor="white")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Usable Duration (sec)")
            ax.set_title("Age vs. Usable Duration")
            fig.tight_layout()
            p = out_dir / "age_vs_duration.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))

    # 5. Muscle ratio histogram
    if "muscle_ratio" in qc_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = qc_df["muscle_ratio"].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=20, color="#ccb974", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Muscle Ratio (30-45 Hz power / 8-13 Hz power)")
            ax.set_ylabel("Count")
            ax.set_title("Muscle Artifact Proxy Distribution")
            fig.tight_layout()
            p = out_dir / "muscle_ratio_hist.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))
        else:
            plt.close(fig)

    return created
