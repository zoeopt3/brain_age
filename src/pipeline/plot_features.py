"""Feature-stage plots: PAF distributions, age relationships, PSD examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_feature_plots(
    feat_df: pd.DataFrame,
    out_dir: str | Path,
    seed: int = 1337,
) -> list[str]:
    """Generate all Stage 3 summary figures.

    Args:
        feat_df: Features DataFrame.
        out_dir: Output directory for PNGs.
        seed: Random seed for reproducible example selection.

    Returns:
        List of paths to generated figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    if feat_df.empty:
        return created

    # 1. Age vs PAF scatter
    if "age" in feat_df.columns and "paf_hz" in feat_df.columns:
        valid = feat_df.dropna(subset=["age", "paf_hz"])
        if len(valid) >= 3:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(valid["age"], valid["paf_hz"],
                       alpha=0.6, s=35, color="#4c72b0", edgecolor="white")
            # Trendline
            z = np.polyfit(valid["age"], valid["paf_hz"], 1)
            x_line = np.linspace(valid["age"].min(), valid["age"].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#c44e52", ls="--",
                    label=f"trend: {z[0]:+.2f} Hz/yr")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Peak Alpha Frequency (Hz)")
            ax.set_title("Age vs. Peak Alpha Frequency")
            ax.legend()
            fig.tight_layout()
            p = out_dir / "age_vs_paf.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))

    # 2. PAF histogram
    if "paf_hz" in feat_df.columns:
        vals = feat_df["paf_hz"].dropna()
        if len(vals) >= 3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(vals, bins=20, color="#55a868", edgecolor="white", alpha=0.85)
            ax.axvline(vals.median(), color="#c44e52", ls="--",
                       label=f"median = {vals.median():.1f} Hz")
            ax.set_xlabel("Peak Alpha Frequency (Hz)")
            ax.set_ylabel("Count")
            ax.set_title("PAF Distribution")
            ax.legend()
            fig.tight_layout()
            p = out_dir / "paf_histogram.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))

    # 3. Aperiodic exponent distribution
    if "ap_exponent" in feat_df.columns:
        vals = feat_df["ap_exponent"].dropna()
        if len(vals) >= 3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(vals, bins=20, color="#8172b2", edgecolor="white", alpha=0.85)
            ax.axvline(vals.median(), color="#c44e52", ls="--",
                       label=f"median = {vals.median():.2f}")
            ax.set_xlabel("Aperiodic Exponent (1/f slope)")
            ax.set_ylabel("Count")
            ax.set_title("Aperiodic Exponent Distribution")
            ax.legend()
            fig.tight_layout()
            p = out_dir / "aperiodic_exponent_hist.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))

    # 4. Age vs aperiodic exponent
    if "age" in feat_df.columns and "ap_exponent" in feat_df.columns:
        valid = feat_df.dropna(subset=["age", "ap_exponent"])
        if len(valid) >= 3:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(valid["age"], valid["ap_exponent"],
                       alpha=0.6, s=35, color="#dd8452", edgecolor="white")
            z = np.polyfit(valid["age"], valid["ap_exponent"], 1)
            x_line = np.linspace(valid["age"].min(), valid["age"].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#c44e52", ls="--",
                    label=f"trend: {z[0]:+.3f}/yr")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Aperiodic Exponent")
            ax.set_title("Age vs. Aperiodic Exponent")
            ax.legend()
            fig.tight_layout()
            p = out_dir / "age_vs_aperiodic.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(str(p))

    # 5. Global band power boxplots
    bp_cols = [c for c in feat_df.columns if c.startswith("global_bp_")]
    if bp_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [feat_df[c].dropna().values for c in bp_cols]
        labels = [c.replace("global_bp_", "") for c in bp_cols]
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("log10(V^2)")
        ax.set_title("Global Band Power Distributions")
        fig.tight_layout()
        p = out_dir / "global_bandpower_boxplot.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(str(p))

    return created
