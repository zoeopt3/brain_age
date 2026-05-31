"""Model evaluation plots: calibration, residuals, error distribution."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pred_vs_true(
    cv_pred_df: pd.DataFrame,
    out_path: str | Path,
    model_name: str = "",
) -> None:
    """Scatter plot of predicted vs. true age (calibration-style).

    Includes identity line and per-condition coloring if available.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = cv_pred_df[cv_pred_df["model"] == model_name] if model_name else cv_pred_df

    fig, ax = plt.subplots(figsize=(6, 6))

    conditions = df["condition"].unique() if "condition" in df.columns else ["all"]
    colors = {"EO": "#4c72b0", "EC": "#dd8452", "unknown": "#55a868", "all": "#4c72b0"}

    for cond in conditions:
        sub = df[df["condition"] == cond] if cond != "all" else df
        ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.5, s=30,
                   color=colors.get(cond, "#4c72b0"), label=cond, edgecolor="white")

    lims = [min(df["y_true"].min(), df["y_pred"].min()) - 1,
            max(df["y_true"].max(), df["y_pred"].max()) + 1]
    ax.plot(lims, lims, "k--", alpha=0.4, label="perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Age (years)")
    ax.set_ylabel("Predicted Age (years)")
    title = f"Predicted vs True Age"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_age(
    cv_pred_df: pd.DataFrame,
    out_path: str | Path,
    model_name: str = "",
) -> None:
    """Residual (pred - true) vs. true age scatter."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = cv_pred_df[cv_pred_df["model"] == model_name] if model_name else cv_pred_df

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["y_true"], df["residual"], alpha=0.5, s=30,
               color="#8172b2", edgecolor="white")
    ax.axhline(0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("True Age (years)")
    ax.set_ylabel("Residual (Predicted - True)")
    title = "Residuals vs Age"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(
    cv_pred_df: pd.DataFrame,
    out_path: str | Path,
    model_name: str = "",
) -> None:
    """Histogram of absolute prediction errors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = cv_pred_df[cv_pred_df["model"] == model_name] if model_name else cv_pred_df
    errors = np.abs(df["residual"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=20, color="#55a868", edgecolor="white", alpha=0.85)
    ax.axvline(errors.mean(), color="#c44e52", ls="--",
               label=f"MAE = {errors.mean():.2f} years")
    ax.set_xlabel("Absolute Error (years)")
    ax.set_ylabel("Count")
    title = "Prediction Error Distribution"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    cv_results: dict,
    out_path: str | Path,
) -> None:
    """Bar chart comparing MAE across models."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(cv_results.keys())
    maes = [cv_results[n]["mae_mean"] for n in names]
    stds = [cv_results[n]["mae_std"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    bars = ax.bar(x, maes, yerr=stds, capsize=5, color="#4c72b0", alpha=0.85,
                  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Model Comparison (CV Mean MAE ± Std)")
    for i, (m, s) in enumerate(zip(maes, stds)):
        ax.text(i, m + s + 0.05, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
