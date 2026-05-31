#!/usr/bin/env python3
"""Stage 4 — Train and evaluate age-prediction models.

Usage:
    python scripts/stage4_train_models.py --config configs/stage4.yml

Loads outputs/features.parquet, runs leakage-safe GroupKFold CV
for each enabled model, writes predictions + metrics + plots.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipeline.config import load_yaml
from src.pipeline.io import write_json, write_parquet
from src.pipeline.logging_utils import get_logger, log_environment_info
from src.pipeline.modeling import (
    select_feature_columns,
    run_group_cv,
    fit_final_model,
    build_model_results_json,
    save_model_artifacts,
)
from src.pipeline.plot_model import (
    plot_pred_vs_true,
    plot_residuals_vs_age,
    plot_error_histogram,
    plot_model_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: train age-prediction models")
    parser.add_argument("--config", default="configs/stage4.yml")
    parser.add_argument("--project-config", default="configs/project.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_config)
    cfg = load_yaml(args.config)

    logger = get_logger("stage4",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 4: Modeling ===")
    log_environment_info(logger)

    # Ensure output dirs
    for d in [cfg["output"]["figures_dir"], cfg["output"]["models_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────
    feat_path = cfg["input"]["features_path"]
    if not Path(feat_path).exists():
        logger.error("Features file not found: %s", feat_path)
        logger.error("Run Stage 3 first.")
        sys.exit(1)

    df = pd.read_parquet(feat_path)
    logger.info("Features loaded: %d rows, %d columns", len(df), len(df.columns))

    # ── Filter ─────────────────────────────────────────────────────────
    target_col = cfg["target"]["column"]
    group_col = cfg["groups"]["column"]
    filt = cfg.get("filtering", {})

    if filt.get("drop_na_target", True):
        n_before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = n_before - len(df)
        if dropped:
            logger.info("Dropped %d rows with missing %s", dropped, target_col)

    exclude_cond = filt.get("exclude_conditions", [])
    if exclude_cond and "condition" in df.columns:
        df = df[~df["condition"].isin(exclude_cond)]
        logger.info("After condition filter: %d rows", len(df))

    min_rows = filt.get("min_rows", 30)
    if len(df) < min_rows:
        logger.error("Too few rows (%d < %d). Cannot train models.", len(df), min_rows)
        sys.exit(1)

    n_groups = df[group_col].nunique()
    logger.info("Dataset: %d rows, %d participants, age range %.0f-%.0f",
                len(df), n_groups, df[target_col].min(), df[target_col].max())

    # ── Select features ────────────────────────────────────────────────
    feature_cols = select_feature_columns(df, cfg.get("preprocessing", {}))
    logger.info("Feature columns: %d", len(feature_cols))
    for c in feature_cols:
        n_miss = df[c].isna().sum()
        if n_miss > 0:
            logger.info("  %s: %d missing", c, n_miss)

    # ── Cross-validation ───────────────────────────────────────────────
    logger.info("--- Cross-validation (GroupKFold) ---")
    cv_results, cv_pred_df = run_group_cv(
        df, feature_cols, target_col, group_col,
        preprocess_cfg=cfg.get("preprocessing", {}),
        models_cfg=cfg.get("models", {}),
        cv_cfg=cfg.get("cv", {}),
        logger=logger,
    )

    # Summary
    logger.info("")
    logger.info("--- Results Summary ---")
    best_model = None
    best_mae = float("inf")
    for model_name, res in cv_results.items():
        mae = res["mae_mean"]
        r2 = res["r2_mean"]
        logger.info("  %-15s  MAE=%.2f ± %.2f   R²=%.3f ± %.3f",
                    model_name, mae, res["mae_std"], r2, res["r2_std"])
        if mae < best_mae:
            best_mae = mae
            best_model = model_name

    logger.info("Best model by MAE: %s (%.2f years)", best_model, best_mae)

    # ── Write outputs ──────────────────────────────────────────────────
    # CV predictions
    write_parquet(cfg["output"]["cv_predictions_path"], cv_pred_df)
    logger.info("CV predictions: %s (%d rows)", cfg["output"]["cv_predictions_path"], len(cv_pred_df))

    # Model results JSON
    results_json = build_model_results_json(cv_results, cfg, df, feature_cols, best_model)
    write_json(cfg["output"]["model_results_path"], results_json)
    logger.info("Model results: %s", cfg["output"]["model_results_path"])

    # ── Plots ──────────────────────────────────────────────────────────
    figs_dir = Path(cfg["output"]["figures_dir"])

    if best_model and not cv_pred_df.empty:
        plot_pred_vs_true(cv_pred_df, figs_dir / "calibration_pred_vs_true.png", best_model)
        plot_residuals_vs_age(cv_pred_df, figs_dir / "residuals_vs_age.png", best_model)
        plot_error_histogram(cv_pred_df, figs_dir / "error_histogram.png", best_model)
        plot_model_comparison(cv_results, figs_dir / "model_comparison.png")
        logger.info("Plots written to %s", figs_dir)

    # ── Final model ────────────────────────────────────────────────────
    if cfg["output"].get("save_models", False) and best_model and best_model != "baseline_mean":
        logger.info("Fitting final %s on all data ...", best_model)
        pipeline, _ = fit_final_model(
            df, feature_cols, target_col, best_model,
            cfg.get("preprocessing", {}), cfg.get("models", {}),
        )
        saved_path = save_model_artifacts(pipeline, cfg["output"]["models_dir"], best_model)
        logger.info("Saved model: %s", saved_path)

    logger.info("")
    logger.info("Stage 4 complete. Next:")
    logger.info("  Review %s", cfg["output"]["model_results_path"])
    logger.info("  Then proceed to Stage 5 (explainability)")


if __name__ == "__main__":
    main()
