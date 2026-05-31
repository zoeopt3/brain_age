#!/usr/bin/env python3
"""Stage 5 — Explainability: permutation importance, SHAP, ablations.

Usage:
    python scripts/stage5_explain_and_ablations.py --config configs/stage5.yml

Computes feature importance on held-out CV folds, runs ablation
experiments (remove PAF, remove aperiodic, etc.), optionally computes
SHAP values, and generates interpretive notes.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipeline.config import load_yaml
from src.pipeline.io import write_json, write_parquet, write_markdown
from src.pipeline.logging_utils import get_logger, log_environment_info
from src.pipeline.modeling import select_feature_columns
from src.pipeline.explainability import (
    load_best_model_name,
    build_feature_groups,
    compute_foldwise_permutation_importance,
    compute_shap_if_available,
    run_ablation_experiments,
    generate_explainability_notes,
)
from src.pipeline.plot_explain import (
    plot_top_importances,
    plot_ablation_mae,
    plot_shap_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: explainability + ablations")
    parser.add_argument("--config", default="configs/stage5.yml")
    parser.add_argument("--project-config", default="configs/project.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_config)
    cfg = load_yaml(args.config)

    logger = get_logger("stage5",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 5: Explainability + Ablations ===")
    log_environment_info(logger)

    # Output dirs
    out_dir = Path(cfg["output"]["out_dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    feat_path = cfg["input"]["features_path"]
    if not Path(feat_path).exists():
        logger.error("Features not found: %s — run Stage 3 first.", feat_path)
        sys.exit(1)

    df = pd.read_parquet(feat_path)
    target_col = cfg["target"]["column"]
    group_col = cfg["groups"]["column"]

    df = df.dropna(subset=[target_col])
    logger.info("Features: %d rows, %d participants", len(df), df[group_col].nunique())

    # Select feature columns
    feature_cols = select_feature_columns(df, cfg.get("preprocessing", {}))
    logger.info("Feature columns: %d", len(feature_cols))

    # Best model
    model_name = load_best_model_name(
        cfg["input"]["model_results_path"],
        cfg.get("model_selection", {}),
    )
    logger.info("Best model: %s", model_name)

    # Feature groups
    fg = build_feature_groups(feature_cols, cfg.get("feature_groups", {}))
    for gname, gcols in fg.items():
        logger.info("  Feature group '%s': %d columns%s",
                    gname, len(gcols),
                    f" ({', '.join(gcols[:3])}{'...' if len(gcols) > 3 else ''})" if gcols else "")

    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values
    preprocess_cfg = cfg.get("preprocessing", {})
    cv_cfg = cfg.get("cv", {})

    # ── Permutation importance ─────────────────────────────────────────
    perm_cfg = cfg.get("explainability", {}).get("permutation", {})
    if perm_cfg.get("enabled", True):
        logger.info("--- Permutation Importance (foldwise) ---")
        imp_df = compute_foldwise_permutation_importance(
            model_name, X, y, groups, feature_cols,
            preprocess_cfg, cv_cfg,
        )
        write_parquet(out_dir / "global_importance.parquet", imp_df)
        logger.info("Importance written: %s", out_dir / "global_importance.parquet")

        # Top features markdown
        top_md = "# Top Features\n\n"
        top_md += "| Rank | Feature | Importance |\n|------|---------|------------|\n"
        for i, row in imp_df.head(20).iterrows():
            top_md += f"| {i+1} | `{row['feature']}` | {row['importance_mean']:.4f} ± {row['importance_std']:.4f} |\n"
        write_markdown(out_dir / "top_features.md", top_md)

        # Plot
        plot_top_importances(imp_df, fig_dir / "importance_bar_top20.png")
        logger.info("Importance plot saved")
    else:
        imp_df = pd.DataFrame()
        logger.info("Permutation importance disabled")

    # ── SHAP (optional) ───────────────────────────────────────────────
    shap_cfg = cfg.get("explainability", {}).get("shap", {})
    shap_df = None
    if shap_cfg.get("enabled", True):
        logger.info("--- SHAP Values (optional) ---")
        shap_df = compute_shap_if_available(
            model_name, X, y, feature_cols,
            preprocess_cfg, shap_cfg,
        )
        if shap_df is not None:
            write_parquet(out_dir / "shap_values.parquet", shap_df)
            plot_shap_summary(shap_df, fig_dir / "shap_summary.png")
            logger.info("SHAP values computed and saved (%d rows)", len(shap_df))
        else:
            logger.info("SHAP skipped (package not installed or model not supported)")

    # ── Ablation experiments ───────────────────────────────────────────
    abl_cfg = cfg.get("ablations", {})
    ablation_results = {}
    if abl_cfg.get("enabled", True):
        logger.info("--- Ablation Experiments ---")
        experiments = abl_cfg.get("experiments", [])
        ablation_results = run_ablation_experiments(
            model_name, df, feature_cols, target_col, group_col,
            fg, experiments, preprocess_cfg, cv_cfg, logger,
        )
        write_json(out_dir / "ablation_results.json", ablation_results)
        plot_ablation_mae(ablation_results, fig_dir / "ablation_mae_comparison.png")
        logger.info("Ablation results saved")

    # ── Narrative notes ────────────────────────────────────────────────
    if not imp_df.empty:
        notes = generate_explainability_notes(imp_df, ablation_results, fg, model_name)
        write_markdown(cfg["output"]["notes_path"], notes)
        logger.info("Explainability notes: %s", cfg["output"]["notes_path"])

    logger.info("")
    logger.info("Stage 5 complete. Next:")
    logger.info("  Review %s", cfg["output"]["notes_path"])
    logger.info("  Then proceed to Stage 6 (final report)")


if __name__ == "__main__":
    main()
