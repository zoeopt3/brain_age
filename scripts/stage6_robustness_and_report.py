#!/usr/bin/env python3
"""Stage 6 — Robustness checks, final report, and release bundle.

Usage:
    python scripts/stage6_robustness_and_report.py --config configs/stage6.yml
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipeline.config import load_yaml
from src.pipeline.io import write_json, write_markdown
from src.pipeline.logging_utils import get_logger, log_environment_info
from src.pipeline.modeling import select_feature_columns
from src.pipeline.robustness import (
    load_best_model_type,
    run_eo_vs_ec,
    run_feature_family_sensitivity,
)
from src.pipeline.plot_robustness import plot_eo_vs_ec, plot_feature_family
from src.pipeline.reporting import render_final_report
from src.pipeline.release import make_release_manifest, build_release_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: robustness + report + release")
    parser.add_argument("--config", default="configs/stage6.yml")
    parser.add_argument("--project-config", default="configs/project.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_config)
    cfg = load_yaml(args.config)

    logger = get_logger("stage6",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 6: Robustness + Report + Release ===")
    log_environment_info(logger)

    # Dirs
    rob_dir = Path("outputs/robustness")
    fig_dir = Path("outputs/figures/robustness")
    rob_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    feat_path = cfg["input"]["features_path"]
    df = pd.read_parquet(feat_path)
    target_col = cfg["target"]["column"]
    group_col = cfg["groups"]["column"]
    df = df.dropna(subset=[target_col])
    logger.info("Features: %d rows, %d participants", len(df), df[group_col].nunique())

    feature_cols = select_feature_columns(df, cfg.get("preprocessing", {}))
    preprocess_cfg = cfg.get("preprocessing", {})
    cv_cfg = cfg.get("robustness", {}).get("cv", {})

    # Best model
    model_name = cfg.get("robustness", {}).get("model_name_override")
    if not model_name:
        model_name = load_best_model_type(cfg["input"]["model_results_path"])
    logger.info("Model for robustness: %s", model_name)

    # ── Robustness checks ──────────────────────────────────────────────
    enabled = cfg.get("robustness", {}).get("enabled_checks", [])
    robustness_results: dict = {}

    if "eo_vs_ec" in enabled:
        logger.info("--- Robustness: EO vs EC ---")
        eoec = run_eo_vs_ec(df, feature_cols, model_name,
                            preprocess_cfg, cv_cfg, target_col, group_col, logger)
        robustness_results["eo_vs_ec"] = eoec
        plot_eo_vs_ec(eoec, fig_dir / "eo_vs_ec_performance.png")

    if "feature_family_sensitivity" in enabled:
        logger.info("--- Robustness: Feature Family Sensitivity ---")
        fam_cfg = cfg.get("robustness", {}).get("feature_families", {})
        fam = run_feature_family_sensitivity(
            df, feature_cols, model_name, fam_cfg,
            preprocess_cfg, cv_cfg, target_col, group_col, logger,
        )
        # Remove large feature lists from JSON for readability
        fam_clean = {}
        for k, v in fam.items():
            if isinstance(v, dict):
                fam_clean[k] = {kk: vv for kk, vv in v.items() if kk != "features"}
            else:
                fam_clean[k] = v
        robustness_results["feature_family_sensitivity"] = fam_clean
        plot_feature_family(fam, fig_dir / "feature_family_sensitivity.png")

    # Combined robustness plot
    if robustness_results:
        write_json(rob_dir / "robustness_results.json", robustness_results)
        logger.info("Robustness results: %s", rob_dir / "robustness_results.json")

        # Summary comparison plot
        _plot_combined_robustness(robustness_results, fig_dir / "robustness_mae_comparison.png")

    # ── Final report ───────────────────────────────────────────────────
    logger.info("--- Generating final report ---")

    # Load supporting data
    model_results = {}
    mr_path = cfg["input"]["model_results_path"]
    if Path(mr_path).exists():
        with open(mr_path) as f:
            model_results = json.load(f)

    fingerprint = {}
    fp_path = cfg["input"].get("dataset_fingerprint_path", "")
    if fp_path and Path(fp_path).exists():
        with open(fp_path) as f:
            fingerprint = json.load(f)

    importance_df = pd.DataFrame()
    imp_path = Path(cfg["input"]["explainability_dir"]) / "global_importance.parquet"
    if imp_path.exists():
        importance_df = pd.read_parquet(imp_path)

    report_md = render_final_report(cfg, {
        "features_df": df,
        "model_results": model_results,
        "robustness_results": robustness_results,
        "importance_df": importance_df,
        "dataset_fingerprint": fingerprint,
    })
    report_path = cfg["report"]["output_path"]
    write_markdown(report_path, report_md)
    logger.info("Final report: %s", report_path)

    # ── Release bundle ─────────────────────────────────────────────────
    logger.info("--- Building release bundle ---")
    manifest = make_release_manifest(cfg)
    rel_dir = Path(cfg["release"]["out_dir"])
    rel_dir.mkdir(parents=True, exist_ok=True)
    write_json(rel_dir / "release_manifest.json", manifest)

    zip_path = build_release_zip(cfg, manifest)
    logger.info("Release bundle: %s (%d files, %.1f KB)",
                zip_path, manifest["n_files"],
                manifest["total_size_bytes"] / 1024)

    logger.info("")
    logger.info("Stage 6 complete.")
    logger.info("  Final report: %s", report_path)
    logger.info("  Release bundle: %s", zip_path)
    logger.info("")
    logger.info("All stages finished. The pipeline is complete.")


def _plot_combined_robustness(results: dict, out_path) -> None:
    """Quick combined MAE comparison across all robustness checks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels, maes, stds = [], [], []

    eoec = results.get("eo_vs_ec", {})
    for cond in ["EO", "EC", "combined"]:
        if cond in eoec and "mae_mean" in eoec[cond]:
            labels.append(f"EO/EC: {cond}")
            maes.append(eoec[cond]["mae_mean"])
            stds.append(eoec[cond]["mae_std"])

    fam = results.get("feature_family_sensitivity", {})
    for k, v in fam.items():
        if isinstance(v, dict) and "mae_mean" in v:
            labels.append(k.replace("_", " ").title())
            maes.append(v["mae_mean"])
            stds.append(v["mae_std"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 5))
    x = range(len(labels))
    ax.bar(x, maes, yerr=stds, capsize=4, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE (years)")
    ax.set_title("Robustness: MAE Across All Checks")
    for i, (m, s) in enumerate(zip(maes, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
