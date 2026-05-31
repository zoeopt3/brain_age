#!/usr/bin/env python3
"""Stage 3 — Feature extraction from QC-passed EEG recordings.

Usage:
    python scripts/stage3_extract_features.py --config configs/stage3.yml

Reads manifest + QC summary, filters to included recordings, loads each
via MNE-BIDS, preprocesses, computes spectral features (band powers,
PAF, aperiodic), and writes a single-row-per-recording feature table.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipeline.config import load_yaml
from src.pipeline.io import write_parquet, write_markdown
from src.pipeline.logging_utils import get_logger, log_environment_info
from src.pipeline.preprocess import (
    load_raw_from_bids,
    apply_minimal_preprocessing,
    make_fixed_length_epochs,
    reject_bad_epochs,
)
from src.pipeline.features import compute_psd, build_feature_row
from src.pipeline.feature_dictionary import generate_feature_dictionary
from src.pipeline.plot_features import make_feature_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: feature extraction")
    parser.add_argument("--config", default="configs/stage3.yml")
    parser.add_argument("--project-config", default="configs/project.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_config)
    cfg = load_yaml(args.config)

    logger = get_logger("stage3",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 3: Feature Extraction ===")
    log_environment_info(logger)

    bids_root = Path(cfg["dataset"]["bids_root"])
    pp_cfg = cfg["preprocessing"]
    epoch_len = pp_cfg.get("epoch_length_sec", 2.0)

    # Ensure output dirs
    for d in [cfg["output"]["features_path"], cfg["output"]["feature_qc_path"],
              cfg["output"]["feature_dict_path"]]:
        Path(d).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["figures_dir"]).mkdir(parents=True, exist_ok=True)

    # Load manifest + QC
    manifest = pd.read_parquet(cfg["input"]["manifest_path"])
    logger.info("Manifest: %d recordings", len(manifest))

    qc_path = cfg["input"]["qc_summary_path"]
    if Path(qc_path).exists():
        qc = pd.read_parquet(qc_path)
        # Merge QC into manifest on shared keys
        merge_keys = [k for k in ["sub", "ses", "task", "run", "condition"]
                      if k in manifest.columns and k in qc.columns]
        if merge_keys:
            manifest = manifest.merge(qc[merge_keys + ["usable_duration_sec", "rejection_fraction"]],
                                      on=merge_keys, how="left", suffixes=("", "_qc"))
        logger.info("QC summary merged (%d QC rows)", len(qc))
    else:
        logger.warning("QC summary not found at %s — skipping QC filtering", qc_path)

    # Filter by QC thresholds
    incl = cfg.get("inclusion", {})
    min_dur = incl.get("min_usable_duration_sec", 60)
    max_rej = incl.get("max_rejection_fraction", 0.5)

    n_before = len(manifest)
    if "usable_duration_sec" in manifest.columns:
        manifest = manifest[manifest["usable_duration_sec"] >= min_dur]
    if "rejection_fraction" in manifest.columns:
        manifest = manifest[manifest["rejection_fraction"] <= max_rej]
    n_after = len(manifest)
    logger.info("Inclusion filter: %d -> %d recordings (dropped %d)",
                n_before, n_after, n_before - n_after)

    if manifest.empty:
        logger.error("No recordings pass QC thresholds. Adjust inclusion config.")
        sys.exit(1)

    # Process each recording
    feat_rows: list[dict] = []
    qc_rows: list[dict] = []
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    for idx, (_, row) in enumerate(manifest.iterrows(), 1):
        row_dict = row.to_dict()
        sub = row_dict.get("sub", f"row_{idx}")
        task = row_dict.get("task", "")
        ses = row_dict.get("ses", "")

        logger.info("[%d/%d] sub-%s ses=%s task=%s", idx, len(manifest), sub, ses, task)
        t0 = time.time()

        # Load
        raw = load_raw_from_bids(row_dict, bids_root)
        if raw is None:
            logger.warning("  SKIP: load failed")
            n_fail += 1
            continue

        import mne
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            logger.warning("  SKIP: no EEG channels")
            n_fail += 1
            continue

        # Preprocess
        try:
            raw = apply_minimal_preprocessing(raw, pp_cfg)
            epochs = make_fixed_length_epochs(raw, epoch_len)
            epochs, _ = reject_bad_epochs(epochs, pp_cfg)
        except Exception as e:
            logger.warning("  SKIP: preprocess error: %s", e)
            n_fail += 1
            continue

        if len(epochs) == 0:
            logger.warning("  SKIP: no epochs survived rejection")
            n_fail += 1
            continue

        # Compute PSD
        try:
            freqs, psd, ch_names = compute_psd(epochs, cfg.get("psd", {}))
        except Exception as e:
            logger.warning("  SKIP: PSD error: %s", e)
            n_fail += 1
            continue

        # Build feature row
        try:
            feat, fqc = build_feature_row(row_dict, freqs, psd, ch_names, cfg)
            feat_rows.append(feat)
            qc_rows.append(fqc)
            n_ok += 1

            sp_status = "OK" if fqc.get("specparam_ok") else fqc.get("specparam_reason", "?")[:30]
            logger.info("  OK: PAF=%.1f Hz, exponent=%.2f, specparam=%s (%.1fs)",
                        feat.get("paf_hz", float("nan")),
                        feat.get("ap_exponent", float("nan")),
                        sp_status, time.time() - t0)
        except Exception as e:
            logger.warning("  SKIP: feature error: %s", e)
            n_fail += 1

    elapsed = time.time() - t_start
    logger.info("Feature extraction: %d OK, %d failed, %.1fs total", n_ok, n_fail, elapsed)

    # Write outputs
    feat_df = pd.DataFrame(feat_rows)
    qc_df = pd.DataFrame(qc_rows)

    if not feat_df.empty:
        write_parquet(cfg["output"]["features_path"], feat_df)
        logger.info("Features written: %s (%d rows, %d columns)",
                    cfg["output"]["features_path"], len(feat_df), len(feat_df.columns))
    else:
        logger.warning("No feature rows — parquet not written")

    if not qc_df.empty:
        write_parquet(cfg["output"]["feature_qc_path"], qc_df)
        logger.info("Feature QC written: %s", cfg["output"]["feature_qc_path"])

    # Feature dictionary
    if not feat_df.empty:
        dict_md = generate_feature_dictionary(list(feat_df.columns), cfg)
        write_markdown(cfg["output"]["feature_dict_path"], dict_md)
        logger.info("Feature dictionary: %s", cfg["output"]["feature_dict_path"])

    # Plots
    if not feat_df.empty:
        seed = project_cfg.get("project", {}).get("seed", 1337)
        figs = make_feature_plots(feat_df, cfg["output"]["figures_dir"], seed)
        logger.info("Feature plots: %d figures in %s", len(figs), cfg["output"]["figures_dir"])

    logger.info("")
    logger.info("Stage 3 complete. Next:")
    logger.info("  Review %s", cfg["output"]["feature_dict_path"])
    logger.info("  Then proceed to Stage 4 (modeling)")


if __name__ == "__main__":
    main()
