#!/usr/bin/env python3
"""Stage 2 — QC metrics and lightweight preprocessing.

Usage:
    python scripts/stage2_run_qc.py --config configs/stage2.yml

Reads outputs/manifest.parquet (from Stage 1), loads each EEG recording,
applies minimal preprocessing, computes QC metrics, and writes:
  - outputs/qc_summary.parquet
  - outputs/drop_log.jsonl
  - outputs/qc_report.md
  - outputs/qc_figures/*.png
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipeline.config import load_yaml, merge_dicts
from src.pipeline.io import write_parquet, write_markdown, write_json
from src.pipeline.logging_utils import get_logger, log_environment_info
from src.pipeline.preprocess import (
    load_raw_from_bids,
    apply_minimal_preprocessing,
    make_fixed_length_epochs,
    reject_bad_epochs,
    compute_usable_duration,
)
from src.pipeline.qc import (
    compute_qc_metrics,
    append_drop_log,
    DropReason,
)
from src.pipeline.plot_qc import make_qc_plots


def _generate_qc_report(
    qc_df: pd.DataFrame,
    n_attempted: int,
    n_success: int,
    drop_log_path: str,
) -> str:
    """Generate a Markdown QC report from the summary DataFrame."""
    lines = [
        "# Stage 2 — QC Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Processing Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Recordings attempted | {n_attempted} |",
        f"| Recordings succeeded | {n_success} |",
        f"| Recordings failed | {n_attempted - n_success} |",
        f"| Success rate | {n_success / n_attempted * 100:.1f}% |" if n_attempted > 0 else "",
        "",
    ]

    # Failure reasons
    drop_path = Path(drop_log_path)
    if drop_path.exists():
        import json
        reasons: dict[str, int] = {}
        with open(drop_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    r = entry.get("reason", "UNKNOWN")
                    reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            lines.append("## Failure Reasons")
            lines.append("")
            lines.append("| Reason | Count |")
            lines.append("|--------|------:|")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {count} |")
            lines.append("")

    if qc_df.empty:
        lines.append("**No recordings processed successfully.**")
        return "\n".join(lines)

    # Duration stats
    if "usable_duration_sec" in qc_df.columns:
        dur = qc_df["usable_duration_sec"].dropna()
        lines.append("## Usable Duration (seconds)")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|------|------:|")
        lines.append(f"| Mean | {dur.mean():.1f} |")
        lines.append(f"| Median | {dur.median():.1f} |")
        lines.append(f"| Min | {dur.min():.1f} |")
        lines.append(f"| Max | {dur.max():.1f} |")
        min_good = 60  # from config default
        n_short = int((dur < min_good).sum())
        lines.append(f"| Below {min_good}s threshold | {n_short} |")
        lines.append("")

    # Rejection fraction
    if "rejection_fraction" in qc_df.columns:
        rej = qc_df["rejection_fraction"].dropna()
        lines.append("## Rejection Fraction")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|------|------:|")
        lines.append(f"| Mean | {rej.mean():.3f} |")
        lines.append(f"| Median | {rej.median():.3f} |")
        lines.append(f"| Max | {rej.max():.3f} |")
        n_high = int((rej > 0.5).sum())
        lines.append(f"| >50% rejected | {n_high} |")
        lines.append("")

    # Line noise
    if "line_noise_ratio" in qc_df.columns:
        ln = qc_df["line_noise_ratio"].dropna()
        if len(ln) > 0:
            lines.append("## Line-Noise Ratio")
            lines.append("")
            lines.append("| Stat | Value |")
            lines.append("|------|------:|")
            lines.append(f"| Mean | {ln.mean():.2f} |")
            lines.append(f"| Median | {ln.median():.2f} |")
            lines.append(f"| Max | {ln.max():.2f} |")
            lines.append("")
            lines.append("*Ratio > 2 suggests residual line noise after notch filter.*")
            lines.append("")

    # Muscle proxy
    if "muscle_ratio" in qc_df.columns:
        mu = qc_df["muscle_ratio"].dropna()
        if len(mu) > 0:
            lines.append("## Muscle Artifact Proxy")
            lines.append("")
            lines.append("| Stat | Value |")
            lines.append("|------|------:|")
            lines.append(f"| Mean | {mu.mean():.3f} |")
            lines.append(f"| Median | {mu.median():.3f} |")
            lines.append(f"| Max | {mu.max():.3f} |")
            lines.append("")

    lines.append("## Recommended Review")
    lines.append("")
    lines.append("- Review recordings with rejection fraction > 0.50")
    lines.append("- Review recordings with usable duration < 60 seconds")
    lines.append("- Review recordings with line-noise ratio > 2.0")
    lines.append("- Review channel variance outliers > 2")
    lines.append("")
    lines.append("---")
    lines.append("*This report is auto-generated. Thresholds are observational — "
                 "do not auto-exclude without review.*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: QC + preprocessing")
    parser.add_argument("--config", default="configs/stage2.yml")
    parser.add_argument("--project-config", default="configs/project.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_config)
    stage_cfg = load_yaml(args.config)

    logger = get_logger("stage2",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 2: QC + Preprocessing ===")
    log_environment_info(logger)

    # Paths
    bids_root = Path(stage_cfg["dataset"]["bids_root"])
    manifest_path = stage_cfg["input"]["manifest_path"]
    pp_cfg = stage_cfg["preprocessing"]
    qc_cfg = stage_cfg["qc"]
    out_cfg = stage_cfg["output"]

    drop_log_path = out_cfg["drop_log_path"]
    # Clear previous drop log
    Path(drop_log_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(drop_log_path).exists():
        Path(drop_log_path).unlink()

    # Load manifest
    if not Path(manifest_path).exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.error("Run Stage 1 first.")
        sys.exit(1)

    manifest = pd.read_parquet(manifest_path)
    logger.info("Manifest loaded: %d recordings", len(manifest))

    if manifest.empty:
        logger.error("Manifest is empty. Nothing to process.")
        sys.exit(1)

    # Process each recording
    qc_rows: list[dict] = []
    n_attempted = 0
    n_success = 0
    epoch_len = pp_cfg.get("epoch_length_sec", 2.0)
    min_good = pp_cfg.get("reject", {}).get("min_good_duration_sec", 60)

    t_start = time.time()

    for idx, row in manifest.iterrows():
        row_dict = row.to_dict()
        sub = row_dict.get("sub", f"row_{idx}")
        n_attempted += 1

        logger.info("[%d/%d] Processing sub-%s ...", n_attempted, len(manifest), sub)
        t0 = time.time()

        # 1. Load
        raw = load_raw_from_bids(row_dict, bids_root)
        if raw is None:
            logger.warning("  SKIP: could not load EEG for sub-%s", sub)
            append_drop_log(drop_log_path, sub, DropReason.LOAD_ERROR,
                            f"Failed to load from {row_dict.get('rel_path', '?')}")
            continue

        # Check EEG channels
        import mne
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            logger.warning("  SKIP: no EEG channels for sub-%s", sub)
            append_drop_log(drop_log_path, sub, DropReason.NO_EEG_CHANNELS)
            continue

        # Check minimum raw duration
        if raw.times[-1] < epoch_len * 2:
            logger.warning("  SKIP: recording too short (%.1fs) for sub-%s",
                           raw.times[-1], sub)
            append_drop_log(drop_log_path, sub, DropReason.TOO_SHORT,
                            f"Duration {raw.times[-1]:.1f}s < {epoch_len * 2}s")
            continue

        # 2. Preprocess
        try:
            raw = apply_minimal_preprocessing(raw, pp_cfg)
        except Exception as e:
            logger.warning("  SKIP: preprocessing error for sub-%s: %s", sub, e)
            append_drop_log(drop_log_path, sub, DropReason.PREPROCESS_ERROR, str(e))
            continue

        # 3. Epoch + reject
        try:
            epochs = make_fixed_length_epochs(raw, epoch_len)
            epochs, rejection_info = reject_bad_epochs(epochs, pp_cfg)
        except Exception as e:
            logger.warning("  SKIP: epoching/rejection error for sub-%s: %s", sub, e)
            append_drop_log(drop_log_path, sub, DropReason.PREPROCESS_ERROR, str(e))
            continue

        usable_sec = compute_usable_duration(epochs, epoch_len)
        if usable_sec < min_good:
            logger.warning("  FLAG: sub-%s has only %.0fs usable (threshold %ds)",
                           sub, usable_sec, min_good)
            # Still compute metrics but flag
            append_drop_log(drop_log_path, sub, DropReason.EXCESSIVE_ARTIFACT,
                            f"Usable {usable_sec:.0f}s < {min_good}s")

        # 4. Compute QC metrics
        try:
            # Merge preprocessing and qc config sections for the metric function
            merged_qc_cfg = {**pp_cfg, **qc_cfg, "epoch_length_sec": epoch_len}
            metrics = compute_qc_metrics(raw, epochs, rejection_info,
                                         row_dict, merged_qc_cfg)
            # Carry over age if present
            if "age" in row_dict and row_dict["age"] is not None:
                metrics["age"] = row_dict["age"]

            qc_rows.append(metrics)
            n_success += 1
            logger.info("  OK: %.0fs usable, %.0f%% rejected, %.1fs elapsed",
                        usable_sec,
                        rejection_info.get("rejection_fraction", 0) * 100,
                        time.time() - t0)
        except Exception as e:
            logger.warning("  SKIP: QC error for sub-%s: %s", sub, e)
            append_drop_log(drop_log_path, sub, DropReason.QC_ERROR, str(e))
            continue

        # 5. Optionally save cleaned data
        if out_cfg.get("save_cleaned", {}).get("enabled", False):
            clean_dir = Path(out_cfg["save_cleaned"]["dir"])
            clean_dir.mkdir(parents=True, exist_ok=True)
            fmt = out_cfg["save_cleaned"].get("format", "fif")
            fname = clean_dir / f"sub-{sub}_cleaned.{fmt}"
            try:
                if fmt == "fif":
                    epochs.save(str(fname), overwrite=True, verbose=False)
                logger.info("  Saved cleaned epochs: %s", fname)
            except Exception as e:
                logger.warning("  Could not save cleaned data: %s", e)

    elapsed = time.time() - t_start
    logger.info("Processing complete: %d/%d succeeded in %.1fs",
                n_success, n_attempted, elapsed)

    # Write outputs
    qc_df = pd.DataFrame(qc_rows)

    # QC summary parquet
    qc_path = out_cfg["qc_summary_path"]
    if not qc_df.empty:
        write_parquet(qc_path, qc_df)
        logger.info("QC summary written: %s (%d rows)", qc_path, len(qc_df))
    else:
        logger.warning("No QC rows — parquet not written")

    # QC report
    report_text = _generate_qc_report(qc_df, n_attempted, n_success, drop_log_path)
    report_path = out_cfg["qc_report_path"]
    write_markdown(report_path, report_text)
    logger.info("QC report written: %s", report_path)

    # QC figures
    figures_dir = out_cfg["figures_dir"]
    if not qc_df.empty:
        figs = make_qc_plots(qc_df, figures_dir,
                             seed=project_cfg.get("project", {}).get("seed", 1337))
        logger.info("QC figures written: %d plots in %s", len(figs), figures_dir)

    logger.info("")
    logger.info("Stage 2 complete. Next:")
    logger.info("  Review %s", report_path)
    logger.info("  Then proceed to Stage 3 (feature extraction)")


if __name__ == "__main__":
    main()
