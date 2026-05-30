#!/usr/bin/env python3
"""Stage 1b — Build cohort manifest and BIDS sanity report.

Usage:
    python scripts/stage1_build_manifest.py --config configs/stage1.yml

This script:
  1. Scans the BIDS root for EEG recording files
  2. Joins with participants.tsv for age and other demographics
  3. Writes outputs/manifest.parquet
  4. Writes outputs/bids_sanity_report.md with counts and missingness
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.pipeline.config import load_yaml
from src.pipeline.io import read_tsv, write_parquet, write_json, write_markdown
from src.pipeline.bids_scan import find_eeg_files, list_subject_dirs
from src.pipeline.logging_utils import get_logger


def build_manifest(bids_root: Path, logger) -> pd.DataFrame:
    """Scan BIDS root and join with participants.tsv to build manifest."""

    # --- Find EEG files ---
    eeg_records = find_eeg_files(bids_root)
    logger.info("Found %d EEG files under %s", len(eeg_records), bids_root)

    if not eeg_records:
        logger.warning("No EEG files found. The download may have failed or "
                        "the BIDS layout differs from expectations.")
        return pd.DataFrame()

    eeg_df = pd.DataFrame(eeg_records)

    # --- Read participants.tsv ---
    participants_tsv = bids_root / "participants.tsv"
    if not participants_tsv.exists():
        logger.warning("participants.tsv not found — manifest will lack demographics")
        return eeg_df

    demo = read_tsv(participants_tsv)

    # Normalise participant_id to bare ID (remove "sub-" prefix if present)
    if "participant_id" in demo.columns:
        demo["participant_id"] = (
            demo["participant_id"]
            .astype(str)
            .str.removeprefix("sub-")
        )
        demo = demo.drop_duplicates(subset=["participant_id"])
    else:
        logger.warning("participants.tsv has no 'participant_id' column")

    # --- Join ---
    manifest = eeg_df.merge(
        demo,
        left_on="sub",
        right_on="participant_id",
        how="left",
    )

    # Try to find age column (datasets use various names)
    age_col = None
    for candidate in ["age", "Age", "age_years", "Age_Years", "age_at_scan"]:
        if candidate in manifest.columns:
            age_col = candidate
            break
    if age_col and age_col != "age":
        manifest = manifest.rename(columns={age_col: "age"})
    if "age" in manifest.columns:
        manifest["age"] = pd.to_numeric(manifest["age"], errors="coerce")

    return manifest


def generate_sanity_report(manifest: pd.DataFrame, bids_root: Path,
                           logger) -> str:
    """Generate a Markdown sanity report from the manifest."""

    lines = [
        "# BIDS Sanity Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**BIDS root:** `{bids_root}`",
        "",
    ]

    # Participants.tsv vs folders
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists():
        demo = read_tsv(participants_tsv)
        if "participant_id" in demo.columns:
            n_tsv_raw = len(demo)
            n_tsv_unique = demo["participant_id"].nunique()
            n_dupes = n_tsv_raw - n_tsv_unique
        else:
            n_tsv_raw = n_tsv_unique = 0
            n_dupes = 0
    else:
        n_tsv_raw = n_tsv_unique = n_dupes = 0

    subject_dirs = list_subject_dirs(bids_root)
    n_dirs = len(subject_dirs)

    lines.append("## Participant Counts")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Rows in participants.tsv | {n_tsv_raw} |")
    lines.append(f"| Unique participant_id | {n_tsv_unique} |")
    lines.append(f"| Duplicate participant_id | {n_dupes} |")
    lines.append(f"| Subject folders (sub-*) | {n_dirs} |")
    lines.append(f"| TSV vs folders mismatch | {abs(n_tsv_unique - n_dirs)} |")
    lines.append("")

    if manifest.empty:
        lines.append("**No EEG files found — remaining checks skipped.**")
        return "\n".join(lines)

    # EEG file counts
    n_files = len(manifest)
    n_subs_with_eeg = manifest["sub"].nunique()

    lines.append("## EEG File Counts")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Total EEG files | {n_files} |")
    lines.append(f"| Subjects with EEG | {n_subs_with_eeg} |")
    lines.append(f"| Extensions found | {', '.join(sorted(manifest['extension'].unique()))} |")
    lines.append("")

    # Condition breakdown
    if "condition" in manifest.columns:
        cond_counts = manifest["condition"].value_counts()
        lines.append("## Condition Breakdown (EO / EC)")
        lines.append("")
        lines.append("| Condition | Files |")
        lines.append("|-----------|------:|")
        for cond, count in cond_counts.items():
            lines.append(f"| {cond} | {count} |")
        lines.append("")

    # Age missingness
    if "age" in manifest.columns:
        n_age_present = manifest["age"].notna().sum()
        n_age_missing = manifest["age"].isna().sum()
        pct_missing = (n_age_missing / n_files * 100) if n_files > 0 else 0
        lines.append("## Age Data")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|------:|")
        lines.append(f"| Age present | {n_age_present} |")
        lines.append(f"| Age missing | {n_age_missing} ({pct_missing:.1f}%) |")
        if n_age_present > 0:
            lines.append(f"| Age min | {manifest['age'].min():.1f} |")
            lines.append(f"| Age max | {manifest['age'].max():.1f} |")
            lines.append(f"| Age mean | {manifest['age'].mean():.1f} |")
        lines.append("")
    else:
        lines.append("## Age Data")
        lines.append("")
        lines.append("**Age column not found in participants.tsv.**")
        lines.append("")

    # Subjects in folders but not in TSV (or vice versa)
    if participants_tsv.exists() and "participant_id" in demo.columns:
        tsv_ids = set(demo["participant_id"].astype(str).str.removeprefix("sub-"))
        dir_ids = set(subject_dirs)
        in_tsv_not_dirs = sorted(tsv_ids - dir_ids)
        in_dirs_not_tsv = sorted(dir_ids - tsv_ids)
        if in_tsv_not_dirs or in_dirs_not_tsv:
            lines.append("## TSV vs Folder Mismatches")
            lines.append("")
            if in_tsv_not_dirs:
                lines.append(f"- **In TSV but no folder:** {len(in_tsv_not_dirs)} "
                             f"(e.g. {', '.join(in_tsv_not_dirs[:5])})")
            if in_dirs_not_tsv:
                lines.append(f"- **Folder but not in TSV:** {len(in_dirs_not_tsv)} "
                             f"(e.g. {', '.join(in_dirs_not_tsv[:5])})")
            lines.append("")

    lines.append("---")
    lines.append("*This report is auto-generated. Review before proceeding to Stage 2.*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1b: build manifest + sanity report")
    parser.add_argument("--config", default="configs/stage1.yml")
    parser.add_argument("--project-config", default="configs/project.yml")
    args = parser.parse_args()

    project_cfg = load_yaml(args.project_config)
    stage_cfg = load_yaml(args.config)

    logger = get_logger("stage1-manifest",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 1b: Build Manifest ===")

    bids_root = Path(stage_cfg["dataset"]["target_dir"])
    if not bids_root.exists():
        logger.error("BIDS root not found: %s", bids_root)
        logger.error("Run stage1_download_subset.py first.")
        sys.exit(1)

    # Build manifest
    manifest = build_manifest(bids_root, logger)
    manifest_path = stage_cfg["outputs"]["manifest"]
    if not manifest.empty:
        write_parquet(manifest_path, manifest)
        logger.info("Manifest written: %s (%d rows)", manifest_path, len(manifest))
    else:
        logger.warning("Manifest is empty — no Parquet written")

    # Sanity report
    report_text = generate_sanity_report(manifest, bids_root, logger)
    report_path = stage_cfg["outputs"]["sanity_report"]
    write_markdown(report_path, report_text)
    logger.info("Sanity report written: %s", report_path)

    logger.info("")
    logger.info("Stage 1b complete. Next:")
    logger.info("  Review %s", report_path)
    logger.info("  Then proceed to Stage 2 (QC + preprocessing)")


if __name__ == "__main__":
    main()
