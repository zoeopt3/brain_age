#!/usr/bin/env python3
"""Stage 1a — Download a small subset of an OpenNeuro EEG-BIDS dataset.

Usage:
    python scripts/stage1_download_subset.py --config configs/stage1.yml

This script:
  1. Downloads BIDS metadata files (participants.tsv, dataset_description.json, etc.)
  2. Reads participants.tsv to pick N subjects (seeded random sample)
  3. Downloads only those subjects' folders via openneuro-py
  4. Writes outputs/dataset_fingerprint.json

Requires: openneuro-py (``pip install openneuro-py``)
If openneuro-py is not installed, the script prints installation instructions.
"""

import argparse
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_yaml, merge_dicts, hash_config
from src.pipeline.io import ensure_dirs, write_json, read_tsv
from src.pipeline.logging_utils import get_logger, log_environment_info


def _check_openneuro() -> bool:
    """Return True if the ``openneuro`` Python package is importable."""
    try:
        import openneuro
        return True
    except ImportError:
        return False


def _run_openneuro(args: list[str], logger) -> bool:
    """Run an openneuro-py command via ``python -m openneuro``, streaming output."""
    cmd = [sys.executable, "-m", "openneuro"] + args
    logger.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.stdout:
            for line in proc.stdout.strip().split("\n"):
                logger.info("  %s", line)
        if proc.returncode != 0:
            logger.error("openneuro-py failed (exit %d)", proc.returncode)
            if proc.stderr:
                for line in proc.stderr.strip().split("\n"):
                    logger.error("  %s", line)
            return False
        return True
    except FileNotFoundError:
        logger.error("python -m openneuro not found")
        return False
    except subprocess.TimeoutExpired:
        logger.error("openneuro-py timed out")
        return False


def download_metadata(dataset_id: str, target_dir: Path,
                      metadata_files: list[str], logger) -> bool:
    """Download only the top-level BIDS metadata files."""
    target_dir.mkdir(parents=True, exist_ok=True)

    include_args = []
    for mf in metadata_files:
        include_args.extend(["--include", mf])

    return _run_openneuro(
        ["download", "--dataset", dataset_id, "--target-dir", str(target_dir)]
        + include_args,
        logger,
    )


def pick_subjects(participants_tsv: Path, n: int, seed: int, logger) -> list[str]:
    """Read participants.tsv, deduplicate, and pick n subjects."""
    df = read_tsv(participants_tsv)

    if "participant_id" not in df.columns:
        logger.error("participants.tsv missing 'participant_id' column")
        logger.error("Columns found: %s", list(df.columns))
        sys.exit(1)

    # Deduplicate
    n_before = len(df)
    df = df.drop_duplicates(subset=["participant_id"])
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning("Removed %d duplicate participant_id entries", n_dupes)

    all_ids = sorted(df["participant_id"].tolist())
    logger.info("participants.tsv contains %d unique participants", len(all_ids))

    # Seeded random sample
    rng = random.Random(seed)
    n_pick = min(n, len(all_ids))
    selected = sorted(rng.sample(all_ids, n_pick))
    logger.info("Selected %d subjects (seed=%d): %s", n_pick, seed,
                ", ".join(selected[:5]) + ("..." if n_pick > 5 else ""))
    return selected


def download_subjects(dataset_id: str, target_dir: Path,
                      subject_ids: list[str], logger) -> int:
    """Download selected subjects' folders."""
    success_count = 0
    for i, sid in enumerate(subject_ids, 1):
        # Ensure sub- prefix for the include pattern
        sub_folder = sid if sid.startswith("sub-") else f"sub-{sid}"
        logger.info("[%d/%d] Downloading %s ...", i, len(subject_ids), sub_folder)
        ok = _run_openneuro(
            ["download", "--dataset", dataset_id,
             "--target-dir", str(target_dir),
             "--include", f"{sub_folder}/*"],
            logger,
        )
        if ok:
            success_count += 1
        else:
            logger.warning("Failed to download %s — skipping", sub_folder)
    return success_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: download OpenNeuro subset")
    parser.add_argument("--config", default="configs/stage1.yml")
    parser.add_argument("--project-config", default="configs/project.yml",
                        help="Base project config for paths (default: configs/project.yml)")
    args = parser.parse_args()

    # Load configs
    project_cfg = load_yaml(args.project_config)
    stage_cfg = load_yaml(args.config)

    logger = get_logger("stage1-download",
                        project_cfg.get("runtime", {}).get("log_level", "INFO"))
    logger.info("=== Stage 1a: Download Subset ===")
    log_environment_info(logger)

    # Check openneuro-py
    if not _check_openneuro():
        logger.error("")
        logger.error("openneuro-py is not installed.")
        logger.error("Install it with:")
        logger.error("  pip install openneuro-py")
        logger.error("")
        logger.error("Then re-run this script.")
        sys.exit(1)

    dataset_id = stage_cfg["dataset"]["dataset_id"]
    target_dir = Path(stage_cfg["dataset"]["target_dir"])
    n_subjects = stage_cfg["dataset"]["n_subjects"]
    seed = stage_cfg["dataset"]["seed"]
    metadata_files = stage_cfg["download"]["metadata_files"]

    # Step 1: Download metadata
    logger.info("--- Step 1: Download metadata files ---")
    ok = download_metadata(dataset_id, target_dir, metadata_files, logger)
    if not ok:
        logger.error("Metadata download failed. Check your internet connection.")
        sys.exit(1)

    # Step 2: Pick subjects
    logger.info("--- Step 2: Select subjects ---")
    participants_tsv = target_dir / "participants.tsv"
    if not participants_tsv.exists():
        logger.error("participants.tsv not found at %s", participants_tsv)
        logger.error("The metadata download may have failed or the dataset layout differs.")
        sys.exit(1)

    selected = pick_subjects(participants_tsv, n_subjects, seed, logger)

    # Step 3: Download subject folders
    logger.info("--- Step 3: Download subject data ---")
    n_ok = download_subjects(dataset_id, target_dir, selected, logger)
    logger.info("Downloaded %d / %d subjects successfully", n_ok, len(selected))

    # Step 4: Write fingerprint
    logger.info("--- Step 4: Write dataset fingerprint ---")
    fingerprint_path = stage_cfg["outputs"]["fingerprint"]
    fingerprint = {
        "stage": 1,
        "step": "download",
        "dataset_id": dataset_id,
        "download_method": "openneuro-py",
        "target_dir": str(target_dir),
        "n_subjects_requested": n_subjects,
        "n_subjects_downloaded": n_ok,
        "selected_subjects": selected,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": hash_config(stage_cfg),
    }
    write_json(fingerprint_path, fingerprint)
    logger.info("Fingerprint written to: %s", fingerprint_path)

    logger.info("")
    logger.info("Stage 1a complete. Next:")
    logger.info("  python scripts/stage1_build_manifest.py --config configs/stage1.yml")


if __name__ == "__main__":
    main()
