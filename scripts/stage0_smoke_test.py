#!/usr/bin/env python3
"""Stage 0 smoke test — verify project skeleton, config, and paths.

Usage:
    python scripts/stage0_smoke_test.py --config configs/project.yml

This script does NOT download any data.  It validates that the repo
structure, configuration, and core utilities are working correctly.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on sys.path so ``src`` is importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_yaml, validate_config_basic, hash_config
from src.pipeline.io import ensure_dirs, write_json
from src.pipeline.logging_utils import get_logger, log_environment_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: project skeleton smoke test")
    parser.add_argument(
        "--config",
        default="configs/project.yml",
        help="Path to the project YAML config (default: configs/project.yml)",
    )
    args = parser.parse_args()

    # ── 1. Load and validate config ────────────────────────────────────
    cfg = load_yaml(args.config)
    log_level = cfg.get("runtime", {}).get("log_level", "INFO")
    logger = get_logger("stage0", log_level)

    logger.info("=== Stage 0: Smoke Test ===")
    log_environment_info(logger)

    logger.info("Config loaded from: %s", args.config)
    errors = validate_config_basic(cfg)
    if errors:
        for err in errors:
            logger.error(err)
        logger.error("Config validation FAILED — fix the issues above.")
        sys.exit(1)
    logger.info("Config validation passed.")

    # ── 2. Create required directories ─────────────────────────────────
    created = ensure_dirs(cfg)
    logger.info("Ensured %d directories:", len(created))
    for d in created:
        logger.info("  %s", d)

    # ── 3. Write stage-0 manifest ──────────────────────────────────────
    config_hash = hash_config(cfg)
    manifest = {
        "stage": 0,
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_file": args.config,
        "config_hash": config_hash,
        "created_paths": created,
        "project_name": cfg["project"]["name"],
        "seed": cfg["project"]["seed"],
    }

    out_path = Path(cfg["paths"]["outputs"]) / "stage0_ok.json"
    write_json(out_path, manifest)
    logger.info("Manifest written to: %s", out_path)

    # ── 4. Done ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Stage 0 complete.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Place the Brain_research PDFs in the repo root (if not already present).")
    logger.info("  2. Run Stage 1 to download a subset of ds004186:")
    logger.info("     python scripts/stage1_download.py --config configs/project.yml")
    logger.info("")


if __name__ == "__main__":
    main()
