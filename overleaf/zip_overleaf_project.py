#!/usr/bin/env python3
"""Zip the overleaf/ folder into an uploadable Overleaf project bundle.

Usage:
    python overleaf/zip_overleaf_project.py

Creates: outputs/release/overleaf_project.zip
"""

import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLEAF_DIR = REPO_ROOT / "overleaf"
OUT_DIR = REPO_ROOT / "outputs" / "release"
ZIP_NAME = "overleaf_project.zip"

# Figure source → overleaf destination mapping
FIGURE_COPIES = {
    "outputs/qc_figures/usable_duration_hist.png": "figures/fig2_usable_duration.png",
    "outputs/qc_figures/rejection_fraction_hist.png": "figures/fig2_rejection_fraction.png",
    "outputs/figures/features/age_vs_paf.png": "figures/fig3_age_vs_paf.png",
    "outputs/figures/features/paf_histogram.png": "figures/fig3_paf_hist.png",
    "outputs/figures/features/aperiodic_exponent_hist.png": "figures/fig3_aperiodic_hist.png",
    "outputs/figures/features/global_bandpower_boxplot.png": "figures/fig3_bandpower_boxplot.png",
    "outputs/figures/model/calibration_pred_vs_true.png": "figures/fig4_pred_vs_true.png",
    "outputs/figures/model/residuals_vs_age.png": "figures/fig4_residuals.png",
    "outputs/figures/model/model_comparison.png": "figures/fig4_model_comparison.png",
    "outputs/figures/explain/importance_bar_top20.png": "figures/fig5_importance.png",
    "outputs/figures/explain/ablation_mae_comparison.png": "figures/fig5_ablation.png",
}

# Excluded patterns
EXCLUDE_SUFFIXES = {".pyc", ".DS_Store"}
EXCLUDE_NAMES = {"__pycache__", ".git"}


def main():
    # Refresh figures
    print("Copying figures to overleaf/figures/ ...")
    for src_rel, dst_rel in FIGURE_COPIES.items():
        src = REPO_ROOT / src_rel
        dst = OVERLEAF_DIR / dst_rel
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  {src_rel} -> overleaf/{dst_rel}")
        else:
            print(f"  MISSING: {src_rel}")

    # Build zip
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / ZIP_NAME

    print(f"\nCreating {zip_path} ...")
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(OVERLEAF_DIR.rglob("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix in EXCLUDE_SUFFIXES:
                continue
            if any(part in EXCLUDE_NAMES for part in fpath.parts):
                continue
            arcname = fpath.relative_to(OVERLEAF_DIR)
            zf.write(fpath, arcname)
            n_files += 1

    size_kb = zip_path.stat().st_size / 1024
    print(f"Done: {n_files} files, {size_kb:.0f} KB")
    print(f"Upload to Overleaf: {zip_path}")


if __name__ == "__main__":
    main()
