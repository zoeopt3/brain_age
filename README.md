# EEG Brain-Age Clock

An explainable "brain-age clock" that predicts chronological age (5-21 years) from resting-state EEG recordings. The project uses the Healthy Brain Network (HBN) dataset and builds interpretable machine-learning models so that the features driving each prediction are transparent and auditable.

This is a local research workflow organized into stages: each stage reads from the previous stage's outputs and writes its own, making the full pipeline reproducible from raw data to final figures.

---

## Guardrails

- **Research and education only.** This project is an exploratory research tool. It is not a diagnostic device, clinical decision-support system, or medical product.
- **Not validated for clinical use.** Model outputs (predicted brain age, feature importances) have not been clinically validated and must not be used to diagnose, treat, or make decisions about any individual.
- **Privacy.** The HBN dataset is de-identified. Do not attempt to re-identify participants.

---

## Stage Plan

| Stage | Name | Description |
|-------|------|-------------|
| **0** | Setup | Project skeleton, config validation, smoke test (this stage) |
| **1** | Download | Fetch a subset of ds004186 (HBN) + build participant manifest |
| **2** | QC | Automated quality control, artifact rejection, bad-channel detection |
| **3** | Features | Extract spectral power, connectivity, and complexity features |
| **4** | Model | Train age-prediction models (ridge, gradient boosting, etc.) |
| **5** | Explain | SHAP / permutation importance, topographic maps, age-trajectory plots |
| **6** | Report | Final figures, tables, and reproducibility summary |

---

## Quickstart (Stage 0)

```bash
# 1. Create the conda environment
conda env create -f environment.yml
conda activate eeg-brain-age

# 2. Run the Stage 0 smoke test
python scripts/stage0_smoke_test.py --config configs/project.yml

# 3. Check the output
cat outputs/stage0_ok.json
```

If you see `"status": "ok"` in the JSON, the skeleton is working.

---

## Stage 1: Download + Manifest

Stage 1 downloads a small subset of the HBN dataset (default 20 subjects) and builds a participant manifest with age labels.

```bash
# 1. Download subset (requires internet)
python scripts/stage1_download_subset.py --config configs/stage1.yml

# 2. Build manifest and sanity report
python scripts/stage1_build_manifest.py --config configs/stage1.yml

# 3. Check outputs
cat outputs/dataset_fingerprint.json
cat outputs/bids_sanity_report.md
```

**Expected outputs:**
- `outputs/dataset_fingerprint.json` — dataset ID, download timestamp, config hash
- `outputs/manifest.parquet` — participant_id, age, condition (EO/EC), file paths
- `outputs/bids_sanity_report.md` — counts, missingness, duplicates, TSV/folder mismatches

---

## Stage 2: QC + Preprocessing

Stage 2 loads each EEG recording from the manifest, applies minimal preprocessing (bandpass 1-40 Hz, 60 Hz notch, average reference), rejects artifacts, and computes QC metrics.

```bash
python scripts/stage2_run_qc.py --config configs/stage2.yml
```

**Expected outputs:**
- `outputs/qc_summary.parquet` — per-recording QC metrics (duration, rejection rate, noise proxies, band powers)
- `outputs/drop_log.jsonl` — recordings that failed or were flagged, with reason codes
- `outputs/qc_report.md` — human-readable summary of QC results
- `outputs/qc_figures/` — distribution plots (usable duration, rejection fraction, line noise, muscle artifact)

**Troubleshooting:**
- *Missing montage/channel names:* MNE may warn about unrecognized channels. The pipeline picks EEG channels by type and falls back to prefix matching for posterior channels.
- *Memory issues:* Each recording is loaded, processed, and released one at a time. No large intermediates are kept in memory.
- *High rejection rates:* Review `outputs/qc_report.md` for recommended thresholds. Consider adjusting `peak_to_peak_uV` in `configs/stage2.yml`.

---

## Stage 3: Feature Extraction

Stage 3 extracts spectral features from QC-passed recordings: band powers, peak alpha frequency, individualized alpha power, and aperiodic (1/f) parameters.

```bash
python scripts/stage3_extract_features.py --config configs/stage3.yml
```

**Expected outputs:**
- `outputs/features.parquet` — single-row-per-recording feature table for modeling
- `outputs/feature_qc.parquet` — fit success/failure flags per recording
- `reports/feature_dictionary.md` — plain-language description of each feature
- `outputs/figures/features/` — age vs PAF scatter, band power boxplots, aperiodic distributions

**Key features extracted:**
- Global and regional band powers (delta, theta, alpha, beta) in log10(V^2)
- Peak alpha frequency (PAF) from posterior channels
- Individualized alpha power (PAF +/- 2 Hz)
- Aperiodic exponent and offset via specparam
- Theta/alpha and theta/beta ratios

---

## Repo Structure

```
configs/            YAML configuration files
data/
  raw/              Raw BIDS data (not committed)
  interim/          Intermediate processing outputs
  processed/        Final cleaned features
outputs/            Stage manifests, model artifacts
  figures/          Plots and topographic maps
reports/            Generated reports
src/
  pipeline/         Core utilities (config, I/O, logging)
scripts/            Stage runner scripts
```
