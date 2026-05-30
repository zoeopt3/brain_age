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

## Next: Stage 1

Stage 1 will download a small subset of the HBN dataset and build a participant manifest with age labels. Run:

```bash
python scripts/stage1_download.py --config configs/project.yml
```

*(Script will be created in Stage 1.)*

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
