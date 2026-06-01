# Figure Mapping

Each figure in `main.tex` corresponds to a pipeline output. This table shows the mapping:

| LaTeX filename | Pipeline source | Description | Status |
|---------------|----------------|-------------|--------|
| `fig2_usable_duration.png` | `outputs/qc_figures/usable_duration_hist.png` | QC: usable duration distribution | Present |
| `fig2_rejection_fraction.png` | `outputs/qc_figures/rejection_fraction_hist.png` | QC: rejection fraction distribution | Present |
| `fig3_age_vs_paf.png` | `outputs/figures/features/age_vs_paf.png` | Age vs PAF scatter | Present |
| `fig3_paf_hist.png` | `outputs/figures/features/paf_histogram.png` | PAF distribution | Present |
| `fig3_aperiodic_hist.png` | `outputs/figures/features/aperiodic_exponent_hist.png` | Aperiodic exponent distribution | Present |
| `fig3_bandpower_boxplot.png` | `outputs/figures/features/global_bandpower_boxplot.png` | Global band power boxplot | Present |
| `fig4_pred_vs_true.png` | `outputs/figures/model/calibration_pred_vs_true.png` | Predicted vs true age | Present |
| `fig4_residuals.png` | `outputs/figures/model/residuals_vs_age.png` | Residuals vs age | Present |
| `fig4_model_comparison.png` | `outputs/figures/model/model_comparison.png` | Model MAE comparison | Present |
| `fig5_importance.png` | `outputs/figures/explain/importance_bar_top20.png` | Feature importance (top 20) | Present |
| `fig5_ablation.png` | `outputs/figures/explain/ablation_mae_comparison.png` | Ablation MAE comparison | Present |

## To update figures

Re-run the pipeline stages and copy updated PNGs to this folder using:

```bash
python overleaf/zip_overleaf_project.py
```

This script re-copies all figures and rebuilds the zip.
