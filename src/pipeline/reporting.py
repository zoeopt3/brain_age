"""Final report generation in Markdown.

Renders a beginner-friendly report summarizing the entire pipeline:
data, QC, features, modeling, explainability, robustness, and limitations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def render_final_report(cfg: dict, inputs: dict) -> str:
    """Generate the full final report as a Markdown string.

    Args:
        cfg: Stage 6 config.
        inputs: Dict with loaded data:
            features_df, model_results, robustness_results,
            importance_df, dataset_fingerprint, qc_report_text.

    Returns:
        Markdown string.
    """
    title = cfg.get("report", {}).get("title", "EEG Brain-Age Clock")
    feat_df = inputs.get("features_df", pd.DataFrame())
    model_res = inputs.get("model_results", {})
    robust_res = inputs.get("robustness_results", {})
    imp_df = inputs.get("importance_df", pd.DataFrame())
    fingerprint = inputs.get("dataset_fingerprint", {})

    lines = [
        f"# {title}",
        "",
        f"*Report generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}*",
        "",
        "> **Important:** This project is for research and education only. "
        "It is not a medical device, diagnostic tool, or clinical decision-support system. "
        "Model predictions have not been clinically validated and must not be used to make "
        "decisions about any individual's health or development.",
        "",
        "---",
        "",
        "## 1. What Is This Project?",
        "",
        "Your brain changes as you grow up. One way scientists can see these changes is by ",
        "measuring the tiny electrical signals your brain produces, using a technique called ",
        "**EEG** (electroencephalography). These signals create patterns — brain waves — that ",
        "shift predictably with age.",
        "",
        "This project builds a \"brain-age clock\": a computer model that looks at someone's ",
        "brain-wave patterns and tries to guess their age. By studying which patterns the model ",
        "pays attention to, we can learn about how the brain develops during childhood and adolescence.",
        "",
    ]

    # ── Data section ───────────────────────────────────────────────────
    lines.extend([
        "## 2. The Data",
        "",
    ])
    ds_id = fingerprint.get("dataset_id", "unknown")
    n_subs = fingerprint.get("n_subjects_downloaded", "?")
    if not feat_df.empty:
        n_rows = len(feat_df)
        n_parts = feat_df["sub"].nunique() if "sub" in feat_df.columns else "?"
        age_min = feat_df["age"].min() if "age" in feat_df.columns else "?"
        age_max = feat_df["age"].max() if "age" in feat_df.columns else "?"
        conditions = feat_df["condition"].value_counts().to_dict() if "condition" in feat_df.columns else {}
        lines.extend([
            f"We used a public EEG dataset from OpenNeuro (dataset **{ds_id}**). "
            f"After downloading and quality checking, we had:",
            "",
            f"- **{n_parts} participants** (ages {age_min}–{age_max})",
            f"- **{n_rows} recordings** total",
        ])
        if conditions:
            cond_str = ", ".join(f"{k}: {v}" for k, v in conditions.items())
            lines.append(f"- Conditions: {cond_str}")
        lines.append("")
        lines.append("Each recording captures about 5 minutes of resting brain activity, "
                      "measured through electrodes placed on the scalp.")
        lines.append("")
    else:
        lines.append("*(Feature data not available for summary.)*\n")

    # ── QC section ─────────────────────────────────────────────────────
    lines.extend([
        "## 3. Quality Control",
        "",
        "Not all brain recordings are usable — muscle movements, eye blinks, and electrical ",
        "noise can contaminate the signal. We applied automated quality checks:",
        "",
        "- **Bandpass filter** (1–40 Hz): removes very slow drifts and high-frequency noise",
        "- **Notch filter** (60 Hz): removes power-line interference",
        "- **Artifact rejection**: segments with unusually large or flat signals are discarded",
        "",
        "Recordings with less than 60 seconds of clean data after rejection were flagged. ",
        "See `outputs/qc_report.md` for detailed QC statistics.",
        "",
    ])

    # ── Features section ───────────────────────────────────────────────
    n_features = len([c for c in feat_df.columns
                      if c not in ("sub", "ses", "task", "run", "condition", "age", "sex",
                                   "psd_ok", "specparam_ok")]) if not feat_df.empty else 0
    lines.extend([
        "## 4. Features: What We Measured",
        "",
        f"From each clean recording, we extracted **{n_features} features** that describe ",
        "the brain's electrical activity:",
        "",
        "- **Band powers**: How much energy is in different frequency ranges — delta (1–4 Hz), ",
        "  theta (4–8 Hz), alpha (8–12 Hz), and beta (13–30 Hz). These were measured globally ",
        "  and for five scalp regions (frontal, central, parietal, occipital, temporal).",
        "",
        "- **Peak alpha frequency (PAF)**: The exact frequency where the alpha rhythm is strongest, ",
        "  measured from the back of the head. PAF tends to increase as children grow older.",
        "",
        "- **Individualized alpha power**: Alpha energy measured in a personalized window centered ",
        "  on each person's own PAF, rather than a fixed 8–12 Hz band.",
        "",
        "- **Aperiodic (1/f) slope**: The overall \"tilt\" of the power spectrum. Younger brains ",
        "  tend to have steeper slopes; as the brain matures, the slope flattens.",
        "",
        "See `reports/feature_dictionary.md` for a complete description of every feature.",
        "",
    ])

    # ── Modeling section ───────────────────────────────────────────────
    models = model_res.get("models", {})
    best = model_res.get("best_model", "?")
    lines.extend([
        "## 5. Modeling: Predicting Age",
        "",
        "We trained several models to predict age from the EEG features:",
        "",
        "| Model | MAE (years) | R² |",
        "|-------|------------|-----|",
    ])
    for mname, mres in models.items():
        mae = f"{mres['mae_mean']:.2f} ± {mres['mae_std']:.2f}"
        r2 = f"{mres['r2_mean']:.3f} ± {mres['r2_std']:.3f}"
        marker = " **(best)**" if mname == best else ""
        lines.append(f"| {mname}{marker} | {mae} | {r2} |")
    lines.extend([
        "",
        f"The best model ({best}) achieved a mean absolute error of "
        f"**{models.get(best, {}).get('mae_mean', '?'):.2f} years**.",
        "",
        "**How we prevented cheating (data leakage):** We used GroupKFold cross-validation, ",
        "which ensures that all recordings from the same person are always in the same group. ",
        "The model never sees a person's data during training that it will be tested on.",
        "",
        "![Predicted vs True Age](outputs/figures/model/calibration_pred_vs_true.png)",
        "",
    ])

    # ── Explainability section ─────────────────────────────────────────
    lines.extend([
        "## 6. What the Model Pays Attention To",
        "",
    ])
    if not imp_df.empty:
        lines.append("The top features by permutation importance (how much MAE increases "
                      "when a feature is scrambled):")
        lines.append("")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for i, row in imp_df.head(10).iterrows():
            lines.append(f"| {i+1} | {row['feature']} | {row['importance_mean']:.4f} |")
        lines.append("")
    lines.extend([
        "**Key takeaways:**",
        "",
        "- The aperiodic (1/f) slope captures how the overall spectral background changes with age",
        "- Peak alpha frequency reflects thalamocortical circuit maturation",
        "- Regional band powers show that different brain areas develop at different rates",
        "",
        "![Feature Importance](outputs/figures/explain/importance_bar_top20.png)",
        "",
    ])

    # ── Robustness section ─────────────────────────────────────────────
    lines.extend([
        "## 7. Robustness Checks",
        "",
        "We tested whether our results are stable under different conditions:",
        "",
    ])
    eoec = robust_res.get("eo_vs_ec", {})
    if eoec and "error" not in eoec:
        lines.extend([
            "### Eyes-Open vs Eyes-Closed",
            "",
            "| Condition | MAE (years) | Participants |",
            "|-----------|------------|--------------|",
        ])
        for cond in ["EO", "EC", "combined"]:
            if cond in eoec and "mae_mean" in eoec[cond]:
                lines.append(f"| {cond} | {eoec[cond]['mae_mean']:.2f} ± {eoec[cond]['mae_std']:.2f} "
                             f"| {eoec[cond].get('n_participants', '?')} |")
        lines.append("")

    fam = robust_res.get("feature_family_sensitivity", {})
    if fam:
        lines.extend([
            "### Feature Family Sensitivity",
            "",
            "| Feature Set | MAE (years) | # Features |",
            "|------------|------------|------------|",
        ])
        for exp_name, exp_res in fam.items():
            if isinstance(exp_res, dict) and "mae_mean" in exp_res:
                lines.append(f"| {exp_name.replace('_', ' ').title()} "
                             f"| {exp_res['mae_mean']:.2f} ± {exp_res['mae_std']:.2f} "
                             f"| {exp_res.get('n_features', '?')} |")
        lines.append("")

    if robust_res:
        lines.append("![Robustness Comparison](outputs/figures/robustness/robustness_mae_comparison.png)")
        lines.append("")

    # ── Limitations ────────────────────────────────────────────────────
    lines.extend([
        "## 8. Limitations and Ethics",
        "",
        "This project has important limitations:",
        "",
        "- **Not a diagnostic tool.** This model cannot and should not be used to diagnose "
        "any medical condition or make clinical decisions.",
        "- **Narrow age range.** The current dataset covers ages 18–22; a wider range (5–21) "
        "would show clearer developmental trends.",
        "- **Small sample size.** With 20 participants, results have high statistical uncertainty.",
        "- **EEG artifacts.** Despite quality control, some noise may remain in the data.",
        "- **Model limitations.** The model learns correlations, not causes. Brain-wave patterns "
        "associated with age may also be affected by sleep, caffeine, medications, or attention.",
        "- **Privacy.** The dataset is de-identified. Do not attempt to re-identify participants.",
        "",
    ])

    # ── Reproducibility ────────────────────────────────────────────────
    config_hash = model_res.get("config_hash", "?")
    git_hash = model_res.get("git_hash", "?")[:12]
    lines.extend([
        "## 9. Reproducibility",
        "",
        f"- Dataset: `{ds_id}`",
        f"- Config hash: `{config_hash}`",
        f"- Git commit: `{git_hash}`" if git_hash and git_hash != "?" else "",
        f"- Participants: {n_subs}",
        "- Random seed: 1337",
        "- All pipeline stages are scripted and config-driven",
        "",
        "To reproduce: run Stages 0–6 in order using the configs in `configs/`.",
        "",
        "---",
        "",
        "*This report was auto-generated by the EEG Brain-Age Clock pipeline. "
        "All results are for research and education purposes only.*",
    ])

    return "\n".join(lines)
