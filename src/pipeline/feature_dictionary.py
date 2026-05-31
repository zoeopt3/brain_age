"""Generate a plain-language feature dictionary as Markdown."""

from __future__ import annotations

from typing import Any

_DESCRIPTIONS = {
    # Identifiers
    "sub": ("Participant ID", "Subject identifier from the BIDS dataset."),
    "ses": ("Session", "Recording session (e.g. session1, session2)."),
    "task": ("Task", "BIDS task label (e.g. eyesopen, eyesclosed)."),
    "run": ("Run", "Run number within a session/task."),
    "condition": ("Condition", "Eyes-open (EO), eyes-closed (EC), or unknown."),
    "age": ("Age", "Participant age in years."),
    "sex": ("Sex", "Participant sex (m/f)."),

    # QC carry-through
    "usable_duration_sec": ("Usable duration", "Seconds of clean EEG after artifact rejection."),
    "rejection_fraction": ("Rejection fraction", "Fraction of epochs rejected during QC."),

    # Global band powers
    "global_bp_delta": ("Global delta power", "log10 band power 1-4 Hz, averaged across all EEG channels."),
    "global_bp_theta": ("Global theta power", "log10 band power 4-8 Hz, averaged across all EEG channels."),
    "global_bp_alpha": ("Global alpha power", "log10 band power 8-12 Hz, averaged across all EEG channels."),
    "global_bp_beta": ("Global beta power", "log10 band power 13-30 Hz, averaged across all EEG channels."),

    # PAF
    "paf_hz": ("Peak alpha frequency", "Frequency (Hz) of the spectral peak in the alpha range (6-13 Hz), "
               "estimated from posterior channels (O, PO, P). NaN if no clear peak detected."),
    "iaf_bandpower": ("Individualized alpha power", "log10 band power in a ±2 Hz window centered on the individual's "
                      "peak alpha frequency. Captures alpha power at the subject's own dominant frequency."),

    # Ratios
    "theta_alpha_ratio": ("Theta/alpha ratio", "Ratio of theta (4-8 Hz) to alpha (8-12 Hz) absolute power. "
                          "Higher values may indicate increased slow-wave activity relative to alpha."),
    "theta_beta_ratio": ("Theta/beta ratio", "Ratio of theta (4-8 Hz) to beta (13-30 Hz) absolute power."),

    # Aperiodic
    "ap_exponent": ("Aperiodic exponent", "Slope of the 1/f background spectrum (specparam fit). "
                    "More negative values indicate steeper spectral falloff. Changes with development."),
    "ap_offset": ("Aperiodic offset", "Y-intercept of the 1/f background spectrum (specparam fit). "
                  "Reflects broadband power level."),
    "alpha_peak_center_hz": ("Alpha peak center (specparam)", "Center frequency of the alpha peak identified by "
                             "specparam model fitting. May differ slightly from PAF."),
    "alpha_peak_power": ("Alpha peak power (specparam)", "Height (above aperiodic background) of the alpha peak "
                         "identified by specparam."),

    # Flags
    "psd_ok": ("PSD computed", "True if PSD computation succeeded for this recording."),
    "specparam_ok": ("Specparam fit OK", "True if the specparam (FOOOF) aperiodic fit succeeded."),
}

# Regional band power descriptions are generated dynamically
_REGION_NAMES = ["frontal", "central", "parietal", "occipital", "temporal"]
_BAND_NAMES = ["delta", "theta", "alpha", "beta"]

for region in _REGION_NAMES:
    for band in _BAND_NAMES:
        key = f"{region}_bp_{band}"
        _DESCRIPTIONS[key] = (
            f"{region.capitalize()} {band} power",
            f"log10 band power for the {band} band in {region} channels.",
        )


def generate_feature_dictionary(
    feature_columns: list[str],
    cfg: dict[str, Any],
) -> str:
    """Generate a Markdown feature dictionary.

    Args:
        feature_columns: List of column names from the features table.
        cfg: Full stage3 config (for band definitions, etc.).

    Returns:
        Markdown string.
    """
    bands = cfg.get("bands", {})
    lines = [
        "# Feature Dictionary",
        "",
        "This document describes every column in `outputs/features.parquet`.",
        "",
        "## Band Definitions",
        "",
        "| Band | Range (Hz) |",
        "|------|-----------|",
    ]
    for bname, (f1, f2) in bands.items():
        lines.append(f"| {bname} | {f1} - {f2} |")
    lines.append("")
    lines.append("Band powers are computed via trapezoidal integration of the Welch PSD "
                 "and reported as **log10(V^2)** by default.")
    lines.append("")

    lines.append("## Feature Descriptions")
    lines.append("")
    lines.append("| Column | Label | Description |")
    lines.append("|--------|-------|-------------|")

    for col in feature_columns:
        if col in _DESCRIPTIONS:
            label, desc = _DESCRIPTIONS[col]
        else:
            label = col.replace("_", " ").title()
            desc = "(auto-generated — no specific description available)"
        lines.append(f"| `{col}` | {label} | {desc} |")

    lines.append("")
    lines.append("---")
    lines.append("*This dictionary is auto-generated from the feature extraction config. "
                 "Band powers use log10 transform unless otherwise configured.*")
    return "\n".join(lines)
