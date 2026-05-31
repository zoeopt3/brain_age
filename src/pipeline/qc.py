"""QC metric computation and drop-log management.

Computes per-recording quality metrics (line noise, muscle artifact,
channel variance, usable duration) and writes structured drop logs
with machine-readable reason codes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import numpy as np


# ── Reason codes for drop log ──────────────────────────────────────────

class DropReason:
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    LOAD_ERROR = "LOAD_ERROR"
    NO_EEG_CHANNELS = "NO_EEG_CHANNELS"
    TOO_SHORT = "TOO_SHORT"
    EXCESSIVE_ARTIFACT = "EXCESSIVE_ARTIFACT"
    PREPROCESS_ERROR = "PREPROCESS_ERROR"
    QC_ERROR = "QC_ERROR"


# ── Channel helpers ────────────────────────────────────────────────────

def _resolve_posterior_channels(
    raw_info: mne.Info,
    preferred: list[str],
) -> list[str]:
    """Find posterior channels by name, falling back to prefix matching.

    Args:
        raw_info: MNE Info object.
        preferred: List of preferred channel names from config.

    Returns:
        List of available channel names that are posterior.
    """
    all_ch = raw_info["ch_names"]
    eeg_picks = mne.pick_types(raw_info, eeg=True, exclude=[])
    eeg_names = set(np.array(all_ch)[eeg_picks])

    # Try exact matches first
    found = [ch for ch in preferred if ch in eeg_names]
    if found:
        return found

    # Fallback: prefix match for occipital/parietal
    prefixes = ("O", "PO", "Pz", "P3", "P4", "P7", "P8")
    found = [ch for ch in eeg_names if any(ch.startswith(p) for p in prefixes)]
    return sorted(found)[:6]  # cap at 6


# ── Spectral helpers ───────────────────────────────────────────────────

def _band_power(psd: np.ndarray, freqs: np.ndarray,
                fmin: float, fmax: float) -> float:
    """Mean power in a frequency band (linear scale)."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(np.mean(psd[:, mask]))


# ── Main QC computation ───────────────────────────────────────────────

def compute_qc_metrics(
    raw: mne.io.BaseRaw,
    epochs: mne.Epochs | None,
    rejection_info: dict[str, Any],
    row: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute a dictionary of QC metrics for one recording.

    Args:
        raw: Preprocessed Raw object.
        epochs: Epochs after artifact rejection (may be None if epoching failed).
        rejection_info: Output of ``reject_bad_epochs``.
        row: Manifest row for this recording.
        cfg: The ``qc`` section of stage2.yml.

    Returns:
        Flat dictionary of metrics suitable for a DataFrame row.
    """
    epoch_length = cfg.get("epoch_length_sec", 2.0)
    if isinstance(epoch_length, str):
        epoch_length = 2.0

    # -- from preprocess config passed through --
    pp_cfg = cfg  # caller merges preprocessing + qc sections

    info = raw.info
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
    n_eeg = len(eeg_picks)

    metrics: dict[str, Any] = {
        "sub": row.get("sub", ""),
        "ses": row.get("ses", ""),
        "task": row.get("task", ""),
        "run": row.get("run", ""),
        "condition": row.get("condition", "unknown"),
        "sfreq": info["sfreq"],
        "duration_sec": round(raw.times[-1], 2) if len(raw.times) > 0 else 0.0,
        "n_channels": len(info["ch_names"]),
        "n_eeg_channels": n_eeg,
    }

    # Bad channels
    n_bad = len(info.get("bads", []))
    metrics["n_bad_channels"] = n_bad
    metrics["pct_bad_channels"] = round(n_bad / n_eeg * 100, 1) if n_eeg > 0 else 0.0

    # Rejection info
    metrics.update(rejection_info)
    usable = rejection_info.get("n_good_epochs", 0) * epoch_length
    metrics["usable_duration_sec"] = round(usable, 2)

    # Channel variance outliers (IQR method on epoch variances)
    metrics["channel_variance_outliers"] = 0
    if epochs is not None and len(epochs) > 0 and n_eeg > 0:
        try:
            data = epochs.get_data(picks="eeg")  # (n_epochs, n_ch, n_times)
            ch_var = np.var(data, axis=2).mean(axis=0)  # (n_ch,)
            q1, q3 = np.percentile(ch_var, [25, 75])
            iqr = q3 - q1
            outliers = int(np.sum((ch_var < q1 - 3 * iqr) | (ch_var > q3 + 3 * iqr)))
            metrics["channel_variance_outliers"] = outliers
        except Exception:
            pass

    # Spectral metrics (from good epochs)
    qc_cfg = cfg
    if qc_cfg.get("compute_psd", True) and epochs is not None and len(epochs) > 0:
        try:
            psd_obj = epochs.compute_psd(
                method="welch",
                fmin=qc_cfg.get("psd_fmin", 1.0),
                fmax=qc_cfg.get("psd_fmax", 45.0),
                picks="eeg",
                verbose=False,
            )
            psd_data = psd_obj.get_data()          # (n_epochs, n_ch, n_freqs)
            freqs = psd_obj.freqs
            mean_psd = psd_data.mean(axis=0)       # (n_ch, n_freqs)

            # Line-noise proxy: ratio of power at line freq vs neighbours
            lf = qc_cfg.get("line_noise_center_hz", 60.0)
            lb = qc_cfg.get("line_noise_band_hz", 2.0)
            noise_power = _band_power(mean_psd, freqs, lf - lb, lf + lb)
            neighbour_power = (
                _band_power(mean_psd, freqs, lf - 3 * lb, lf - lb)
                + _band_power(mean_psd, freqs, lf + lb, lf + 3 * lb)
            ) / 2
            metrics["line_noise_ratio"] = (
                round(noise_power / neighbour_power, 3)
                if neighbour_power > 0 else 0.0
            )

            # Muscle proxy: ratio of high-freq power to alpha
            muscle_band = qc_cfg.get("muscle_band_hz", [30.0, 45.0])
            alpha_band = qc_cfg.get("alpha_band_hz", [8.0, 13.0])
            muscle_power = _band_power(mean_psd, freqs, *muscle_band)
            alpha_power = _band_power(mean_psd, freqs, *alpha_band)
            metrics["muscle_ratio"] = (
                round(muscle_power / alpha_power, 4)
                if alpha_power > 0 else 0.0
            )

            # Band powers (median across channels, in dB)
            bands = {
                "delta": (1, 4), "theta": (4, 8),
                "alpha": (8, 13), "beta": (13, 30),
            }
            for bname, (f1, f2) in bands.items():
                bp = _band_power(mean_psd, freqs, f1, f2)
                metrics[f"median_power_{bname}_dB"] = (
                    round(10 * np.log10(bp), 2) if bp > 0 else -999.0
                )

            # Posterior alpha (if channels available)
            posterior = _resolve_posterior_channels(
                info, qc_cfg.get("posterior_channels", [])
            )
            if posterior:
                post_idx = [info["ch_names"].index(ch) for ch in posterior
                            if ch in info["ch_names"]]
                if post_idx:
                    post_psd = mean_psd[post_idx, :]
                    pa = _band_power(post_psd, freqs, *alpha_band)
                    metrics["posterior_alpha_power_dB"] = (
                        round(10 * np.log10(pa), 2) if pa > 0 else -999.0
                    )

        except Exception:
            metrics["line_noise_ratio"] = None
            metrics["muscle_ratio"] = None

    return metrics


# ── Drop log ───────────────────────────────────────────────────────────

def append_drop_log(
    path: str | Path,
    sub: str,
    reason: str,
    detail: str = "",
) -> None:
    """Append one JSON-lines entry to the drop log.

    Args:
        path: Path to the JSONL file.
        sub: Subject / participant ID.
        reason: Machine-readable reason code (see ``DropReason``).
        detail: Optional human-readable detail.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sub": sub,
        "reason": reason,
        "detail": detail,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
