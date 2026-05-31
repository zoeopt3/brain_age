"""Spectral feature extraction for the EEG brain-age pipeline.

Computes band powers (global + regional), peak alpha frequency (PAF),
individualized alpha power, and aperiodic 1/f parameters via specparam.
All functions operate on a single recording and return dicts suitable
for DataFrame rows.
"""

from __future__ import annotations

import warnings
from typing import Any

import mne
import numpy as np

# ── PSD computation ────────────────────────────────────────────────────

def compute_psd(
    epochs: mne.Epochs,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute PSD from cleaned epochs using Welch's method.

    Args:
        epochs: Cleaned MNE Epochs (EEG channels only).
        cfg: The ``psd`` section of stage3.yml.

    Returns:
        (freqs, psd, ch_names) where psd is (n_channels, n_freqs)
        averaged across epochs.  Units: V^2/Hz.
    """
    fmin = cfg.get("fmin", 1.0)
    fmax = cfg.get("fmax", 45.0)

    psd_obj = epochs.compute_psd(
        method=cfg.get("method", "welch"),
        fmin=fmin,
        fmax=fmax,
        picks="eeg",
        verbose=False,
    )

    # Average across epochs (mean or median)
    psd_all = psd_obj.get_data()  # (n_epochs, n_ch, n_freqs)
    avg = cfg.get("average", "median")
    if avg == "median":
        psd = np.median(psd_all, axis=0)
    else:
        psd = np.mean(psd_all, axis=0)

    freqs = psd_obj.freqs
    ch_names = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True, exclude=[])]

    return freqs, psd, ch_names


# ── Channel selection helpers ──────────────────────────────────────────

def _channels_by_prefix(ch_names: list[str], prefixes: list[str]) -> list[str]:
    """Select channels whose names start with any of the given prefixes."""
    selected = []
    for ch in ch_names:
        for pfx in prefixes:
            if ch.startswith(pfx) and ch not in selected:
                selected.append(ch)
                break
    return selected


def select_posterior_channels(
    ch_names: list[str],
    cfg: dict[str, Any],
) -> list[str]:
    """Select posterior channels for PAF estimation.

    Tries prefix-based matching (O, PO, P by default).

    Args:
        ch_names: Available EEG channel names.
        cfg: The ``paf`` config section.

    Returns:
        List of posterior channel names (may be empty).
    """
    prefixes = cfg.get("posterior_priority_prefixes", ["O", "PO", "P"])
    return _channels_by_prefix(ch_names, prefixes)


def build_region_map(
    ch_names: list[str],
    cfg: dict[str, Any],
) -> dict[str, list[int]]:
    """Map channel indices to scalp regions using prefix matching.

    Args:
        ch_names: Available EEG channel names.
        cfg: The ``regions.definitions`` config section.

    Returns:
        Dict of region_name -> list of channel indices.
    """
    region_map: dict[str, list[int]] = {}
    for region_key, prefixes in cfg.items():
        region_name = region_key.replace("_prefixes", "")
        indices = []
        for i, ch in enumerate(ch_names):
            for pfx in prefixes:
                if ch.startswith(pfx):
                    indices.append(i)
                    break
        if indices:
            region_map[region_name] = indices
    return region_map


# ── Band power ─────────────────────────────────────────────────────────

def bandpower(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    log_transform: bool = True,
) -> float:
    """Compute absolute band power via trapezoidal integration.

    Args:
        freqs: Frequency vector.
        psd: PSD array (n_channels, n_freqs) or (n_freqs,).
        fmin: Lower band edge (Hz).
        fmax: Upper band edge (Hz).
        log_transform: If True, return log10(power).

    Returns:
        Band power (scalar). log10(V^2) if log_transform, else V^2.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return float("nan")

    if psd.ndim == 2:
        bp = np.trapz(psd[:, mask], freqs[mask], axis=1).mean()
    else:
        bp = np.trapz(psd[mask], freqs[mask])

    if bp <= 0:
        return float("nan")
    return float(np.log10(bp)) if log_transform else float(bp)


def compute_bandpowers_all(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, list[float]],
    log_transform: bool = True,
) -> dict[str, float]:
    """Compute band powers for all configured bands.

    Args:
        freqs: Frequency vector.
        psd: PSD (n_channels, n_freqs) or (n_freqs,).
        bands: Dict of band_name -> [fmin, fmax].
        log_transform: Whether to log10 the result.

    Returns:
        Dict of band_name -> power value.
    """
    result = {}
    for bname, (f1, f2) in bands.items():
        result[bname] = bandpower(freqs, psd, f1, f2, log_transform)
    return result


# ── Peak Alpha Frequency ──────────────────────────────────────────────

def compute_paf(
    freqs: np.ndarray,
    psd_posterior: np.ndarray,
    search_range: list[float],
) -> tuple[float, str]:
    """Estimate peak alpha frequency from posterior PSD.

    Finds the frequency with maximum power within the search range,
    averaged across posterior channels.

    Args:
        freqs: Frequency vector.
        psd_posterior: PSD for posterior channels (n_ch, n_freqs).
        search_range: [fmin, fmax] for the alpha search.

    Returns:
        (paf_hz, reason) where reason is "" on success or an explanation.
    """
    if psd_posterior.size == 0:
        return float("nan"), "no_posterior_channels"

    fmin, fmax = search_range
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return float("nan"), "search_range_outside_psd"

    mean_psd = psd_posterior[:, mask].mean(axis=0) if psd_posterior.ndim == 2 else psd_posterior[mask]

    peak_idx = np.argmax(mean_psd)
    paf = float(freqs[mask][peak_idx])

    # Sanity: is the peak at the edge?  If so, it may not be a real peak.
    if peak_idx == 0 or peak_idx == len(mean_psd) - 1:
        return paf, "peak_at_edge"

    return paf, ""


def compute_individualized_alpha_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    paf: float,
    half_width: float,
    clip_range: list[float],
    log_transform: bool = True,
) -> float:
    """Compute alpha power in an individualized band centered on PAF.

    Args:
        freqs: Frequency vector.
        psd: PSD (n_channels, n_freqs).
        paf: Peak alpha frequency (Hz). NaN → return NaN.
        half_width: Half-width of the individualized band (Hz).
        clip_range: [min, max] to clip the band edges.
        log_transform: Whether to log10 the result.

    Returns:
        Individualized alpha band power.
    """
    if np.isnan(paf):
        return float("nan")

    fmin = max(paf - half_width, clip_range[0])
    fmax = min(paf + half_width, clip_range[1])
    return bandpower(freqs, psd, fmin, fmax, log_transform)


# ── Aperiodic (specparam / FOOOF) ────────────────────────────────────

def fit_aperiodic_specparam(
    freqs: np.ndarray,
    psd_mean: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit aperiodic + periodic model using specparam.

    Args:
        freqs: Frequency vector.
        psd_mean: Mean PSD across channels (1-D, in V^2/Hz).
        cfg: The ``aperiodic`` config section.

    Returns:
        Dict with: ap_exponent, ap_offset, alpha_peak_center_hz,
        alpha_peak_power, specparam_ok, specparam_reason.
    """
    result = {
        "ap_exponent": float("nan"),
        "ap_offset": float("nan"),
        "alpha_peak_center_hz": float("nan"),
        "alpha_peak_power": float("nan"),
        "specparam_ok": False,
        "specparam_reason": "",
    }

    if not cfg.get("enabled", True):
        result["specparam_reason"] = "disabled"
        return result

    try:
        from specparam import SpectralModel
    except ImportError:
        result["specparam_reason"] = "specparam_not_installed"
        return result

    try:
        fit_range = cfg.get("fit_range_hz", [2.0, 40.0])
        sm = SpectralModel(
            max_n_peaks=cfg.get("max_n_peaks", 6),
            peak_width_limits=cfg.get("peak_width_limits", [1.0, 8.0]),
            min_peak_height=cfg.get("min_peak_height", 0.1),
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm.fit(freqs, psd_mean, fit_range)

        # New specparam API uses get_params(); fall back to old attributes
        try:
            ap = sm.get_params("aperiodic")
        except (AttributeError, TypeError):
            ap = getattr(sm, "aperiodic_params_", None)

        if ap is not None and len(ap) >= 2:
            result["ap_offset"] = float(ap[0])
            result["ap_exponent"] = float(ap[-1])
            result["specparam_ok"] = True

        # Extract alpha peak if found (7-14 Hz range)
        try:
            peaks = sm.get_params("peak")
        except (AttributeError, TypeError):
            peaks = getattr(sm, "peak_params_", None)

        if peaks is not None and hasattr(peaks, '__len__') and len(peaks) > 0:
            peaks = np.atleast_2d(peaks)
            alpha_peaks = peaks[(peaks[:, 0] >= 7) & (peaks[:, 0] <= 14)]
            if len(alpha_peaks) > 0:
                best = alpha_peaks[np.argmax(alpha_peaks[:, 1])]
                result["alpha_peak_center_hz"] = float(best[0])
                result["alpha_peak_power"] = float(best[1])

    except Exception as e:
        result["specparam_reason"] = str(e)[:200]

    return result


# ── Top-level feature row builder ──────────────────────────────────────

def build_feature_row(
    row: dict[str, Any],
    freqs: np.ndarray,
    psd: np.ndarray,
    ch_names: list[str],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete feature row + feature QC row for one recording.

    Args:
        row: Manifest row (sub, ses, task, condition, age, etc.).
        freqs: PSD frequency vector.
        psd: PSD array (n_channels, n_freqs).
        ch_names: Channel names matching psd rows.
        cfg: Full stage3 config.

    Returns:
        (feature_dict, feature_qc_dict)
    """
    bands = cfg.get("bands", {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [13, 30]})
    log_tf = cfg.get("output", {}).get("log_transform_bandpower", True)
    paf_cfg = cfg.get("paf", {})
    ap_cfg = cfg.get("aperiodic", {})
    region_cfg = cfg.get("regions", {})

    feat: dict[str, Any] = {
        "sub": row.get("sub", ""),
        "ses": row.get("ses", ""),
        "task": row.get("task", ""),
        "run": row.get("run", ""),
        "condition": row.get("condition", "unknown"),
    }
    # Carry through demographics and QC
    if "age" in row:
        feat["age"] = row["age"]
    if "sex" in row:
        feat["sex"] = row["sex"]
    for qc_col in ("usable_duration_sec", "rejection_fraction"):
        if qc_col in row:
            feat[qc_col] = row[qc_col]

    qc: dict[str, Any] = {
        "sub": feat["sub"], "ses": feat["ses"],
        "task": feat["task"], "condition": feat["condition"],
        "psd_ok": True, "specparam_ok": False,
        "paf_reason": "", "specparam_reason": "",
        "n_channels_used": len(ch_names),
    }

    # Global band powers
    global_bps = compute_bandpowers_all(freqs, psd, bands, log_tf)
    for bname, val in global_bps.items():
        feat[f"global_bp_{bname}"] = val

    # Regional band powers
    if region_cfg.get("enabled", False):
        region_defs = region_cfg.get("definitions", {})
        rmap = build_region_map(ch_names, region_defs)
        for region_name, ch_idx in rmap.items():
            region_psd = psd[ch_idx, :]
            rbps = compute_bandpowers_all(freqs, region_psd, bands, log_tf)
            for bname, val in rbps.items():
                feat[f"{region_name}_bp_{bname}"] = val
        qc["n_regions"] = len(rmap)

    # PAF
    posterior_ch = select_posterior_channels(ch_names, paf_cfg)
    qc["n_posterior_channels"] = len(posterior_ch)
    min_post = paf_cfg.get("fallback_min_channels", 3)

    if len(posterior_ch) >= min_post:
        post_idx = [ch_names.index(ch) for ch in posterior_ch]
        psd_post = psd[post_idx, :]
        paf_hz, paf_reason = compute_paf(
            freqs, psd_post, paf_cfg.get("search_range_hz", [6.0, 13.0])
        )
    else:
        paf_hz = float("nan")
        paf_reason = f"too_few_posterior ({len(posterior_ch)} < {min_post})"

    feat["paf_hz"] = paf_hz
    qc["paf_reason"] = paf_reason

    # Individualized alpha power
    feat["iaf_bandpower"] = compute_individualized_alpha_power(
        freqs, psd, paf_hz,
        half_width=paf_cfg.get("individualized_half_width_hz", 2.0),
        clip_range=paf_cfg.get("clip_range_hz", [4.0, 14.0]),
        log_transform=log_tf,
    )

    # Theta/alpha and theta/beta ratios
    if not np.isnan(global_bps.get("theta", float("nan"))) and not np.isnan(global_bps.get("alpha", float("nan"))):
        # Ratios on linear scale (undo log)
        if log_tf:
            theta_lin = 10 ** global_bps["theta"]
            alpha_lin = 10 ** global_bps["alpha"]
            beta_lin = 10 ** global_bps.get("beta", float("nan"))
        else:
            theta_lin = global_bps["theta"]
            alpha_lin = global_bps["alpha"]
            beta_lin = global_bps.get("beta", float("nan"))
        feat["theta_alpha_ratio"] = theta_lin / alpha_lin if alpha_lin > 0 else float("nan")
        feat["theta_beta_ratio"] = theta_lin / beta_lin if beta_lin > 0 else float("nan")

    # Aperiodic
    psd_mean = psd.mean(axis=0)
    ap_result = fit_aperiodic_specparam(freqs, psd_mean, ap_cfg)
    feat["ap_exponent"] = ap_result["ap_exponent"]
    feat["ap_offset"] = ap_result["ap_offset"]
    feat["alpha_peak_center_hz"] = ap_result["alpha_peak_center_hz"]
    feat["alpha_peak_power"] = ap_result["alpha_peak_power"]
    qc["specparam_ok"] = ap_result["specparam_ok"]
    qc["specparam_reason"] = ap_result["specparam_reason"]

    # Summary flags
    feat["psd_ok"] = True
    feat["specparam_ok"] = ap_result["specparam_ok"]

    return feat, qc
