"""Lightweight EEG preprocessing: load, filter, re-reference, artifact reject.

All parameters are config-driven.  No ICA by default — the goal is a
fast, defensible preprocessing pass that produces usable segments for
spectral feature extraction without introducing opaque transforms.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np

# Suppress MNE's verbose console output by default
mne.set_log_level("WARNING")


# ── Loading ────────────────────────────────────────────────────────────

def load_raw_from_bids(
    row: dict[str, Any],
    bids_root: str | Path,
) -> mne.io.BaseRaw | None:
    """Load a Raw object from a BIDS path identified in the manifest.

    Tries ``mne_bids.read_raw_bids`` first; falls back to direct MNE
    readers if that fails (some datasets have quirky sidecars).

    Args:
        row: A single manifest row (must contain ``rel_path`` and ``sub``).
        bids_root: Path to the BIDS dataset root.

    Returns:
        Raw object, or ``None`` if loading fails.
    """
    bids_root = Path(bids_root)
    rel_path = row.get("rel_path", row.get("path", ""))
    full_path = bids_root / rel_path if rel_path else None

    if full_path is None or not full_path.exists():
        return None

    # Try mne_bids first (handles sidecars, montages, events)
    try:
        import mne_bids
        bids_path = mne_bids.BIDSPath(
            subject=row.get("sub", ""),
            session=row.get("ses", None) or None,
            task=row.get("task", None) or None,
            run=row.get("run", None) or None,
            datatype="eeg",
            root=str(bids_root),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        return raw
    except Exception:
        pass

    # Direct fallback
    try:
        ext = full_path.suffix.lower()
        if ext == ".set":
            raw = mne.io.read_raw_eeglab(str(full_path), preload=True, verbose=False)
        elif ext in (".edf", ".bdf"):
            raw = mne.io.read_raw_edf(str(full_path), preload=True, verbose=False)
        elif ext == ".vhdr":
            raw = mne.io.read_raw_brainvision(str(full_path), preload=True, verbose=False)
        else:
            return None
        return raw
    except Exception:
        return None


# ── Preprocessing ──────────────────────────────────────────────────────

def apply_minimal_preprocessing(
    raw: mne.io.BaseRaw,
    cfg: dict[str, Any],
) -> mne.io.BaseRaw:
    """Apply bandpass, notch, optional resample, and re-reference.

    Modifies ``raw`` in place and returns it.

    Args:
        raw: Loaded MNE Raw object (must be preloaded).
        cfg: The ``preprocessing`` section of stage2.yml.

    Returns:
        The same Raw object after preprocessing.
    """
    pp = cfg

    # Pick only EEG channels for processing
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return raw

    # Band-pass filter
    l_freq = pp.get("l_freq", 1.0)
    h_freq = pp.get("h_freq", 40.0)
    raw.filter(l_freq, h_freq, picks=eeg_picks, verbose=False)

    # Notch filter
    notch_freqs = pp.get("notch_freqs", [60.0])
    if notch_freqs:
        raw.notch_filter(notch_freqs, picks=eeg_picks, verbose=False)

    # Resample
    resample_hz = pp.get("resample_hz")
    if resample_hz is not None:
        raw.resample(resample_hz, verbose=False)

    # Re-reference
    reref = pp.get("reref", "average")
    if reref == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw


# ── Epoching ───────────────────────────────────────────────────────────

def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    epoch_length_sec: float = 2.0,
) -> mne.Epochs:
    """Create fixed-length epochs from continuous data for QC / PSD.

    Args:
        raw: Preprocessed Raw object.
        epoch_length_sec: Duration of each epoch in seconds.

    Returns:
        Epochs object (not baseline-corrected).
    """
    events = mne.make_fixed_length_events(raw, duration=epoch_length_sec)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=epoch_length_sec,
        baseline=None,
        preload=True,
        verbose=False,
        reject=None,       # rejection done separately
    )
    return epochs


# ── Artifact Rejection ─────────────────────────────────────────────────

def reject_bad_epochs(
    epochs: mne.Epochs,
    cfg: dict[str, Any],
) -> tuple[mne.Epochs, dict[str, Any]]:
    """Drop epochs exceeding amplitude or flatline thresholds.

    MNE's reject/flat dicts expect values in **Volts** for EEG.
    Config specifies thresholds in **microvolts** for readability.

    Args:
        epochs: Fixed-length Epochs object.
        cfg: The ``preprocessing.reject`` section of stage2.yml.

    Returns:
        (cleaned_epochs, rejection_info) where rejection_info has
        counts and fraction.
    """
    reject_cfg = cfg.get("reject", {})
    if not reject_cfg.get("enabled", True):
        return epochs, {"n_total": len(epochs), "n_dropped": 0, "fraction": 0.0}

    peak_to_peak_uv = reject_cfg.get("peak_to_peak_uV", 200.0)
    flat_uv = reject_cfg.get("flat_uV", 1.0)

    # Convert microvolts → volts for MNE
    reject_dict = {"eeg": peak_to_peak_uv * 1e-6}
    flat_dict = {"eeg": flat_uv * 1e-6}

    n_before = len(epochs)
    epochs.drop_bad(reject=reject_dict, flat=flat_dict, verbose=False)
    n_after = len(epochs)
    n_dropped = n_before - n_after
    fraction = n_dropped / n_before if n_before > 0 else 0.0

    return epochs, {
        "n_total_epochs": n_before,
        "n_good_epochs": n_after,
        "n_dropped_epochs": n_dropped,
        "rejection_fraction": round(fraction, 4),
    }


def compute_usable_duration(
    epochs: mne.Epochs,
    epoch_length_sec: float,
) -> float:
    """Compute total usable duration in seconds after rejection."""
    return len(epochs) * epoch_length_sec
