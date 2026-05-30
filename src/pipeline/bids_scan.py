"""BIDS directory scanner for EEG datasets.

Finds EEG recording files, extracts BIDS entities (subject, session,
task, run) from filenames, and detects eyes-open / eyes-closed conditions.
"""

import re
from pathlib import Path
from typing import Any


# Common EEG file extensions in BIDS
EEG_EXTENSIONS = {".set", ".edf", ".bdf", ".vhdr", ".fdt", ".eeg"}

# Regex for BIDS key-value entities in filenames
_ENTITY_RE = re.compile(r"(sub|ses|task|run|acq)-([A-Za-z0-9]+)")

# Task names that suggest eyes-open or eyes-closed
_EO_KEYWORDS = {"resteyesopen", "resteo", "eo", "eyesopen", "open"}
_EC_KEYWORDS = {"resteyesclosed", "restec", "ec", "eyesclosed", "closed"}


def extract_bids_entities(filename: str) -> dict[str, str]:
    """Parse BIDS key-value entities from a filename.

    Example:
        ``sub-NDARAB123_ses-01_task-restEO_eeg.set``
        -> ``{"sub": "NDARAB123", "ses": "01", "task": "restEO"}``

    Args:
        filename: Filename (not full path) of a BIDS file.

    Returns:
        Dictionary of entity key -> value.
    """
    return dict(_ENTITY_RE.findall(filename))


def classify_condition(task_label: str | None) -> str:
    """Classify a BIDS task label as EO, EC, or unknown.

    Args:
        task_label: The ``task-XXX`` value from a BIDS filename, or None.

    Returns:
        One of ``"EO"``, ``"EC"``, or ``"unknown"``.
    """
    if task_label is None:
        return "unknown"
    normalised = task_label.lower().replace("_", "").replace("-", "")
    if normalised in _EO_KEYWORDS:
        return "EO"
    if normalised in _EC_KEYWORDS:
        return "EC"
    # Partial match fallback
    if "open" in normalised:
        return "EO"
    if "closed" in normalised or "close" in normalised:
        return "EC"
    return "unknown"


def find_eeg_files(bids_root: str | Path) -> list[dict[str, Any]]:
    """Recursively find EEG recording files under a BIDS root.

    For each file found, extracts BIDS entities and classifies the
    eyes-open / eyes-closed condition from the task label.

    Args:
        bids_root: Path to the BIDS dataset root directory.

    Returns:
        List of dicts, each with keys:
        ``path``, ``filename``, ``sub``, ``ses``, ``task``, ``run``,
        ``condition``, ``extension``.
    """
    bids_root = Path(bids_root)
    if not bids_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for fpath in sorted(bids_root.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in EEG_EXTENSIONS:
            continue
        # Skip derivative/hidden directories
        rel = fpath.relative_to(bids_root)
        parts = rel.parts
        if any(p.startswith(".") or p == "derivatives" for p in parts):
            continue

        entities = extract_bids_entities(fpath.name)
        records.append({
            "path": str(fpath),
            "rel_path": str(rel),
            "filename": fpath.name,
            "sub": entities.get("sub", ""),
            "ses": entities.get("ses", ""),
            "task": entities.get("task", ""),
            "run": entities.get("run", ""),
            "condition": classify_condition(entities.get("task")),
            "extension": fpath.suffix.lower(),
        })

    return records


def list_subject_dirs(bids_root: str | Path) -> list[str]:
    """List subject directory names (``sub-XXXX``) found under a BIDS root.

    Args:
        bids_root: Path to the BIDS dataset root.

    Returns:
        Sorted list of subject IDs (e.g. ``["NDARAB123", "NDARCD456"]``).
    """
    bids_root = Path(bids_root)
    if not bids_root.exists():
        return []
    subject_ids = []
    for d in sorted(bids_root.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subject_ids.append(d.name.removeprefix("sub-"))
    return subject_ids
