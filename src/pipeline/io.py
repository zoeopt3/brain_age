"""File I/O helpers: directory creation, JSON/Parquet output, file hashing."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(cfg: dict) -> list[str]:
    """Create all directories declared under ``cfg["paths"]``.

    Args:
        cfg: Full project configuration (must contain a ``paths`` section).

    Returns:
        Sorted list of directory paths that were ensured to exist.
    """
    paths_section = cfg.get("paths", {})
    created: list[str] = []
    for key, dir_path in paths_section.items():
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        created.append(str(p))
    return sorted(created)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Write a Python object as pretty-printed JSON.

    Args:
        path: Destination file path.
        obj: JSON-serializable object.
        indent: Indentation level (default 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def hash_file(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute the hex digest of a file.

    Args:
        path: File to hash.
        algorithm: Hash algorithm name (default ``sha256``).

    Returns:
        Hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash — file not found: {path}")
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a BIDS-style TSV file into a DataFrame.

    Handles 'n/a' as missing values (BIDS convention).

    Args:
        path: Path to the TSV file.

    Returns:
        DataFrame with parsed contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")
    return pd.read_csv(path, sep="\t", na_values=["n/a", "N/A", ""])


def write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to Parquet format.

    Args:
        path: Destination file path.
        df: DataFrame to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_markdown(path: str | Path, text: str) -> None:
    """Write a string to a Markdown file.

    Args:
        path: Destination file path.
        text: Markdown content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
