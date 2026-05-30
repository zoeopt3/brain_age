"""Configuration loading, merging, and validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    return cfg


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins).

    Args:
        base: Base configuration dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        Merged dictionary (new object; originals are not mutated).
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


_REQUIRED_KEYS = [
    ("project", "name"),
    ("project", "seed"),
    ("paths", "data_raw"),
    ("paths", "data_interim"),
    ("paths", "data_processed"),
    ("paths", "outputs"),
    ("dataset", "default_dataset_id"),
    ("runtime", "log_level"),
]


def validate_config_basic(cfg: dict) -> list[str]:
    """Check that required top-level and nested keys exist.

    Args:
        cfg: Loaded configuration dictionary.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    for key_path in _REQUIRED_KEYS:
        node = cfg
        for part in key_path:
            if not isinstance(node, dict) or part not in node:
                errors.append(f"Missing required config key: {'.'.join(key_path)}")
                break
            node = node[part]
    return errors


def hash_config(cfg: dict) -> str:
    """Produce a deterministic SHA-256 hex digest of a config dict.

    Keys are sorted recursively so that insertion order does not matter.

    Args:
        cfg: Configuration dictionary.

    Returns:
        64-character hex string.
    """
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
