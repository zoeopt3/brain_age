"""Release bundle: manifest + zip packaging."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_release_manifest(cfg: dict) -> dict[str, Any]:
    """Build a manifest of files included in the release bundle.

    Returns:
        Dict with file list, hashes, and metadata.
    """
    include_paths = cfg.get("release", {}).get("include_paths", [])
    files: list[dict[str, str]] = []

    for p in include_paths:
        path = Path(p)
        if path.is_file():
            files.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256_short": _file_hash(path),
            })
        elif path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    files.append({
                        "path": str(f),
                        "size_bytes": f.stat().st_size,
                        "sha256_short": _file_hash(f),
                    })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "total_size_bytes": sum(f["size_bytes"] for f in files),
        "files": files,
    }


def build_release_zip(cfg: dict, manifest: dict) -> str:
    """Create a zip bundle containing all release files.

    Args:
        cfg: Stage 6 config.
        manifest: Output of make_release_manifest.

    Returns:
        Path to the created zip file.
    """
    out_dir = Path(cfg.get("release", {}).get("out_dir", "outputs/release"))
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_name = cfg.get("release", {}).get("bundle_name", "release_bundle.zip")
    zip_path = out_dir / bundle_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in manifest.get("files", []):
            fpath = Path(entry["path"])
            if fpath.exists():
                zf.write(fpath, fpath)

        # Include the manifest itself
        manifest_json = json.dumps(manifest, indent=2, default=str)
        zf.writestr("release_manifest.json", manifest_json)

    return str(zip_path)
