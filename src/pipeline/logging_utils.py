"""Logging setup and environment introspection."""

import logging
import platform
import sys
from importlib.metadata import version, PackageNotFoundError


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Return a configured logger with console output.

    Args:
        name: Logger name (typically ``__name__``).
        log_level: Level string, e.g. ``"INFO"``, ``"DEBUG"``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger


_PACKAGES_TO_CHECK = [
    "pyyaml",
    "pandas",
    "numpy",
    "mne",
    "scikit-learn",
    "matplotlib",
    "shap",
]


def log_environment_info(logger: logging.Logger) -> None:
    """Log Python version, platform, and key package versions.

    Packages that are not installed are silently skipped.

    Args:
        logger: Logger instance to write to.
    """
    logger.info("Python %s", sys.version.split()[0])
    logger.info("Platform: %s %s", platform.system(), platform.release())

    for pkg in _PACKAGES_TO_CHECK:
        try:
            ver = version(pkg)
            logger.info("  %-14s %s", pkg, ver)
        except PackageNotFoundError:
            pass
