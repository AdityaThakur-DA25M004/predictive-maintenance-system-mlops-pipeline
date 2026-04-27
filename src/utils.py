"""
Utility functions for the Predictive Maintenance System.
Provides logging configuration, config loading, and shared helpers.

Local vs Docker config selection
--------------------------------
Two environment variables let you point this module at the right config
files for the environment you're running in:

  * ``DOTENV_PATH``     — path to the .env file to load
                          (default: ``.env``, which is what Docker uses)
  * ``PM_CONFIG_PATH``  — path to the YAML config file
                          (default: ``configs/config.yaml``, again Docker-flavoured)

In Docker, neither variable is set and defaults kick in. When running
locally, the ``scripts/activate-local.ps1`` helper sets them to point at
``.env.local`` and ``configs/config.local.yaml`` respectively. Nothing
breaks if both are absent — the module falls back to existing behaviour.
"""

import os
import logging
import yaml
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


# ---------------------------------------------------------------------------
# .env loading (idempotent — multiple imports won't re-load)
# ---------------------------------------------------------------------------
_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """
    Load the appropriate .env file exactly once per process.

    Honours the ``DOTENV_PATH`` env var if set; otherwise falls back to
    ``.env`` in the project root. Silent no-op if python-dotenv isn't
    installed or the file doesn't exist — matches existing behaviour
    where missing .env doesn't crash the app.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_LOADED = True
        return

    dotenv_path = os.environ.get(
        "DOTENV_PATH",
        str(PROJECT_ROOT / ".env"),
    )
    if Path(dotenv_path).exists():
        load_dotenv(dotenv_path, override=False)
    _DOTENV_LOADED = True


# Load on import so any module using utils.py gets env vars populated.
_load_dotenv_once()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """
    Load YAML configuration from disk.

    Resolution order:
      1. Explicit ``config_path`` argument (highest priority)
      2. ``PM_CONFIG_PATH`` env var (set by activate-local.ps1)
      3. ``configs/config.yaml`` in the project root (Docker default)

    Args:
        config_path: Optional override path.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
    """
    if config_path is None:
        config_path = os.environ.get(
            "PM_CONFIG_PATH",
            str(PROJECT_ROOT / "configs" / "config.yaml"),
        )
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def save_json(data: dict, filepath: str) -> None:
    """Save a dictionary as pretty-printed JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> dict:
    """Load and return a JSON file as a dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p