"""
common.py
---------
Shared helper functions: dynamic project root resolution, safe YAML loading,
and JSON read/write helpers for metric and artifact tracking.
"""

import json
from pathlib import Path
from typing import Any
import yaml
from src.utils.logger import get_logger

log = get_logger("common")

# Dynamic root: two directories up from src/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_rel_path: str = "config/config.yaml") -> dict[str, Any]:
    """Reads project configuration from a YAML file."""
    config_path = PROJECT_ROOT / config_rel_path

    if not config_path.exists():
        log.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file missing: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_json(data: dict[str, Any], file_path: Path) -> None:
    """Safely saves a dictionary to a formatted JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    log.info(f"Metadata written successfully -> {file_path}")


def load_json(file_path: Path) -> dict[str, Any]:
    """Loads a JSON file into a Python dictionary."""
    if not file_path.exists():
        raise FileNotFoundError(f"Target JSON file does not exist: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)