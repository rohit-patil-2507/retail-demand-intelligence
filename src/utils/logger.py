"""
src/utils/logger.py
Centralized logging utility for the Retail Demand Forecasting platform.
Supports both direct `logger` imports and per-module `get_logger(name)` calls.
"""

import logging
from pathlib import Path
import sys

# Dynamic root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "running_pipeline.log"

# Base configuration for standard output + persistent log file
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)


def get_logger(module_name: str = "RetailForecasting") -> logging.Logger:
    """Returns a module-specific logger instance for granular tracking."""
    return logging.getLogger(module_name)


# Backward-compatible global logger object
logger = get_logger("RetailForecastingLogger")