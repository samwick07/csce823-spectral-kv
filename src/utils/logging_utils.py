"""Logging and experiment tracking setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    experiment_name: str | None = None,
) -> logging.Logger:
    """Set up logging with console and optional file output.

    Args:
        level: Logging level.
        log_file: Optional file path for log output.
        experiment_name: Optional experiment name for the log header.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if experiment_name:
        logger.info(f"=== {experiment_name} ===")

    return logger
