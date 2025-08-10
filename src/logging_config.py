"""
Logging configuration for the medical triage ML pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_format: str = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file, mode="a")] if log_file else []),
        ],
    )

    logger = logging.getLogger("medical_triage")
    logger.info(f"Logging initialized at level: {log_level}")

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance with the specified name."""
    logger_name = f"medical_triage.{name}" if name else "medical_triage"
    return logging.getLogger(logger_name)
