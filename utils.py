"""Utility functions for file saving and logging setup."""

import logging
import os
from datetime import datetime


def setup_logging(log_file: str = "meeting_ai.log", level: int = logging.INFO):
    """Configure logging to both console and file."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def save_text(content: str, filepath: str) -> str:
    """Save text content to a file. Returns the absolute path."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(filepath)


def generate_output_dir() -> str:
    """Create a timestamped output directory and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"meeting_{timestamp}"
    os.makedirs(dirname, exist_ok=True)
    return dirname


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
