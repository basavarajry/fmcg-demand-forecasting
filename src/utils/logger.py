"""Structured logging setup."""

import logging
import sys
from typing import Optional

try:
    from loguru import logger as _loguru_logger

    _USE_LOGURU = True
except ImportError:
    _USE_LOGURU = False


def get_logger(name: Optional[str] = None, level: str = "INFO"):
    """Return configured logger (loguru if available, else stdlib)."""
    if _USE_LOGURU:
        _loguru_logger.remove()
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        _loguru_logger.add(sys.stderr, format=fmt, level=level, enqueue=True)
        if name:
            return _loguru_logger.bind(module=name)
        return _loguru_logger

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return logging.getLogger(name or "fmcg-forecast")
