"""Loguru configuration from settings: a readable sink by default, JSON on demand.

`log_format="json"` emits one JSON object per line (ready for Loki/Promtail or
any log shipper); anything else uses a compact human format. Idempotent: removes
the default handler and installs exactly one sink.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.config import Settings

_HUMAN_FORMAT = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {name}:{function} - {message}"


def configure_logging(settings: Settings) -> None:
    """Install a single loguru sink driven by settings.log_level / log_format."""
    logger.remove()
    serialize = settings.log_format.lower() == "json"
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format="{message}" if serialize else _HUMAN_FORMAT,
        serialize=serialize,
        backtrace=False,
        diagnose=False,
    )
