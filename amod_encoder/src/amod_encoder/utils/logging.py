"""
Structured logging utilities for the AMOD encoder pipeline.

This module corresponds to AMOD script(s): (none — MATLAB uses disp())
Key matched choices:
  - Structured logging with rich formatting for readability
  - Log level settable from config or environment
Assumptions / deviations:
  - MATLAB uses bare disp() statements; we use Python logging with structured output
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_configured = False


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a structured logger with rich formatting.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : str, optional
        Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _configured
    if not _configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=_console, rich_tracebacks=True)],
        )
        _configured = True

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log_matlab_note(logger: logging.Logger, script_name: str, detail: str) -> None:
    """Log a note about MATLAB correspondence.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use.
    script_name : str
        Name of the corresponding MATLAB script.
    detail : str
        Description of what is being matched.
    """
    logger.info("[MATLAB≈%s] %s", script_name, detail)
