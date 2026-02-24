"""
Structured Logging System for Brain-Encoders
=============================================

Provides a two-tier logging system inspired by the node-reference-code
pattern: a Rich-powered live dashboard for interactive use and a
simultaneous plain-text file logger for batch/Slurm runs.

Design Principles:
    - No hidden globals (module-level singleton with explicit set/get)
    - Pipe-delimited key=value format for structured log messages
    - Graceful fallback if Rich or pynvml are unavailable
    - Color-coded severity levels for fast visual scanning
    - Simultaneous file + console output for auditability
    - MATLAB correspondence notes preserved for paper verification

Severity Levels:
    info     (cyan)     — routine progress
    ok       (green)    — successful completion
    warn     (yellow)   — recoverable issues
    error    (red)      — failures
    metric   (magenta)  — quantitative results (r-values, accuracy, etc.)
    matlab   (blue)     — MATLAB correspondence notes

Usage::

    from amod_encoder.utils.logging import get_logger, log

    logger = get_logger(__name__)
    logger.info("fit | subject=01 roi=amygdala n_voxels=247")
    log("cv_complete | mean_r=0.049 folds=5", severity="metric")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Severity colour map (matches node-reference-code pattern)
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "info":   "cyan",
    "ok":     "green",
    "warn":   "yellow",
    "error":  "red",
    "metric": "magenta",
    "matlab": "bright_blue",
}

# ---------------------------------------------------------------------------
# Rich availability check (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_console: Optional["Console"] = Console(stderr=True) if HAS_RICH else None
_file_handler: Optional[logging.FileHandler] = None
_configured = False
_log_dir: Optional[Path] = None


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[str | Path] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the global logging system.

    Call once at pipeline start. Sets up Rich console handler + optional
    file handler.  Safe to call multiple times (idempotent).

    Args:
        level:    Log level string (DEBUG, INFO, WARNING, ERROR).
        log_dir:  Directory for log files.  Created if needed.
        log_file: Log filename.  Defaults to ``brain_encoders_<timestamp>.log``.
    """
    global _configured, _file_handler, _log_dir

    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # --- Console handler (Rich or plain) ---
    if HAS_RICH and _console is not None:
        console_handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s — %(message)s")
        )
    root.addHandler(console_handler)

    # --- File handler (always plain text, for Slurm/batch) ---
    if log_dir is not None:
        _log_dir = Path(log_dir)
        _log_dir.mkdir(parents=True, exist_ok=True)
        if log_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"brain_encoders_{ts}.log"
        fh = logging.FileHandler(_log_dir / log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s — %(message)s")
        )
        root.addHandler(fh)
        _file_handler = fh

    _configured = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a structured logger with rich formatting.

    Auto-configures on first call if ``configure_logging()`` hasn't
    been called yet (sensible defaults: INFO level, console only).

    Args:
        name:  Logger name (typically ``__name__``).
        level: Per-logger level override.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    if not _configured:
        configure_logging()

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Convenience functions (module-level, like node-reference-code pattern)
# ---------------------------------------------------------------------------


def log(msg: str, severity: str = "info") -> None:
    """
    Quick-log a message with a severity tag.

    Uses the color map for Rich console output.  Falls back to
    standard logging if Rich is unavailable.

    Args:
        msg:      Pipe-delimited message (e.g. ``"fit | sub=01 roi=amy"``).
        severity: One of info, ok, warn, error, metric, matlab.
    """
    logger = get_logger("brain-encoders")
    colour = SEVERITY_COLORS.get(severity, "white")

    if severity == "error":
        logger.error(msg)
    elif severity == "warn":
        logger.warning(msg)
    else:
        if HAS_RICH:
            logger.info(f"[{colour}]{msg}[/{colour}]")
        else:
            logger.info(msg)


def log_matlab_note(
    logger: logging.Logger,
    script_name: str,
    detail: str,
) -> None:
    """
    Log a note about MATLAB correspondence for paper verification.

    These messages are tagged with severity='matlab' and formatted to
    stand out during review, making it easy to trace which MATLAB script
    each Python operation mirrors.

    Args:
        logger:      Logger instance.
        script_name: Name of the corresponding MATLAB script.
        detail:      Description of what is being matched.
    """
    logger.info("[MATLAB≈%s] %s", script_name, detail)
