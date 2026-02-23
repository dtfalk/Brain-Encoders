"""
Logging Utilities
=================

Rich-powered live terminal dashboard for diffusion inference on SLURM.

Features:
    - Live-updating 4-panel layout (log, GPU, progress, video history)
    - GPU utilization with temperature and memory per card
    - Color-coded severity levels (info, ok, warn, error, metric, video)
    - Per-video progress tracking
    - Simultaneous file logging
    - Graceful fallback when Rich/pynvml unavailable

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import os
import time
import threading
from typing import Callable, Optional

# Optional rich imports (graceful fallback)
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Optional pynvml imports (graceful fallback)
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


# =========================================================================
# Severity Colors
# =========================================================================
SEVERITY_COLORS = {
    "info": "cyan",
    "ok": "green",
    "warn": "yellow",
    "error": "red",
    "metric": "magenta",
    "video": "bright_blue",
}


class PrettyLogger:
    """
    Rich terminal dashboard for inference with GPU monitoring.

    Provides a live-updating terminal interface with 4 panels:
    - Scrolling log messages (color-coded by severity)
    - GPU utilization table (util%, temp, memory per card)
    - Video generation progress
    - Recent video completion history

    Falls back to simple file logging if Rich is unavailable.

    Example:
        logger = PrettyLogger("/path/to/log.log")
        logger.start()
        logger.log("Starting generation...")
        logger.set_progress(vid=3, total_vids=105, step=200, total_steps=1000)
        logger.log_video_complete(vid_id=3, time_sec=12.5, seed=1003)
        logger.stop()
    """

    def __init__(
        self,
        log_path: str,
        refresh_hz: int = 4,
        max_messages: int = 200,
        enable_display: bool = True,
    ) -> None:
        self.log_path = log_path
        self.refresh_hz = refresh_hz
        self.max_messages = max_messages
        self.enable_display = enable_display and HAS_RICH

        self.messages: list[str] = []
        self._rich_messages: list[Text] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Progress state
        self._vid: int = 0
        self._total_vids: int = 1
        self._step: int = 0
        self._total_steps: int = 1
        self._character: str = ""

        # Video history rows
        self._video_rows: list[dict] = []
        self._MAX_VIDEO_ROWS = 15

        # File handle
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.f = open(log_path, "w", buffering=1)

        # Rich objects
        if self.enable_display:
            self._console = Console()
            self._layout = self._build_layout()
            self._live: Optional[Live] = None
        else:
            self._console = None
            self._layout = None
            self._live = None

    def _build_layout(self) -> "Layout":
        layout = Layout()
        layout.split_column(
            Layout(name="top", ratio=3),
            Layout(name="bottom", ratio=2),
        )
        layout["top"].split_row(
            Layout(name="log", ratio=3),
            Layout(name="gpu", ratio=1),
        )
        layout["bottom"].split_row(
            Layout(name="progress", ratio=1),
            Layout(name="videos", ratio=2),
        )
        return layout

    # ---- Public API -------------------------------------------------------

    def log(self, msg: str, severity: str = "info") -> None:
        """
        Log a message with optional severity level.

        Args:
            msg: Message to log.
            severity: One of info, ok, warn, error, metric, video.
        """
        ts = time.strftime("%H:%M:%S")
        tag = severity.upper().ljust(6)
        plain = f"[{ts}] [{tag}] {msg}"

        self.f.write(plain + "\n")

        self.messages.append(plain)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        if self.enable_display:
            color = SEVERITY_COLORS.get(severity, "white")
            rich_line = Text(f"[{ts}] ", style="dim")
            rich_line.append(f"[{tag}] ", style=f"bold {color}")
            rich_line.append(msg)
            self._rich_messages.append(rich_line)
            if len(self._rich_messages) > self.max_messages:
                self._rich_messages = self._rich_messages[-self.max_messages:]
        elif not self.enable_display:
            print(plain)

    def set_progress(
        self,
        vid: int = 0,
        total_vids: int = 1,
        step: int = 0,
        total_steps: int = 1,
        character: str = "",
    ) -> None:
        """Update live progress state."""
        self._vid = vid
        self._total_vids = total_vids
        self._step = step
        self._total_steps = total_steps
        if character:
            self._character = character

    def log_video_complete(
        self,
        vid_id: int,
        time_sec: float,
        seed: int = 0,
        latent_l2: float = 0.0,
        gpu_mb: float = 0.0,
    ) -> None:
        """Record a completed video in the history table."""
        row = {
            "vid_id": vid_id,
            "time_sec": time_sec,
            "seed": seed,
            "latent_l2": latent_l2,
            "gpu_mb": gpu_mb,
        }
        self._video_rows.append(row)
        if len(self._video_rows) > self._MAX_VIDEO_ROWS:
            self._video_rows = self._video_rows[-self._MAX_VIDEO_ROWS:]

        self.log(
            f"Video {vid_id:03d} done │ time={time_sec:.1f}s │ seed={seed} │ "
            f"L2={latent_l2:.2f} │ gpu={gpu_mb:.0f}MB",
            severity="video",
        )

    # ---- Rendering --------------------------------------------------------

    def _render_log_panel(self) -> "Panel":
        if self._rich_messages:
            text = Text("\n").join(self._rich_messages[-40:])
        else:
            text = Text("Waiting for events...", style="dim")
        return Panel(text, title="[bold cyan]Inference Log[/]", border_style="cyan",
                     box=box.ROUNDED, padding=(0, 1))

    def _render_gpu_panel(self) -> "Panel":
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True,
                      header_style="bold yellow")
        table.add_column("GPU", justify="center", width=4)
        table.add_column("Name", justify="left", width=12)
        table.add_column("Util", justify="right", width=5)
        table.add_column("Temp", justify="right", width=5)
        table.add_column("Mem", justify="right", width=14)

        if HAS_PYNVML:
            try:
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    name = name.replace("NVIDIA ", "").replace("Tesla ", "")[:12]
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                        temp_str = f"{temp}°C"
                    except Exception:
                        temp_str = "N/A"

                    util_pct = util.gpu
                    util_color = "green" if util_pct > 80 else "yellow" if util_pct > 40 else "red"
                    mem_used_gb = mem.used / 1e9
                    mem_total_gb = mem.total / 1e9
                    mem_pct = (mem.used / mem.total * 100) if mem.total > 0 else 0
                    mem_color = "green" if mem_pct < 80 else "yellow" if mem_pct < 95 else "red"

                    table.add_row(
                        str(i),
                        name,
                        f"[{util_color}]{util_pct}%[/]",
                        temp_str,
                        f"[{mem_color}]{mem_used_gb:.1f}/{mem_total_gb:.1f}GB[/]",
                    )
            except Exception:
                table.add_row("?", "Error", "N/A", "N/A", "N/A")
        else:
            table.add_row("-", "No NVML", "N/A", "N/A", "N/A")

        return Panel(table, title="[bold yellow]GPU Status[/]", border_style="yellow",
                     box=box.ROUNDED, padding=(0, 0))

    def _render_progress_panel(self) -> "Panel":
        lines: list[Text] = []

        # Character
        if self._character:
            char_line = Text(f"  Char: '{self._character}'", style="bold bright_magenta")
            lines.append(char_line)

        # Video progress
        v_pct = (self._vid / self._total_vids * 100) if self._total_vids > 0 else 0
        v_line = Text(f"  Video: {self._vid}/{self._total_vids}  ({v_pct:.0f}%)", style="bold green")
        lines.append(v_line)

        # Step progress
        s_pct = (self._step / self._total_steps * 100) if self._total_steps > 0 else 0
        s_line = Text(f"  Step:  {self._step}/{self._total_steps}  ({s_pct:.0f}%)", style="cyan")
        lines.append(s_line)

        # Visual bar
        bar_width = 30
        filled = int(s_pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        bar_line = Text(f"  [{bar}]", style="dim cyan")
        lines.append(bar_line)

        content = Text("\n").join(lines)
        return Panel(content, title="[bold green]Progress[/]", border_style="green",
                     box=box.ROUNDED, padding=(0, 1))

    def _render_videos_panel(self) -> "Panel":
        table = Table(box=box.SIMPLE, expand=True, show_header=True,
                      header_style="bold bright_blue")
        table.add_column("Vid", justify="right", width=5)
        table.add_column("Time", justify="right", width=7)
        table.add_column("Seed", justify="right", width=8)
        table.add_column("L2", justify="right", width=8)
        table.add_column("GPU MB", justify="right", width=8)

        for row in self._video_rows[-10:]:
            table.add_row(
                str(row["vid_id"]),
                f"{row['time_sec']:.1f}s",
                str(row["seed"]),
                f"{row['latent_l2']:.2f}",
                f"{row['gpu_mb']:.0f}",
            )

        return Panel(table, title="[bold bright_blue]Video History[/]",
                     border_style="bright_blue", box=box.ROUNDED, padding=(0, 0))

    def _render(self) -> "Layout":
        self._layout["log"].update(self._render_log_panel())
        self._layout["gpu"].update(self._render_gpu_panel())
        self._layout["progress"].update(self._render_progress_panel())
        self._layout["videos"].update(self._render_videos_panel())
        return self._layout

    # ---- Thread management ------------------------------------------------

    def _run(self) -> None:
        with Live(self._render(), refresh_per_second=self.refresh_hz,
                  console=self._console, screen=True) as live:
            self._live = live
            while self.running:
                try:
                    live.update(self._render())
                except Exception:
                    pass
                time.sleep(1.0 / self.refresh_hz)

    def start(self) -> None:
        """Start the logger (begins background display thread)."""
        self.running = True

        if self.enable_display:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the logger (ends background thread and closes file)."""
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None

        try:
            self.f.close()
        except Exception:
            pass


class SimpleLogger:
    """
    Simple file-based logger without rich display.

    Minimal logger for environments without Rich or when simple
    output is preferred.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.f = open(log_path, "w", buffering=1)

    def log(self, msg: str, severity: str = "info") -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        tag = severity.upper().ljust(6)
        line = f"[{ts}] [{tag}] {msg}"
        self.f.write(line + "\n")
        print(line)

    def set_progress(self, **kwargs) -> None:
        pass

    def log_video_complete(self, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.f.close()


# Global logger instance
_logger: Optional[PrettyLogger] = None


def get_logger() -> Optional[PrettyLogger]:
    """Get the global logger instance."""
    return _logger


def set_logger(logger: PrettyLogger) -> None:
    """Set the global logger instance."""
    global _logger
    _logger = logger


def log(msg: str, severity: str = "info") -> None:
    """
    Log a message to the global logger.

    Falls back to print if no logger is set.
    """
    if _logger is not None:
        _logger.log(msg, severity=severity)
    else:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")


def create_logger(
    log_path: str,
    use_rich: bool = True,
) -> PrettyLogger:
    """
    Create and register a logger.

    Args:
        log_path: Path to log file.
        use_rich: Whether to use rich display (if available).

    Returns:
        Logger instance.
    """
    if use_rich and HAS_RICH:
        logger = PrettyLogger(log_path)
    else:
        logger = SimpleLogger(log_path)

    set_logger(logger)
    return logger
