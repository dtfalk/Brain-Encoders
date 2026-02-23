"""
Pretty Logger
=============

Rich-powered live terminal dashboard for DDPM training on SLURM.

Features:
    - Live-updating layout with scrolling log pane
    - GPU utilization table (all 4× L40S cards)
    - Real-time training progress bar
    - Epoch summary table
    - Color-coded severity levels
    - Simultaneous file logging
    - Graceful fallback when Rich/pynvml unavailable

Usage:
    from pretty_logger import PrettyLogger

    logger = PrettyLogger("logs/train.log")
    logger.start()
    logger.log("Starting epoch 0...")
    logger.set_progress(epoch=0, batch=10, total_batches=500)
    logger.log_epoch_summary(epoch=0, loss=0.05, lr=1e-4, sps=3200, dt=45.3)
    logger.stop()

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import os
import time
import threading
from typing import Optional

# ---- Rich (optional) -----------------------------------------------------
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.progress_bar import ProgressBar
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---- pynvml (optional) ---------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


# =========================================================================
# Severity Colors
# =========================================================================
SEVERITY_COLORS = {
    "info": "cyan",
    "ok": "green",
    "warn": "yellow",
    "error": "red",
    "metric": "magenta",
    "epoch": "bright_blue",
}


class PrettyLogger:
    """
    Live terminal dashboard + file logger for distributed training.

    Parameters
    ----------
    log_path : str
        Path to plain-text log file.
    refresh_hz : int
        Rich Live display refresh rate.
    max_messages : int
        Number of log lines to keep in the scrollback buffer.
    enable_display : bool
        Set False to disable Rich Live entirely (file-only logging).
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
        self._epoch: int = 0
        self._batch: int = 0
        self._total_batches: int = 1
        self._total_epochs: int = 1
        self._current_loss: float = 0.0

        # Epoch summary rows (most recent N)
        self._epoch_rows: list[dict] = []
        self._MAX_EPOCH_ROWS = 20

        # File handle
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._f = open(log_path, "w", buffering=1)

        # Rich objects
        if self.enable_display:
            self._console = Console()
            self._layout = self._build_layout()
            self._live: Optional[Live] = None
        else:
            self._console = None
            self._layout = None
            self._live = None

    # ---- Layout -----------------------------------------------------------

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
            Layout(name="epochs", ratio=2),
        )
        return layout

    # ---- Public API -------------------------------------------------------

    def log(self, msg: str, severity: str = "info") -> None:
        """
        Log a message.

        Parameters
        ----------
        msg : str
            Message body.
        severity : str
            One of info, ok, warn, error, metric, epoch.
        """
        ts = time.strftime("%H:%M:%S")
        tag = severity.upper().ljust(6)
        plain = f"[{ts}] [{tag}] {msg}"

        # File
        self._f.write(plain + "\n")

        # Buffer for display
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
        epoch: int = 0,
        batch: int = 0,
        total_batches: int = 1,
        total_epochs: int = 1,
        current_loss: float = 0.0,
    ) -> None:
        """Update live progress state."""
        self._epoch = epoch
        self._batch = batch
        self._total_batches = total_batches
        self._total_epochs = total_epochs
        self._current_loss = current_loss

    def log_epoch_summary(
        self,
        epoch: int,
        loss: float,
        lr: float,
        sps: float,
        dt: float,
        grad_norm: float = 0.0,
        gpu_peak_mb: float = 0.0,
        ema_delta: float = 0.0,
    ) -> None:
        """
        Record an epoch summary row for the dashboard table.
        Also emits a log line.
        """
        row = {
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
            "sps": sps,
            "dt": dt,
            "grad_norm": grad_norm,
            "gpu_peak_mb": gpu_peak_mb,
            "ema_delta": ema_delta,
        }
        self._epoch_rows.append(row)
        if len(self._epoch_rows) > self._MAX_EPOCH_ROWS:
            self._epoch_rows = self._epoch_rows[-self._MAX_EPOCH_ROWS:]

        self.log(
            f"Epoch {epoch:03d} │ loss={loss:.6f} │ lr={lr:.2e} │ "
            f"sps={sps:.0f} │ dt={dt:.1f}s │ grad={grad_norm:.4f} │ "
            f"peak_gpu={gpu_peak_mb:.0f}MB",
            severity="epoch",
        )

    # ---- Rendering --------------------------------------------------------

    def _render_log_panel(self) -> "Panel":
        if self._rich_messages:
            text = Text("\n").join(self._rich_messages[-40:])
        else:
            text = Text("Waiting for events...", style="dim")
        return Panel(text, title="[bold cyan]Training Log[/]", border_style="cyan",
                     box=box.ROUNDED, padding=(0, 1))

    def _render_gpu_panel(self) -> "Panel":
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True,
                      header_style="bold yellow")
        table.add_column("GPU", justify="center", width=4)
        table.add_column("Name", justify="left", width=12)
        table.add_column("Util", justify="right", width=5)
        table.add_column("Temp", justify="right", width=5)
        table.add_column("Mem", justify="right", width=14)

        if HAS_NVML:
            try:
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    # Shorten name
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

        # Epoch progress
        ep_pct = (self._epoch / self._total_epochs * 100) if self._total_epochs > 0 else 0
        ep_line = Text(f"  Epoch: {self._epoch}/{self._total_epochs}  ({ep_pct:.0f}%)", style="bold green")
        lines.append(ep_line)

        # Batch progress
        b_pct = (self._batch / self._total_batches * 100) if self._total_batches > 0 else 0
        b_line = Text(f"  Batch: {self._batch}/{self._total_batches}  ({b_pct:.0f}%)", style="cyan")
        lines.append(b_line)

        # Current loss
        loss_line = Text(f"  Loss:  {self._current_loss:.6f}", style="magenta")
        lines.append(loss_line)

        # Visual bar
        bar_width = 30
        filled = int(b_pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        bar_line = Text(f"  [{bar}]", style="dim cyan")
        lines.append(bar_line)

        content = Text("\n").join(lines)
        return Panel(content, title="[bold green]Progress[/]", border_style="green",
                     box=box.ROUNDED, padding=(0, 1))

    def _render_epochs_panel(self) -> "Panel":
        table = Table(box=box.SIMPLE, expand=True, show_header=True,
                      header_style="bold bright_blue")
        table.add_column("Ep", justify="right", width=4)
        table.add_column("Loss", justify="right", width=10)
        table.add_column("LR", justify="right", width=10)
        table.add_column("Samp/s", justify="right", width=8)
        table.add_column("Time", justify="right", width=7)
        table.add_column("Grad", justify="right", width=8)
        table.add_column("GPU MB", justify="right", width=8)

        for row in self._epoch_rows[-10:]:
            table.add_row(
                str(row["epoch"]),
                f"{row['loss']:.6f}",
                f"{row['lr']:.2e}",
                f"{row['sps']:.0f}",
                f"{row['dt']:.1f}s",
                f"{row['grad_norm']:.4f}",
                f"{row['gpu_peak_mb']:.0f}",
            )

        return Panel(table, title="[bold bright_blue]Epoch History[/]",
                     border_style="bright_blue", box=box.ROUNDED, padding=(0, 0))

    def _render(self) -> "Layout":
        self._layout["log"].update(self._render_log_panel())
        self._layout["gpu"].update(self._render_gpu_panel())
        self._layout["progress"].update(self._render_progress_panel())
        self._layout["epochs"].update(self._render_epochs_panel())
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
        """Start the background display thread."""
        self.running = True
        if self.enable_display:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop display and close log file."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "PrettyLogger":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
