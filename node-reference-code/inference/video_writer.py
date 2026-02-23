"""
Video Writer Abstraction
========================

This module provides a clean abstraction for video writing via ffmpeg.

VideoWriter encapsulates ffmpeg subprocess management, allowing the
generation loop to remain focused on diffusion logic.

Design:
- Context manager for safe resource cleanup
- Single grayscale output stream
- Configurable codec, fps, and resolution

"""

from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np


class VideoWriter:
    """
    FFmpeg-based video writer for diffusion frame output.
    
    Manages a grayscale video stream via an ffmpeg subprocess.
    Frames are written as raw bytes to the stdin pipe.
    
    Example:
        with VideoWriter(width=224, height=224, fps=60) as writer:
            writer.set_output_paths(video_path="/path/video.mp4")
            writer.open()
            for frame in frames:
                writer.write_frame(frame)
    
    Attributes:
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        codec: Video codec (default: libx264).
        pixel_format: Output pixel format (default: yuv420p).
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 60,
        codec: str = "libx264",
        pixel_format: str = "yuv420p",
    ) -> None:
        """
        Initialize video writer parameters.
        
        Args:
            width: Output video width.
            height: Output video height.
            fps: Frames per second.
            codec: Video codec.
            pixel_format: Output pixel format.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.pixel_format = pixel_format
        
        self._video_path: Optional[str] = None
        self._video_proc: Optional[subprocess.Popen] = None
        self._is_open = False
    
    def set_output_paths(
        self,
        video_path: Optional[str] = None,
    ) -> 'VideoWriter':
        """
        Set output file path.
        
        Args:
            video_path: Path for output video (None to skip).
            
        Returns:
            Self for chaining.
        """
        self._video_path = video_path
        return self
    
    def open(self) -> 'VideoWriter':
        """
        Open ffmpeg subprocess for writing.
        
        Returns:
            Self for chaining.
        """
        if self._is_open:
            return self
        
        W, H = self.width, self.height
        
        if self._video_path:
            cmd = (
                f"ffmpeg -y -f rawvideo -pix_fmt gray -s {W}x{H} "
                f"-r {self.fps} -i - -c:v {self.codec} -pix_fmt {self.pixel_format} "
                f"{self._video_path}"
            )
            self._video_proc = subprocess.Popen(
                cmd.split(),
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        
        self._is_open = True
        return self
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a grayscale frame.
        
        Args:
            frame: Grayscale array of shape (H, W) and dtype uint8.
        """
        if self._video_proc is not None and self._video_proc.stdin is not None:
            self._video_proc.stdin.write(frame.tobytes())
    
    def close(self) -> None:
        """
        Close ffmpeg subprocess and finalize video.
        """
        if not self._is_open:
            return
        
        if self._video_proc is not None:
            if self._video_proc.stdin is not None:
                self._video_proc.stdin.close()
            self._video_proc.wait()
            self._video_proc = None
        
        self._is_open = False
    
    def __enter__(self) -> 'VideoWriter':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()
    
    @property
    def is_open(self) -> bool:
        """Check if writer is currently open."""
        return self._is_open
