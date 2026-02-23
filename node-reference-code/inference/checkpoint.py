"""
Checkpoint Discovery and Loading
================================

This module handles automatic checkpoint discovery, validation, and loading.

Checkpoints are expected to follow the naming convention:
    emnist_{variation}_epoch{N}.pt

The module searches for the highest-epoch checkpoint in the specified directory.

Checkpoint Structure
--------------------
Expected checkpoint keys:
- classes: List of class names (characters)
- size: Image resolution
- channels: Block output channels tuple
- unet: UNet state dict
- scheduler: Scheduler configuration dict
- num_class_embeds: Number of class embeddings
- null_class_index: Index for null/unconditional class

"""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class CheckpointInfo:
    """
    Structured information about a loaded checkpoint.
    
    Contains all metadata and state needed to construct the model.
    
    Attributes:
        path: Absolute path to checkpoint file.
        epoch: Training epoch number.
        classes: List of class names.
        size: Image resolution.
        channels: Block output channels.
        num_class_embeds: Number of class embeddings.
        null_class_index: Index for null class.
        unet_state_dict: UNet model weights.
        scheduler_config: Scheduler configuration.
        file_size_bytes: File size in bytes.
        sha256: SHA256 hash of checkpoint file.
    """
    
    path: str
    epoch: int
    classes: List[str]
    size: int
    channels: Tuple[int, ...]
    num_class_embeds: int
    null_class_index: int
    unet_state_dict: Dict[str, Any]
    scheduler_config: Dict[str, Any]
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    
    @property
    def num_classes(self) -> int:
        """Number of classes (excluding null)."""
        return len(self.classes)
    
    def get_label_index(self, character: str) -> int:
        """
        Get the label index for a character.
        
        Args:
            character: Character name (e.g., "k", "A", "3").
            
        Returns:
            Integer label index.
            
        Raises:
            ValueError: If character not in classes.
        """
        if character not in self.classes:
            raise ValueError(
                f"Character '{character}' not in checkpoint classes.\n"
                f"Valid classes: {self.classes}"
            )
        return self.classes.index(character)


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        path: Path to file.
        chunk_size: Read chunk size in bytes.
        
    Returns:
        Hexadecimal hash string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def discover_checkpoint(
    search_dir: str,
    pattern: str = r"emnist_.*_epoch(\d+)\.pt",
    target_epoch: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Discover a checkpoint in a directory.
    
    Searches for checkpoint files matching the pattern.  If *target_epoch* is
    ``None`` (the default), returns the checkpoint with the highest epoch
    number.  Otherwise returns the checkpoint whose epoch matches
    *target_epoch* exactly.
    
    Args:
        search_dir: Directory to search.
        pattern: Regex pattern with epoch capture group.
        target_epoch: If given, select this specific epoch instead of the
            latest one.  Set to ``None`` to keep the original behaviour.
        
    Returns:
        Tuple of (checkpoint_path, epoch_number).
        
    Raises:
        RuntimeError: If no checkpoints found, or if *target_epoch* is
            specified but no matching checkpoint exists.
    """
    if not os.path.isdir(search_dir):
        raise RuntimeError(f"Checkpoint directory does not exist: {search_dir}")
    
    ckpts: List[Tuple[int, str]] = []
    
    for f in os.listdir(search_dir):
        m = re.match(pattern, f)
        if m:
            epoch = int(m.group(1))
            ckpts.append((epoch, f))
    
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {search_dir}")
    
    if target_epoch is not None:
        matches = [(e, fn) for e, fn in ckpts if e == target_epoch]
        if not matches:
            available = sorted(e for e, _ in ckpts)
            raise RuntimeError(
                f"No checkpoint with epoch {target_epoch} in {search_dir}.\n"
                f"Available epochs: {available}"
            )
        epoch, fname = matches[0]
    else:
        ckpts.sort()
        epoch, fname = ckpts[-1]
    
    path = os.path.join(search_dir, fname)
    
    return path, epoch


def load_checkpoint(
    path: str,
    compute_hash: bool = True,
) -> CheckpointInfo:
    """
    Load a checkpoint file and return structured info.
    
    Args:
        path: Path to checkpoint .pt file.
        compute_hash: Whether to compute SHA256 hash.
        
    Returns:
        CheckpointInfo with all checkpoint data.
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        KeyError: If checkpoint missing required keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Validate required keys
    required_keys = ["classes", "size", "channels", "unet", "scheduler", 
                     "num_class_embeds", "null_class_index"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")
    
    # Extract epoch from filename
    match = re.search(r"epoch(\d+)", os.path.basename(path))
    epoch = int(match.group(1)) if match else 0
    
    # Compute file metadata
    file_size = int(os.path.getsize(path))
    sha256 = _sha256_file(path) if compute_hash else None
    
    return CheckpointInfo(
        path=path,
        epoch=epoch,
        classes=list(ckpt["classes"]),
        size=int(ckpt["size"]),
        channels=tuple(ckpt["channels"]),
        num_class_embeds=int(ckpt["num_class_embeds"]),
        null_class_index=int(ckpt["null_class_index"]),
        unet_state_dict=ckpt["unet"],
        scheduler_config=dict(ckpt["scheduler"]),
        file_size_bytes=file_size,
        sha256=sha256,
    )


def validate_character(ckpt_info: CheckpointInfo, character: str) -> int:
    """
    Validate that a character exists in checkpoint and return its label.
    
    Args:
        ckpt_info: Loaded checkpoint info.
        character: Character to validate.
        
    Returns:
        Label index for the character.
        
    Raises:
        ValueError: If character not found.
    """
    return ckpt_info.get_label_index(character)


def discover_and_load(
    search_dir: str,
    compute_hash: bool = True,
    target_epoch: Optional[int] = None,
) -> CheckpointInfo:
    """
    Convenience function to discover and load a checkpoint.
    
    Args:
        search_dir: Directory to search.
        compute_hash: Whether to compute SHA256 hash.
        target_epoch: If given, load this specific epoch instead of the latest.
        
    Returns:
        CheckpointInfo for the selected checkpoint.
    """
    path, _ = discover_checkpoint(search_dir, target_epoch=target_epoch)
    return load_checkpoint(path, compute_hash=compute_hash)
