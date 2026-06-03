#!/usr/bin/env python
"""
Utility functions for loading pretrained FOMO25 model weights.

This module provides convenient functions to:
1. Load pretrained checkpoint from the default location
2. Create models with pretrained weights loaded

Usage:
    from utils.load_weights import load_pretrained_checkpoint, get_pretrained_path
    
    # Get the path to pretrained weights
    checkpoint_path = get_pretrained_path()
    
    # Load checkpoint
    checkpoint = load_pretrained_checkpoint()
    
    # Create model and load weights
    model = SelfSupervisedModelCrossPatch(...)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
"""

import os
import torch
from pathlib import Path


# Default location for pretrained weights (at project root/weights/)
# Path: src/utils/load_weights.py -> src/utils/ -> src/ -> project_root/ -> weights/
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
DEFAULT_CHECKPOINT_NAME = "fomo25_mmunetvae_pretrained.ckpt"

# Environment variable for override
ENV_WEIGHTS_PATH = "FOMO25_WEIGHTS_PATH"


def get_weights_dir() -> Path:
    """Get the directory containing pretrained weights."""
    # Check environment variable first
    env_path = os.environ.get(ENV_WEIGHTS_PATH)
    if env_path:
        return Path(env_path).parent
    return WEIGHTS_DIR


def get_pretrained_path(checkpoint_name: str = None) -> Path:
    """
    Get the full path to the pretrained checkpoint.
    
    Args:
        checkpoint_name: Optional checkpoint filename. Defaults to the pretrained model.
        
    Returns:
        Path to the checkpoint file.
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
    """
    # Check environment variable first (takes precedence)
    env_path = os.environ.get(ENV_WEIGHTS_PATH)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    if checkpoint_name is None:
        checkpoint_name = DEFAULT_CHECKPOINT_NAME
    
    checkpoint_path = WEIGHTS_DIR / checkpoint_name
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at: {checkpoint_path}\n"
            f"Please ensure the weights are downloaded to: {WEIGHTS_DIR}\n"
            f"Or set the {ENV_WEIGHTS_PATH} environment variable."
        )
    
    return checkpoint_path


def load_pretrained_checkpoint(
    checkpoint_name: str = None,
    map_location: str = "cpu",
) -> dict:
    """
    Load the pretrained checkpoint.
    
    Args:
        checkpoint_name: Optional checkpoint filename. Defaults to the pretrained model.
        map_location: Device to load the checkpoint to. Default is 'cpu'.
        
    Returns:
        Checkpoint dictionary containing 'state_dict', 'hyper_parameters', etc.
    """
    checkpoint_path = get_pretrained_path(checkpoint_name)
    print(f"Loading pretrained checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Global step: {checkpoint.get('global_step', 'N/A')}")
    
    return checkpoint


def get_pretrained_config(checkpoint_name: str = None) -> dict:
    """
    Get the configuration/hyperparameters from the pretrained checkpoint.
    
    Args:
        checkpoint_name: Optional checkpoint filename.
        
    Returns:
        Configuration dictionary with model hyperparameters.
    """
    checkpoint = load_pretrained_checkpoint(checkpoint_name)
    return checkpoint.get('hyper_parameters', {})


def list_available_checkpoints() -> list:
    """
    List all available checkpoint files in the weights directory.
    
    Returns:
        List of checkpoint filenames.
    """
    if not WEIGHTS_DIR.exists():
        return []
    
    return [f.name for f in WEIGHTS_DIR.glob("*.ckpt")]
