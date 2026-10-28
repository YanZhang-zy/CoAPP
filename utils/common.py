from collections.abc import Mapping
from pathlib import Path
import random

import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_input(images):
    shape = (1, 3, 1, 1) if images.ndim == 4 else (3, 1, 1)
    mean = images.new_tensor(CLIP_MEAN).view(shape)
    std = images.new_tensor(CLIP_STD).view(shape)
    return (images - mean) / std


def resolve_device(name="auto"):
    if str(name) == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available. Use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {device}")
    return device


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(checkpoint, key=None):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a checkpoint dictionary.")
    state = checkpoint
    if key is not None:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint has no '{key}' entry.")
        state = checkpoint[key]
    else:
        for candidate in ("model", "state_dict", "model_state_dict", "params_ema", "params"):
            if candidate in state and isinstance(state[candidate], Mapping):
                state = state[candidate]
                break
    if not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("No model state_dict found in the checkpoint.")
    state = dict(state)
    for prefix in ("module.", "_orig_mod."):
        while state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_paths(folder, recursive=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image directory not found: {folder}")
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
