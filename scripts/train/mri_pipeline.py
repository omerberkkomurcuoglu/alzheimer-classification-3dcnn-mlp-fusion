"""
Shared MRI preprocessing helpers for the project.

The MRI volumes in this repository are already spatially resampled to 128^3.
What changes most across subjects is intensity distribution, not geometry.
These helpers keep train/test/fusion scripts on the same foreground-aware
normalization path.
"""

from __future__ import annotations

import os

import numpy as np


DEFAULT_MRI_SHAPE = (128, 128, 128)
LABEL_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}


def filter_existing_mri_rows(root_dir, data_csv):
    valid_indices = []
    for idx, row in data_csv.iterrows():
        session_id = row["path"]
        if os.path.exists(os.path.join(root_dir, f"{session_id}.npy")):
            valid_indices.append(idx)
    return data_csv.iloc[valid_indices].reset_index(drop=True)


def coerce_memory_label(value) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0.0
    return LABEL_MAP.get(numeric_value, 0)


def sanitize_mri_volume(data: np.ndarray, clip_range=None) -> np.ndarray:
    volume = np.asarray(data, dtype=np.float32)
    if clip_range is None:
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        volume = np.nan_to_num(volume, nan=0.0, posinf=clip_range[1], neginf=clip_range[0])
    if clip_range is not None:
        volume = np.clip(volume, clip_range[0], clip_range[1])
    return volume.astype(np.float32, copy=False)


def normalize_mri_volume(
    data: np.ndarray,
    foreground_threshold: float = 1e-6,
    eps: float = 1e-6,
    clip_std: float | None = 5.0,
) -> np.ndarray:
    volume = np.asarray(data, dtype=np.float32)
    foreground = np.abs(volume) > foreground_threshold

    if not np.any(foreground):
        return np.zeros_like(volume, dtype=np.float32)

    normalized = np.zeros_like(volume, dtype=np.float32)
    foreground_values = volume[foreground]
    mean = float(foreground_values.mean())
    std = float(foreground_values.std())

    if std < eps:
        normalized[foreground] = foreground_values - mean
    else:
        normalized[foreground] = (foreground_values - mean) / std

    if clip_std is not None:
        normalized = np.clip(normalized, -clip_std, clip_std)

    return normalized.astype(np.float32, copy=False)


def load_sanitized_mri_volume(file_path: str, fallback_shape=DEFAULT_MRI_SHAPE) -> np.ndarray:
    try:
        data = np.load(file_path).astype(np.float32)
    except Exception:
        data = np.zeros(fallback_shape, dtype=np.float32)
    return sanitize_mri_volume(data)


def load_normalized_mri_volume(file_path: str, fallback_shape=DEFAULT_MRI_SHAPE) -> np.ndarray:
    volume = load_sanitized_mri_volume(file_path, fallback_shape=fallback_shape)
    return normalize_mri_volume(volume)
