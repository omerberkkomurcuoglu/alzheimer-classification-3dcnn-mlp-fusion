"""
Shared MRI preprocessing helpers for the project.

The default dataset used by train/test scripts is the output of
`scripts/preprocessing/10.1-preprocess_dataset_128.py`. Those `.npy` files are
already resampled, cropped, foreground-normalized, clipped, and padded to 128^3.

For those preprocessed tensors, the correct load path is sanitize-only.
Foreground normalization is kept as a separate helper for legacy callers that
still need it.
"""

from __future__ import annotations

import os

import numpy as np


DEFAULT_MRI_SHAPE = (128, 128, 128)
DEFAULT_AUGMENT_CLIP_RANGE = (-5.0, 5.0)
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


def load_preprocessed_mri_volume(file_path: str, fallback_shape=DEFAULT_MRI_SHAPE) -> np.ndarray:
    """
    Load a cached `.npy` produced by the offline 10.1 preprocessing pipeline.

    Those files are already normalized, so runtime should only sanitize
    NaN/Inf values instead of applying another foreground z-score pass.
    """
    return load_sanitized_mri_volume(file_path, fallback_shape=fallback_shape)


def load_normalized_mri_volume(file_path: str, fallback_shape=DEFAULT_MRI_SHAPE) -> np.ndarray:
    volume = load_sanitized_mri_volume(file_path, fallback_shape=fallback_shape)
    return normalize_mri_volume(volume)


def sanitize_augmented_mri_volume(
    data: np.ndarray,
    clip_range=DEFAULT_AUGMENT_CLIP_RANGE,
) -> np.ndarray:
    """
    Post-augmentation cleanup used by the 14.3 training pipeline.

    MONAI transforms can introduce NaN/Inf values or push normalized
    intensities outside the expected range. Keep the tensor float32 and
    clamp it back to the same foreground-normalized range used elsewhere.
    """
    return sanitize_mri_volume(data, clip_range=clip_range)
