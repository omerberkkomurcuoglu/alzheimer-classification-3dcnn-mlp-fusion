from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for execution
    tqdm = None


DEFAULT_TARGET_SHAPE = (128, 128, 128)
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
NIFTI_SUFFIXES = (".nii", ".nii.gz", ".nii.gz.zip")
RUN_PATTERN = re.compile(r"run-(\d+)", re.IGNORECASE)
ANAT_PATTERN = re.compile(r"anat(\d+)", re.IGNORECASE)


@dataclass
class PreprocessResult:
    session_id: str
    source_file: str
    output_file: str
    original_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    resampled_shape: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    final_shape: tuple[int, int, int]
    status: str
    message: str = ""


def iter_session_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset klasoru bulunamadi: {dataset_dir}")
    return sorted(path for path in dataset_dir.iterdir() if path.is_dir())


def has_nifti_suffix(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def parse_run_number(path: Path) -> int:
    match = RUN_PATTERN.search(path.name)
    return int(match.group(1)) if match else 9999


def parse_anat_number(path: Path) -> int:
    for part in path.parts:
        match = ANAT_PATTERN.fullmatch(part)
        if match:
            return int(match.group(1))
    return -1


def candidate_sort_key(session_dir: Path, path: Path) -> tuple[int, int, int, str]:
    relative_path = path.relative_to(session_dir)
    relative_depth = len(relative_path.parts)
    anat_number = parse_anat_number(relative_path)
    is_zip = 1 if path.name.lower().endswith(".zip") else 0
    return (-anat_number, -parse_run_number(path), is_zip, relative_depth, path.name.lower())


def find_t1_candidates(session_dir: Path) -> list[Path]:
    candidates = []
    for path in session_dir.rglob("*"):
        if not path.is_file():
            continue
        if not has_nifti_suffix(path):
            continue
        if "t1w" not in path.name.lower():
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda path: candidate_sort_key(session_dir, path))


def extract_nifti_from_zip(zip_path: Path, cache_dir: Path) -> Path:
    cache_key = hashlib.sha1(str(zip_path).encode("utf-8")).hexdigest()[:12]
    extract_dir = cache_dir / f"{zip_path.stem}_{cache_key}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zip_file:
        members = [
            member
            for member in zip_file.namelist()
            if member.lower().endswith(".nii") or member.lower().endswith(".nii.gz")
        ]
        if not members:
            raise FileNotFoundError(f"Zip icinde NIfTI dosyasi yok: {zip_path}")

        members = sorted(
            members,
            key=lambda member: (
                0 if "t1w" in member.lower() else 1,
                -parse_run_number(Path(member)),
                len(Path(member).parts),
                member.lower(),
            ),
        )
        chosen_member = members[0]
        target_path = extract_dir / Path(chosen_member).name

        if not target_path.exists():
            with zip_file.open(chosen_member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

    return target_path


def resolve_source_file(session_dir: Path, cache_dir: Path) -> Path | None:
    candidates = find_t1_candidates(session_dir)
    if not candidates:
        return None

    chosen = candidates[0]
    if chosen.name.lower().endswith(".zip"):
        return extract_nifti_from_zip(chosen, cache_dir)
    return chosen


def load_canonical_volume(source_file: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(source_file))
    image = nib.as_closest_canonical(image)

    data = image.get_fdata(dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    spacing = tuple(float(value) if float(value) > 0 else 1.0 for value in image.header.get_zooms()[:3])
    return data, spacing


def resample_to_spacing(
    volume: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    order: int = 1,
) -> np.ndarray:
    zoom_factors = tuple(
        float(current_spacing[axis]) / float(target_spacing[axis]) for axis in range(3)
    )
    if all(abs(factor - 1.0) < 1e-3 for factor in zoom_factors):
        return volume.astype(np.float32, copy=False)
    return ndimage.zoom(volume, zoom=zoom_factors, order=order).astype(np.float32, copy=False)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, component_count = ndimage.label(mask)
    if component_count == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_component = int(component_sizes.argmax())
    return labeled == largest_component


def compute_foreground_mask(volume: np.ndarray) -> np.ndarray:
    nonzero_mask = volume > 0
    if not np.any(nonzero_mask):
        nonzero_mask = np.abs(volume) > 1e-6
    if not np.any(nonzero_mask):
        return np.ones_like(volume, dtype=bool)

    foreground_values = volume[nonzero_mask]
    threshold = float(np.percentile(foreground_values, 10))
    threshold = max(threshold, 1e-6)

    mask = volume > threshold
    if not np.any(mask):
        mask = nonzero_mask

    mask = largest_connected_component(mask)
    mask = ndimage.binary_fill_holes(mask)
    if not np.any(mask):
        mask = nonzero_mask
    return mask.astype(bool)


def crop_to_mask(volume: np.ndarray, mask: np.ndarray, margin: int = 4) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        fallback_mask = np.ones_like(volume, dtype=bool)
        return volume.astype(np.float32, copy=False), fallback_mask

    mins = np.maximum(coordinates.min(axis=0) - margin, 0)
    maxs = np.minimum(coordinates.max(axis=0) + margin + 1, volume.shape)
    slices = tuple(slice(int(mins[axis]), int(maxs[axis])) for axis in range(3))
    return (
        volume[slices].astype(np.float32, copy=False),
        mask[slices].astype(bool, copy=False),
    )


def clip_intensity(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = volume[mask]
    if values.size == 0:
        return volume.astype(np.float32, copy=False)

    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        return volume.astype(np.float32, copy=False)
    return np.clip(volume, low, high).astype(np.float32, copy=False)


def normalize_foreground(volume: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = volume[mask]
    normalized = np.zeros_like(volume, dtype=np.float32)
    if values.size == 0:
        return normalized

    mean = float(values.mean())
    std = float(values.std())
    if std < eps:
        normalized[mask] = values - mean
    else:
        normalized[mask] = (values - mean) / std

    normalized = np.clip(normalized, -5.0, 5.0)
    return normalized.astype(np.float32, copy=False)


def center_crop_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    slices = []
    for current_size, target_size in zip(volume.shape, target_shape):
        if current_size <= target_size:
            slices.append(slice(0, current_size))
            continue
        start = (current_size - target_size) // 2
        slices.append(slice(start, start + target_size))
    return volume[tuple(slices)]


def resize_and_pad(volume: np.ndarray, target_shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    current_shape = np.array(volume.shape, dtype=np.float32)
    target = np.array(target_shape, dtype=np.int32)

    scale = float(np.min(target / current_shape))
    if scale <= 0:
        raise ValueError(f"Gecersiz resize olcegi: {scale}")

    resized = ndimage.zoom(volume, zoom=(scale, scale, scale), order=order).astype(np.float32, copy=False)
    resized = center_crop_to_shape(resized, target_shape)

    output = np.zeros(target_shape, dtype=np.float32)
    insert_slices = []
    for axis, target_size in enumerate(target_shape):
        current_size = resized.shape[axis]
        start = (target_size - current_size) // 2
        insert_slices.append(slice(start, start + current_size))
    output[tuple(insert_slices)] = resized
    return output


def preprocess_session(
    source_file: Path,
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float], tuple[int, int, int], tuple[int, int, int]]:
    original_volume, original_spacing = load_canonical_volume(source_file)
    original_shape = tuple(int(value) for value in original_volume.shape)

    resampled = resample_to_spacing(original_volume, original_spacing, target_spacing)
    resampled_shape = tuple(int(value) for value in resampled.shape)

    mask = compute_foreground_mask(resampled)
    cropped_volume, cropped_mask = crop_to_mask(resampled, mask, margin=4)
    cropped_volume = clip_intensity(cropped_volume, cropped_mask)
    normalized = normalize_foreground(cropped_volume, cropped_mask)
    final_volume = resize_and_pad(normalized, target_shape)

    return (
        final_volume,
        original_shape,
        original_spacing,
        resampled_shape,
        tuple(int(value) for value in cropped_volume.shape),
    )


def write_report(report_path: Path, results: list[PreprocessResult]) -> None:
    fieldnames = [
        "session_id",
        "source_file",
        "output_file",
        "original_shape",
        "original_spacing",
        "resampled_shape",
        "cropped_shape",
        "final_shape",
        "status",
        "message",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "session_id": result.session_id,
                    "source_file": result.source_file,
                    "output_file": result.output_file,
                    "original_shape": result.original_shape,
                    "original_spacing": result.original_spacing,
                    "resampled_shape": result.resampled_shape,
                    "cropped_shape": result.cropped_shape,
                    "final_shape": result.final_shape,
                    "status": result.status,
                    "message": result.message,
                }
            )


def progress_iter(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def process_single_session(
    session_dir_str: str,
    target_dir_str: str,
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    cache_root_str: str,
) -> PreprocessResult:
    session_dir = Path(session_dir_str)
    target_dir = Path(target_dir_str)
    cache_dir = Path(cache_root_str)

    session_id = session_dir.name
    output_path = target_dir / f"{session_id}.npy"

    source_file = resolve_source_file(session_dir, cache_dir)
    if source_file is None:
        return PreprocessResult(
            session_id=session_id,
            source_file="",
            output_file=str(output_path),
            original_shape=(0, 0, 0),
            original_spacing=(0.0, 0.0, 0.0),
            resampled_shape=(0, 0, 0),
            cropped_shape=(0, 0, 0),
            final_shape=(0, 0, 0),
            status="missing_t1w",
            message="T1w NIfTI bulunamadi",
        )

    try:
        final_volume, original_shape, original_spacing, resampled_shape, cropped_shape = preprocess_session(
            source_file=source_file,
            target_spacing=target_spacing,
            target_shape=target_shape,
        )
        np.save(output_path, final_volume.astype(np.float32, copy=False))
        return PreprocessResult(
            session_id=session_id,
            source_file=str(source_file),
            output_file=str(output_path),
            original_shape=original_shape,
            original_spacing=original_spacing,
            resampled_shape=resampled_shape,
            cropped_shape=cropped_shape,
            final_shape=tuple(int(value) for value in final_volume.shape),
            status="ok",
        )
    except Exception as exc:  # pragma: no cover - runtime safety path
        return PreprocessResult(
            session_id=session_id,
            source_file=str(source_file),
            output_file=str(output_path),
            original_shape=(0, 0, 0),
            original_spacing=(0.0, 0.0, 0.0),
            resampled_shape=(0, 0, 0),
            cropped_shape=(0, 0, 0),
            final_shape=(0, 0, 0),
            status="error",
            message=str(exc),
        )


def run_pipeline(
    dataset_dir: Path,
    target_dir: Path,
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
    overwrite: bool,
    workers: int,
) -> list[PreprocessResult]:
    dataset_dir = dataset_dir.resolve()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = iter_session_dirs(dataset_dir)
    results: list[PreprocessResult] = []
    pending_session_dirs: list[Path] = []

    for session_dir in session_dirs:
        session_id = session_dir.name
        output_path = target_dir / f"{session_id}.npy"
        if output_path.exists() and not overwrite:
            results.append(
                PreprocessResult(
                    session_id=session_id,
                    source_file="",
                    output_file=str(output_path),
                    original_shape=(0, 0, 0),
                    original_spacing=(0.0, 0.0, 0.0),
                    resampled_shape=(0, 0, 0),
                    cropped_shape=(0, 0, 0),
                    final_shape=target_shape,
                    status="skipped_existing",
                    message="Cikti zaten var",
                )
            )
        else:
            pending_session_dirs.append(session_dir)

    with tempfile.TemporaryDirectory(prefix="mri_zip_cache_", dir=str(target_dir.parent)) as temp_cache:
        if workers <= 1:
            iterator = progress_iter(pending_session_dirs, total=len(pending_session_dirs), desc="MRI preprocessing")
            for session_dir in iterator:
                results.append(
                    process_single_session(
                        session_dir_str=str(session_dir),
                        target_dir_str=str(target_dir),
                        target_spacing=target_spacing,
                        target_shape=target_shape,
                        cache_root_str=temp_cache,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        process_single_session,
                        str(session_dir),
                        str(target_dir),
                        target_spacing,
                        target_shape,
                        temp_cache,
                    )
                    for session_dir in pending_session_dirs
                ]
                for future in progress_iter(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"MRI preprocessing ({workers} workers)",
                ):
                    results.append(future.result())

    results.sort(key=lambda result: result.session_id)
    write_report(target_dir / "preprocessing_report.csv", results)
    return results


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Dataset altindaki session klasorlerinden dogru 3D MRI preprocessing ile 128^3 npy uretir."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "dataset",
        help="Ham session klasorlerinin oldugu ana klasor",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=project_root / "processed_dataset_128",
        help="Numpy ciktilarinin yazilacagi klasor",
    )
    parser.add_argument(
        "--target-size",
        nargs=3,
        type=int,
        default=DEFAULT_TARGET_SHAPE,
        metavar=("D", "H", "W"),
        help="Hedef hacim boyutu",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=DEFAULT_TARGET_SPACING,
        metavar=("SX", "SY", "SZ"),
        help="Canonical uzayda hedef voxel spacing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan .npy ciktilarini yeniden uret",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Session bazinda paralel worker sayisi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        dataset_dir=args.dataset_dir,
        target_dir=args.target_dir,
        target_spacing=tuple(float(value) for value in args.spacing),
        target_shape=tuple(int(value) for value in args.target_size),
        overwrite=bool(args.overwrite),
        workers=max(1, int(args.workers)),
    )

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    print("Preprocessing tamamlandi.")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")
    print(f"Rapor: {args.target_dir / 'preprocessing_report.csv'}")


if __name__ == "__main__":
    main()
