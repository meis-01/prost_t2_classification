from __future__ import annotations

import numpy as np


def middle_acquisition_index(num_acquisitions: int) -> int:
    if num_acquisitions <= 0:
        raise ValueError("num_acquisitions must be positive.")
    return num_acquisitions // 2


def top_energy_coils(energy: np.ndarray, *, max_coils: int = 5) -> np.ndarray:
    if energy.ndim != 1:
        raise ValueError("energy must be a 1D array.")
    if max_coils <= 0:
        raise ValueError("max_coils must be positive.")
    count = min(max_coils, energy.shape[0])
    return np.argsort(energy)[::-1][:count].astype(np.int64)


def center_crop_last2(array: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None:
        return array
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")

    height, width = array.shape[-2:]
    if height < crop_size or width < crop_size:
        return center_pad_last2(array, crop_size)

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return array[..., top : top + crop_size, left : left + crop_size]


def center_pad_last2(array: np.ndarray, target_size: int) -> np.ndarray:
    height, width = array.shape[-2:]
    pad_height = max(target_size - height, 0)
    pad_width = max(target_size - width, 0)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_widths = [(0, 0)] * array.ndim
    pad_widths[-2] = (pad_top, pad_bottom)
    pad_widths[-1] = (pad_left, pad_right)
    return np.pad(array, pad_widths, mode="constant")


def pad_coil_axis(array: np.ndarray, target_coils: int) -> np.ndarray:
    if array.shape[0] > target_coils:
        return array[:target_coils]
    if array.shape[0] == target_coils:
        return array
    pad_shape = (target_coils - array.shape[0],) + array.shape[1:]
    padding = np.zeros(pad_shape, dtype=array.dtype)
    return np.concatenate([array, padding], axis=0)


def align_multicoil_phase(
    image: np.ndarray,
    *,
    eps: float = 1e-6,
    background_threshold: float = 0.02,
) -> np.ndarray:
    """Remove arbitrary coil offsets while preserving local relative phase."""
    if image.ndim != 3 or image.shape[0] <= 0:
        raise ValueError("image must have shape (coils, height, width).")
    if background_threshold < 0:
        raise ValueError("background_threshold must be non-negative.")

    aligned = image.astype(np.complex64, copy=True)
    rss = np.sqrt(np.sum(np.abs(aligned) ** 2, axis=0))
    signal_scale = float(np.percentile(rss, 99.0))
    if signal_scale <= eps:
        return aligned

    support = rss >= background_threshold * signal_scale
    reference = aligned[0]
    anchor = np.sum(reference[support] * np.abs(reference[support]), dtype=np.complex128)
    if np.abs(anchor) > eps:
        aligned *= np.complex64(np.exp(-1j * np.angle(anchor)))

    reference = aligned[0]
    for coil_index in range(1, aligned.shape[0]):
        cross_phase = np.sum(
            aligned[coil_index, support] * np.conj(reference[support]),
            dtype=np.complex128,
        )
        if np.abs(cross_phase) > eps:
            aligned[coil_index] *= np.complex64(np.exp(-1j * np.angle(cross_phase)))

    aligned[:, ~support] = 0
    return aligned


def scale_complex_by_magnitude(
    image: np.ndarray,
    eps: float = 1e-6,
    percentile: float = 99.0,
    *,
    shared_scale: bool = True,
) -> np.ndarray:
    magnitude = np.abs(image)
    axes = None if shared_scale else (-2, -1)
    scale = np.percentile(magnitude, percentile, axis=axes, keepdims=axes is not None)
    return (image / (scale + eps)).astype(np.complex64)
