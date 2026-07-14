from __future__ import annotations

import numpy as np


def middle_acquisition_index(num_acquisitions: int) -> int:
    if num_acquisitions <= 0:
        raise ValueError("num_acquisitions must be positive.")
    return num_acquisitions // 2


def middle_coil_index(num_coils: int) -> int:
    if num_coils <= 0:
        raise ValueError("num_coils must be positive.")
    return num_coils // 2


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


def standardize_real(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(image, axis=(-2, -1), keepdims=True)
    std = np.std(image, axis=(-2, -1), keepdims=True)
    return ((image - mean) / (std + eps)).astype(np.float32)


def scale_complex_by_magnitude(image: np.ndarray, eps: float = 1e-6, percentile: float = 99.0) -> np.ndarray:
    magnitude = np.abs(image)
    scale = np.percentile(magnitude, percentile, axis=(-2, -1), keepdims=True)
    return (image / (scale + eps)).astype(np.complex64)
