from __future__ import annotations

import numpy as np


def centered_fft2(image: np.ndarray) -> np.ndarray:
    """Apply an orthonormal centered 2-D FFT over the spatial dimensions."""
    shifted = np.fft.ifftshift(image, axes=(-2, -1))
    transformed = np.fft.fft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(transformed, axes=(-2, -1)).astype(np.complex64)


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
