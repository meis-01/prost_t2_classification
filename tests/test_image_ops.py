import numpy as np

from prost_t2_classification.image_ops import (
    align_multicoil_phase,
    centered_fft2,
    scale_complex_by_magnitude,
)


def test_centered_fft2_is_orthonormal_and_places_constant_at_center():
    image = np.ones((2, 8, 8), dtype=np.complex64)

    transformed = centered_fft2(image)

    assert transformed.dtype == np.complex64
    assert np.allclose(
        np.sum(np.abs(transformed) ** 2), np.sum(np.abs(image) ** 2), atol=1e-5
    )
    assert np.count_nonzero(np.abs(transformed) > 1e-6) == 2
    assert np.all(np.abs(transformed[:, 4, 4]) > 0)


def test_scale_complex_by_magnitude_uses_robust_percentile():
    image = np.ones((1, 2, 2), dtype=np.complex64)
    image[0, 0, 0] = 100 + 0j

    scaled = scale_complex_by_magnitude(image, percentile=50)

    assert scaled.dtype == np.complex64
    assert np.isclose(np.abs(scaled[0, 0, 1]), 1.0)
    assert np.isclose(np.abs(scaled[0, 0, 0]), 100.0)


def test_align_multicoil_phase_removes_global_and_coil_offsets():
    base = np.ones((2, 4, 4), dtype=np.complex64)
    base[0] *= np.complex64(np.exp(1j * 0.7))
    base[1] *= np.complex64(0.5 * np.exp(1j * 1.1))

    aligned = align_multicoil_phase(base, background_threshold=0)

    assert np.allclose(aligned[0].imag, 0, atol=1e-6)
    assert np.allclose(aligned[1].imag, 0, atol=1e-6)
    assert np.allclose(aligned[0].real, 1, atol=1e-6)
    assert np.allclose(aligned[1].real, 0.5, atol=1e-6)
