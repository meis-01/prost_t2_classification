import numpy as np

from prost_t2_classification.image_ops import (
    center_crop_last2,
    middle_acquisition_index,
    pad_coil_axis,
    top_energy_coils,
)


def test_middle_acquisition_index_uses_center():
    assert middle_acquisition_index(3) == 1
    assert middle_acquisition_index(4) == 2


def test_top_energy_coils_descending():
    energy = np.array([2.0, 10.0, 4.0, 1.0])
    assert top_energy_coils(energy, max_coils=3).tolist() == [1, 2, 0]


def test_center_crop_last2_and_pad_coils():
    image = np.zeros((2, 6, 6), dtype=np.complex64)
    cropped = center_crop_last2(image, 4)
    assert cropped.shape == (2, 4, 4)
    padded = pad_coil_axis(cropped, 5)
    assert padded.shape == (5, 4, 4)
