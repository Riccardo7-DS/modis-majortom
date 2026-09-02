"""Tests for pure functions in utils/analysis.py — NDVI computation and
MOD35 cloud-mask decoding on synthetic arrays.
"""
import numpy as np

from modis_majortom.utils.analysis import compute_ndvi, decode_mod35


def test_compute_ndvi_known_values():
    red = np.array([0.1, 0.2, 0.0], dtype=np.float32)
    nir = np.array([0.3, 0.2, 0.0], dtype=np.float32)

    ndvi = compute_ndvi(red, nir)

    assert np.isclose(ndvi[0], 0.5, atol=1e-6)
    assert np.isclose(ndvi[1], 0.0, atol=1e-6)
    assert np.isnan(ndvi[2])  # red + nir == 0 -> NaN, not divide-by-zero


def test_compute_ndvi_masks_fill_values():
    red = np.array([-0.2, 0.1], dtype=np.float32)
    nir = np.array([0.3, 0.3], dtype=np.float32)

    ndvi = compute_ndvi(red, nir, fill_below=-0.05)

    assert np.isnan(ndvi[0])  # red below fill_below threshold -> masked
    assert np.isclose(ndvi[1], (0.3 - 0.1) / (0.3 + 0.1), atol=1e-6)


def test_decode_mod35_cloud_mask():
    # 0=confident cloudy, 1=probably cloudy, 2=probably clear, 3=confident clear, 99=fill
    raw = np.array([0, 1, 2, 3, 99], dtype=np.uint8)

    cloudy = decode_mod35(raw)

    np.testing.assert_array_equal(cloudy, [True, True, False, False, False])
