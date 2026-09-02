"""Tests for CalculationsMajorTom (transform/majortom.py) — pure MODIS
sinusoidal-projection math: lat/lon <-> sinusoidal <-> tile (h, v) <-> pixel.
"""
import math

from modis_majortom.transform.majortom import CalculationsMajorTom


def test_pixel_size_from_nominal_resolution():
    """Actual MODIS pixel size derives from pixels-per-tile, not the nominal label."""
    calc_250 = CalculationsMajorTom(pixel_size=250)
    calc_500 = CalculationsMajorTom(pixel_size=500)
    calc_1000 = CalculationsMajorTom(pixel_size=1000)

    assert math.isclose(calc_250.PIXEL_SIZE, 1111950 / 4800, rel_tol=1e-9)
    assert math.isclose(calc_500.PIXEL_SIZE, 1111950 / 2400, rel_tol=1e-9)
    assert math.isclose(calc_1000.PIXEL_SIZE, 1111950 / 1200, rel_tol=1e-9)


def test_latlon_to_sinu_at_origin():
    """(lat=0, lon=0) maps to the sinusoidal projection origin (0, 0)."""
    calc = CalculationsMajorTom()
    x, y = calc.latlon_to_sinu(0.0, 0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)


def test_tile_origin_h0v0():
    """Tile (0, 0) origin is the top-left corner of the sinusoidal grid."""
    calc = CalculationsMajorTom()
    x0, y0 = calc.tile_origin(0, 0)
    assert math.isclose(x0, -20015109, rel_tol=1e-9)
    assert math.isclose(y0, 10007555, rel_tol=1e-9)
