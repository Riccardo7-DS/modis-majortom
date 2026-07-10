from __future__ import annotations

from datetime import datetime

import numpy as np


def compute_cos_sza(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    dt: datetime,
) -> np.ndarray:
    """Vectorised cos(solar zenith angle) via the Spencer approximation (~0.5° accuracy).

    Args:
        lat_deg: Latitude(s) in degrees. Scalar or any broadcastable array.
        lon_deg: Longitude(s) in degrees. Same shape as lat_deg.
        dt: UTC datetime of the observation.

    Returns:
        cos(SZA) as float32 array of the same shape as lat_deg/lon_deg.
    """
    doy = dt.timetuple().tm_yday
    B = np.radians(360.0 / 365.0 * (doy - 81))
    decl_rad = np.radians(23.45 * np.sin(B))
    eot_min = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    utc_h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    solar_time = utc_h + lon_deg / 15.0 + eot_min / 60.0
    hour_angle_rad = np.radians(15.0 * (solar_time - 12.0))
    lat_rad = np.radians(lat_deg)
    cos_sza = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad)
    )
    return cos_sza.astype(np.float32)
