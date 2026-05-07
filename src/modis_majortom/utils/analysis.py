import numpy as np
import xarray as xr


def decode_mod35(raw: np.ndarray) -> np.ndarray:
    """Decode stored MOD35 4-level quality to a binary cloud mask.

    Stored encoding: 0=confident cloudy, 1=probably cloudy,
                     2=probably clear,   3=confident clear, 99=fill.

    Uses the binned algorithm: quality < 2 → cloud, quality >= 2 → clear.
    Fill pixels (99) are returned as False (not flagged as cloud).

    Returns
    -------
    bool array, True = cloudy
    """
    r = np.asarray(raw, dtype=np.uint8)
    return np.where(r == 99, False, r < 2)


def decode_state1km(
    raw: np.ndarray,
    algorithm: str = "binned",
    upsample: bool = False,
) -> np.ndarray:
    """Decode MOD09GA state_1km QA bits to a binary cloud mask.

    Parameters
    ----------
    raw:
        Raw state_1km array (uint16) from the MOD09GA zarr store.
    algorithm:
        Passed to :meth:`MODISQCMask.state_1km_cloud_mask`.
        One of ``"strict"``, ``"cloud_state"``, ``"internal"``, ``"binned"``.
    upsample:
        If True, apply ``upsample_500_to_250`` (nearest-neighbour ×2) to the
        result before returning.  Use when aligning a 500 m mask to 250 m NDVI.

    Returns
    -------
    bool array, True = cloudy
    """
    from ..eo_data.modis import MODISQCMask
    mask = MODISQCMask.state_1km_cloud_mask(
        np.asarray(raw, dtype=np.uint16), algorithm=algorithm
    )
    if upsample:
        from ..transform.cloud_adapter import upsample_500_to_250
        mask = upsample_500_to_250(mask)
    return mask.astype(bool)


def compute_ndvi(band1: xr.DataArray,
                 band2: xr.DataArray) -> xr.DataArray:
    """
    Compute NDVI given two reflectance bands:
    band1 = RED
    band2 = NIR
    """
    denom = band2 + band1
    num   = band2 - band1

    # Mask zero denominators BEFORE division → no warnings
    safe_denom = xr.where(denom == 0, np.nan, denom)

    ndvi = num / safe_denom

    return ndvi.astype("float32")
