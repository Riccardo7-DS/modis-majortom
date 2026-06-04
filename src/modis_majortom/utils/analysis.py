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


def mask_invalid_reflectance(
    band: "np.ndarray | xr.DataArray",
    fill_below: float = -0.05,
) -> "np.ndarray | xr.DataArray":
    """Set physically invalid reflectance pixels to NaN.

    Parameters
    ----------
    band :
        Reflectance array (numpy or xarray).  Modified in-place for numpy
        inputs; returns a new DataArray for xarray inputs.
    fill_below :
        Pixels with value strictly below this threshold are set to NaN.
        The default (−0.05) catches MODIS fill values while keeping slightly
        negative valid retrievals caused by atmospheric over-correction.

    Returns
    -------
    Same type as input, with fill pixels replaced by NaN.
    """
    if isinstance(band, xr.DataArray):
        return band.where(band >= fill_below)
    arr = np.asarray(band, dtype=np.float32)
    arr[arr < fill_below] = np.nan
    return arr


def compute_ndvi(
    band1: "np.ndarray | xr.DataArray",
    band2: "np.ndarray | xr.DataArray",
    fill_below: float | None = None,
) -> "np.ndarray | xr.DataArray":
    """Compute NDVI from two reflectance bands.

    Parameters
    ----------
    band1 :
        Red band (numpy or xarray).
    band2 :
        NIR band (numpy or xarray).
    fill_below :
        If given, apply :func:`mask_invalid_reflectance` with this threshold
        before computing the ratio.  Prevents fill-value blow-up in the
        denominator.  Pass ``-0.05`` to use the MODIS default.

    Returns
    -------
    float32 array of the same type as the inputs.
    """
    if fill_below is not None:
        band1 = mask_invalid_reflectance(band1, fill_below)
        band2 = mask_invalid_reflectance(band2, fill_below)

    denom = band2 + band1
    num   = band2 - band1

    if isinstance(denom, xr.DataArray):
        safe_denom = denom.where(denom != 0)
    else:
        safe_denom = np.where(denom == 0, np.nan, denom)

    ndvi = num / safe_denom
    return np.asarray(ndvi, dtype=np.float32) if not isinstance(ndvi, xr.DataArray) else ndvi.astype("float32")
