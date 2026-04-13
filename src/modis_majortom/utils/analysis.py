import numpy as np
import xarray as xr


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
