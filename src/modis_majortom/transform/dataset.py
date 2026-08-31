"""
dataset.py
==========
Spatially-aligned multi-source PyTorch Dataset for MTG FCI + MODIS.

Each sample exposes two tensors on a common Earth-fixed AEQD grid centred on
the MajorTOM cell:

    X : (18, H, W) float32 — MTG FCI multi-temporal conditioning tensor
        Channels 0-14  : 5 timestamps x [vis06, vis08, cos_sza] (chronological,
                         zero-padded if fewer than 5 timestamps are available)
        Channel 15     : NDVI75    — 75th-percentile NDVI across the 5 timestamps
        Channel 16     : NDVI_std  — temporal std of NDVI across the 5 timestamps
        Channel 17     : CloudScore — fraction of cloud-flagged timestamps [0, 1]

    target_ndvi : (1, H, W) float32 — blended NDVI target
        ``soft_score * ndvi_observed + (1 − soft_score) * ndvi_whittaker``,
        clamped to [−1, 1].  NaN filled from Whittaker; 0 as last resort.
    loss_weight : (1, H, W) float32 — per-pixel training weight
        ``soft_score + alpha * (1 − soft_score)`` (default alpha = 0.2).
    meta_mask   : (1, H, W) float32 — reliable-observation mask
        Equal to soft_score where soft_score ≥ 0.5, else 0.

Pixel [i, j] in X and Y refers to the same point on the ground.
Geometry and reprojection are handled by ``eumetsearch.transform``.
"""

from __future__ import annotations

import logging
import threading
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
import zarr
from scipy.ndimage import laplace as _ndimage_laplace, map_coordinates
from torch.utils.data import Dataset

from eumetsearch.transform import (
    TargetGrid,
    MODISGeometry,
    modis_patch_corner_for_cell,
    grid_id_to_latlon,
)
from eumetsearch.transform.fci_modis_align import (
    _aeqd_to_latlon,
    _latlon_to_fci_px,
    _latlon_to_modis_px,
    _sample_array,
)
from eumetsearch.transform.ndvi import _build_fci_index
from ..utils import compute_ndvi
from ..utils.mtg_reliability import compute_mtg_composites
from ..utils.solar import compute_cos_sza as _compute_cos_sza
from .cloud_adapter import upsample_500_to_250

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  AncillarySource ABC
# ---------------------------------------------------------------------------

class AncillarySource(ABC):
    """Abstract base class for any ancillary data source."""

    @abstractmethod
    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        """Return ``{variable_name: array (H, W)}`` in the source's native grid."""

    @abstractmethod
    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        """Top-left ``(row0, col0)`` of the patch in the source's global pixel grid."""

    @abstractmethod
    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid: TargetGrid,
    ) -> dict[str, np.ndarray]:
        """Reproject native-grid arrays onto *target_grid*.

        Returns ``{name: (H_out, W_out) float32}``.
        """

    def available_dates(self) -> set[str]:
        return set()

    def available_grid_ids(self, date: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# 2.  MODISSource
# ---------------------------------------------------------------------------

class MODISSource(AncillarySource):
    """Wraps raw and processed MODIS zarr stores as an ancillary source.

    ``raw_zarr_path`` must contain ``patches/sur_refl_b01`` and
    ``patches/sur_refl_b02`` (MOD09GQ 250 m bands).  ``ndvi_observed`` is
    computed on-the-fly from these so it never needs to be pre-stored.

    ``processed_zarr_path`` must contain the Whittaker pipeline outputs:
    ``patches/ndvi_envelope`` and ``patches/soft_score`` (written by
    ``WhittakerPipeline.process_zarr`` with ``product="MOD09GA"``).
    Patches are 256×256 (500 m) and are upsampled 2× to 512×512 inside
    :meth:`load` to match the GQ coordinate frame used by :meth:`reproject`.

    Both stores are opened once per thread and cached in thread-local storage.
    """

    PATCH_PX: int = 512

    def __init__(
        self,
        raw_zarr_path: Union[str, Path],
        processed_zarr_path: Union[str, Path],
        use_gap_mask: bool = False,
        gap_threshold_days: float = 7.0,
    ) -> None:
        self._raw_path          = str(raw_zarr_path)
        self._processed_path    = str(processed_zarr_path)
        self._use_gap_mask      = use_gap_mask
        self._gap_threshold     = gap_threshold_days
        self._local             = threading.local()

    @property
    def _raw_store(self):
        if not hasattr(self._local, "raw_store"):
            self._local.raw_store = zarr.open(self._raw_path, mode="r", use_consolidated=False)
        return self._local.raw_store

    @property
    def _processed_store(self):
        if not hasattr(self._local, "processed_store"):
            self._local.processed_store = zarr.open(self._processed_path, mode="r", use_consolidated=False)
        return self._local.processed_store

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        raw  = self._raw_store["patches"]
        proc = self._processed_store["patches"]

        # ndvi_observed: on-the-fly from GQ (512×512, 250 m)
        b01      = raw["sur_refl_b01"][date][grid_id][:].astype(np.float32)
        b02      = raw["sur_refl_b02"][date][grid_id][:].astype(np.float32)
        ndvi_obs = np.asarray(compute_ndvi(b02, b01, fill_below=-0.05), dtype=np.float32)

        # ndvi_envelope + soft_score from GA-processed zarr (256×256, 500 m).
        # Upsample 2× so all bands share the 512×512 GQ coordinate frame expected by reproject().
        ndvi_whi = upsample_500_to_250(proc["ndvi_envelope"][date][grid_id][:].astype(np.float32))
        soft     = upsample_500_to_250(proc["soft_score"][date][grid_id][:].astype(np.float32))
        bands = {"ndvi_observed": ndvi_obs, "ndvi_whittaker": ndvi_whi, "soft_score": soft}

        if self._use_gap_mask:
            try:
                fwd = proc["forward_gap"][date][grid_id][:].astype(np.float32)
                bwd = proc["backward_gap"][date][grid_id][:].astype(np.float32)
                # 1 = reliable (clear obs within ±gap_threshold days), 0 = both gaps exceed threshold
                # inf values (no obs in that direction) satisfy >= threshold → masked out correctly
                gap_mask = (~((fwd >= self._gap_threshold) & (bwd >= self._gap_threshold))).astype(np.float32)
            except KeyError:
                gap_mask = np.ones((ndvi_whi.shape[0] // 2, ndvi_whi.shape[1] // 2), dtype=np.float32)
            bands["gap_cloud_mask"] = upsample_500_to_250(gap_mask)

        return bands

    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        return modis_patch_corner_for_cell(lat, lon, patch_px=self.PATCH_PX)

    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid: TargetGrid,
    ) -> dict[str, np.ndarray]:
        lat, lon = _aeqd_to_latlon(target_grid)
        h, w = next(iter(bands.values())).shape[-2:]
        frac_row, frac_col = _latlon_to_modis_px(lat, lon, row0, col0, h, w)
        return {k: _sample_array(v, frac_row, frac_col) for k, v in bands.items()}

    def available_dates(self) -> set[str]:
        return set(self._processed_store["patches"]["ndvi_envelope"].keys())

    def available_grid_ids(self, date: str) -> set[str]:
        try:
            proc_ids = set(self._processed_store["patches"]["ndvi_envelope"][date].keys())
            raw_b01  = set(self._raw_store["patches"]["sur_refl_b01"][date].keys())
            raw_b02  = set(self._raw_store["patches"]["sur_refl_b02"][date].keys())
            return proc_ids & raw_b01 & raw_b02
        except KeyError:
            return set()


# ---------------------------------------------------------------------------
# 3.  RawGAMODISSource
# ---------------------------------------------------------------------------

class RawGAMODISSource(AncillarySource):
    """MODIS MOD09GA raw-band source — no Whittaker processing required.

    Reads ``sur_refl_b01`` (Red 645 nm) and ``sur_refl_b02`` (NIR 858 nm)
    from the raw GA zarr and computes NDVI on-the-fly.  Cloud mask is
    derived from ``state_1km`` (bits 0-1: 0 = clear).

    Since there is no temporal smoothing, ``ndvi_whittaker`` is set equal
    to ``ndvi_observed`` and ``soft_score`` is binary (1.0 = clear, 0.0 =
    cloudy/mixed/unknown).  This is the right source for quick experiments
    before the Whittaker pipeline has been run.

    Patches are 256×256 at 500 m (MOD09GA native resolution).
    """

    PATCH_PX: int = 256

    def __init__(self, raw_zarr_path: Union[str, Path]) -> None:
        self._raw_path = str(raw_zarr_path)
        self._local    = threading.local()

    @property
    def _raw_store(self):
        if not hasattr(self._local, "raw_store"):
            self._local.raw_store = zarr.open(self._raw_path, mode="r")
        return self._local.raw_store

    def available_dates(self) -> set[str]:
        return set(self._raw_store["patches"]["sur_refl_b01"].keys())

    def available_grid_ids(self, date: str) -> set[str]:
        try:
            p = self._raw_store["patches"]
            return (
                set(p["sur_refl_b01"][date].keys())
                & set(p["sur_refl_b02"][date].keys())
                & set(p["state_1km"][date].keys())
            )
        except KeyError:
            return set()

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        raw = self._raw_store["patches"]
        b01 = raw["sur_refl_b01"][date][grid_id][:].astype(np.float32)  # Red
        b02 = raw["sur_refl_b02"][date][grid_id][:].astype(np.float32)  # NIR
        state = raw["state_1km"][date][grid_id][:].astype(np.int32)

        ndvi_obs = np.asarray(
            compute_ndvi(b02, b01, fill_below=-0.05), dtype=np.float32
        )
        # bits 0-1 of state_1km: 0 = clear, 1 = cloudy, 2 = mixed, 3 = unknown
        soft = ((state & 3) == 0).astype(np.float32)

        return {
            "ndvi_observed":   ndvi_obs,
            "ndvi_whittaker":  ndvi_obs,  # no temporal smoothing available
            "soft_score":      soft,
        }

    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        return modis_patch_corner_for_cell(lat, lon, patch_px=self.PATCH_PX)

    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid: TargetGrid,
    ) -> dict[str, np.ndarray]:
        lat, lon = _aeqd_to_latlon(target_grid)
        h, w = next(iter(bands.values())).shape[-2:]
        frac_row, frac_col = _latlon_to_modis_px(lat, lon, row0, col0, h, w)
        return {k: _sample_array(v, frac_row, frac_col) for k, v in bands.items()}


# ---------------------------------------------------------------------------
# 4.  MOD13A3Source
# ---------------------------------------------------------------------------

class MOD13A3Source:
    """MOD13A3 monthly NDVI composite (1 km, 128×128 patches).

    For a sample at (grid_id, date), loads the composite for the **previous
    month** — always available by ~day 10 of the current month, no look-ahead.

    The 128×128 patch at 1 km covers the same 128 km × 128 km footprint as
    the 256×256 MOD09GA patches at 500 m and is already spatially aligned
    (same MajorTOM grid_id system), so no coordinate reprojection is needed.

    Returns ``{"ndvi_monthly": (128, 128) float32}``.  NaN / fill / out-of-range
    pixels are zeroed (neutral value avoids bias in the diffusion prior).
    """

    def __init__(self, zarr_path: str | Path) -> None:
        self._path  = str(zarr_path)
        self._local = threading.local()

    @property
    def _store(self):
        if not hasattr(self._local, "store"):
            self._local.store = zarr.open(self._path, mode="r", zarr_format=2)
        return self._local.store

    @staticmethod
    def _prev_month_key(date: str) -> str:
        """'2025-07-15' → '2025-06-01'"""
        from datetime import date as _date
        d = _date.fromisoformat(date)
        if d.month == 1:
            return f"{d.year - 1}-12-01"
        return f"{d.year}-{d.month - 1:02d}-01"

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        key = self._prev_month_key(date)
        try:
            arr = self._store["patches"]["ndvi"][key][grid_id][:].astype(np.float32)
            arr = np.where(np.isfinite(arr), np.clip(arr, -0.2, 1.0), 0.0)
        except KeyError:
            arr = np.zeros((128, 128), dtype=np.float32)
        return {"ndvi_monthly": arr}


# ---------------------------------------------------------------------------
# 5.  ERA5Source
# ---------------------------------------------------------------------------

class ERA5Source(AncillarySource):
    """ERA5-Land daily data as an ancillary source.

    Reads a netCDF file (ERA5-Land format) and returns the last *n_days*
    of each variable centred on the sample date, reprojected onto the shared
    AEQD target grid.

    Output tensor (returned by ``AlignedPatchDataset.__getitem__`` as the
    ``"era5"`` key): ``(N_vars, n_days, H, W)`` float32.

    The file is opened once per thread and kept open; spatial slices are
    read lazily so only the ±``SPATIAL_MARGIN_DEG`` bounding box around
    the cell is transferred from disk.

    Performance note
    ----------------
    First read per unique (date-range, lat/lon-extent) takes ~3 s on a cold
    filesystem cache; subsequent reads from the same file are ~5 ms.
    For production training loops, pre-convert the netCDF to zarr with
    chunk shape ``(15, 30, 30)`` to get consistently fast random access.
    """

    SPATIAL_MARGIN_DEG: float = 1.5

    def __init__(
        self,
        nc_path: Union[str, Path],
        variables: list[str] | None = None,
        n_days: int = 15,
    ) -> None:
        import xarray as xr

        self._nc_path = str(nc_path)
        self._n_days  = n_days
        self._local   = threading.local()

        # Read coordinate arrays and date list once at init (fast — metadata only)
        with xr.open_dataset(nc_path) as _ds:
            self._lat    = _ds.latitude.values.astype(np.float64)   # decreasing (N→S)
            self._lon    = _ds.longitude.values.astype(np.float64)  # increasing (W→E)
            self._dates  = [str(t)[:10] for t in _ds.valid_time.values]
            self.variables = variables or [str(v) for v in _ds.data_vars]

        self._date_to_idx: dict[str, int] = {d: i for i, d in enumerate(self._dates)}

    @property
    def _dataset(self):
        """Thread-local xarray Dataset (opened once per worker)."""
        if not hasattr(self._local, "ds"):
            import xarray as xr
            self._local.ds = xr.open_dataset(self._nc_path)
        return self._local.ds

    # ------------------------------------------------------------------
    # AncillarySource interface
    # ------------------------------------------------------------------

    def available_dates(self) -> set[str]:
        """Dates for which a full n_days look-back is available in the file."""
        return set(self._dates[self._n_days - 1:])

    def available_grid_ids(self, date: str) -> set[str]:
        # ERA5 covers the whole domain — return empty set so build_sample_index
        # does not filter by grid_id for this source.
        return set()

    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        return (0, 0)  # not used; ERA5 reproject uses lat/lon directly

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        """Return ``{var: (n_days, H_slice, W_slice)}`` for the last n_days.

        Also adds ``"_lat_slice"`` and ``"_lon_slice"`` arrays (consumed by
        :meth:`reproject`) to the returned dict.
        """
        from datetime import datetime, timedelta

        cell_lat, cell_lon = grid_id_to_latlon(grid_id)
        m = self.SPATIAL_MARGIN_DEG

        lat_mask = (self._lat >= cell_lat - m) & (self._lat <= cell_lat + m)
        lon_mask = (self._lon >= cell_lon - m) & (self._lon <= cell_lon + m)
        lat_idx  = np.where(lat_mask)[0]
        lon_idx  = np.where(lon_mask)[0]

        date_dt   = datetime.fromisoformat(date)
        day_strs  = [(date_dt - timedelta(days=i)).strftime("%Y-%m-%d")
                     for i in range(self._n_days - 1, -1, -1)]
        t_indices = [self._date_to_idx[d] for d in day_strs if d in self._date_to_idx]

        _nan_shape = (self._n_days, 1, 1)
        if len(lat_idx) == 0 or len(lon_idx) == 0 or len(t_indices) == 0:
            # Cell outside ERA5 domain or date not found — return zeros
            bands: dict[str, np.ndarray] = {
                v: np.zeros(_nan_shape, dtype=np.float32) for v in self.variables
            }
            bands["_lat_slice"] = np.array([cell_lat], dtype=np.float32)
            bands["_lon_slice"] = np.array([cell_lon], dtype=np.float32)
            return bands

        lat_s = slice(int(lat_idx[0]),  int(lat_idx[-1]) + 1)
        lon_s = slice(int(lon_idx[0]),  int(lon_idx[-1]) + 1)
        t_s   = slice(int(min(t_indices)), int(max(t_indices)) + 1)

        # Single isel call reads the full spatial+temporal block in one I/O
        sub = self._dataset.isel(valid_time=t_s, latitude=lat_s, longitude=lon_s).load()

        bands = {}
        for var in self.variables:
            arr = sub[var].values.astype(np.float32)  # (n_days, H_sl, W_sl)
            # Pad if fewer dates than n_days were found in the file
            if arr.shape[0] < self._n_days:
                pad = np.zeros(
                    (self._n_days - arr.shape[0], *arr.shape[1:]), dtype=np.float32
                )
                arr = np.concatenate([arr, pad], axis=0)
            bands[var] = arr

        bands["_lat_slice"] = self._lat[lat_s].astype(np.float32)
        bands["_lon_slice"] = self._lon[lon_s].astype(np.float32)
        return bands

    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid: TargetGrid,
    ) -> dict[str, np.ndarray]:
        """Bilinear interpolation from ERA5 lat/lon slice onto the AEQD grid.

        Each variable ``v`` → ``(n_days, H_out, W_out)`` float32.
        """
        # Consume the spatial coordinate arrays injected by load()
        lat_slice = bands.pop("_lat_slice")  # (H_sl,) decreasing
        lon_slice = bands.pop("_lon_slice")  # (W_sl,) increasing

        lat_grid, lon_grid = _aeqd_to_latlon(target_grid)
        H, W = lat_grid.shape

        lat_origin  = float(lat_slice[0])
        lon_origin  = float(lon_slice[0])
        lat_spacing = abs(float(lat_slice[1] - lat_slice[0])) if len(lat_slice) > 1 else 0.1
        lon_spacing = abs(float(lon_slice[1] - lon_slice[0])) if len(lon_slice) > 1 else 0.1

        frac_row = ((lat_origin - lat_grid) / lat_spacing).astype(np.float64)
        frac_col = ((lon_grid - lon_origin) / lon_spacing).astype(np.float64)

        valid = (
            np.isfinite(frac_row) & np.isfinite(frac_col)
            & (frac_row >= 0) & (frac_row <= len(lat_slice) - 1)
            & (frac_col >= 0) & (frac_col <= len(lon_slice) - 1)
        )
        rows_v = frac_row[valid]
        cols_v = frac_col[valid]

        out: dict[str, np.ndarray] = {}
        for name, arr in bands.items():
            n_days = arr.shape[0]
            result = np.zeros((n_days, H, W), dtype=np.float32)
            if valid.any():
                for t in range(n_days):
                    plane = np.zeros((H, W), dtype=np.float32)
                    plane[valid] = map_coordinates(
                        arr[t].astype(np.float64),
                        [rows_v, cols_v],
                        order=1, mode="nearest", cval=0.0, prefilter=False,
                    )
                    result[t] = plane
            out[name] = result
        return out


# ---------------------------------------------------------------------------
# 5.  LAISource
# ---------------------------------------------------------------------------

def _fparlai_qc_bad(qc: np.ndarray) -> np.ndarray:
    """FparLai_QC bad-pixel mask (MCD15A3H / MOD15A2H bit layout).

    Bit 0     MODLAND_QC   0 = good quality (main RT method), 1 = other (backup/fill)
    Bits 4-5  CloudState   00 = no significant clouds present, else = cloud present/undefined

    Duplicated here rather than imported from ``eo_data.modis.MODISQCMask`` so
    this module stays free of that module's heavy optional deps (earthaccess,
    ee/geemap, rasterio) — this is a 2-instruction bitmask, not worth the coupling.

    Returns True = BAD (exclude from supervision).
    """
    qc = np.asarray(qc, dtype=np.uint8)
    modland_bad = (qc & 0b1) != 0
    cloud_state = (qc >> 4) & 0b11
    return modland_bad | (cloud_state != 0)


class LAISource(AncillarySource):
    """MCD15A3H (4-day composite LAI/FPAR) as a *sparse* ancillary source.

    Unlike ``ERA5Source``/``LandCoverSource``, LAI/FPAR is only available on
    ~1-in-4 days (the fixed MCD15A3H compositing calendar), so this source
    must NOT be added to the ``ancillary_sources`` dict passed to
    ``build_sample_index`` / ``AlignedPatchDataset`` — doing so would
    intersect its sparse ``available_dates()``/``available_grid_ids()`` with
    every other source and restrict the *entire* daily training set (NDVI
    included) down to composite dates.

    Instead, pass it as ``AlignedPatchDataset(..., lai_source=LAISource(...))``:
    the dataset calls :meth:`has` per sample and falls back to an all-zero
    target + mask when no real composite exists for that (grid_id, date), so
    the LAI head simply gets no gradient on non-composite days (see
    ``EDMDiffusion._lai_loss``).

    Reads ``Lai_500m`` / ``FparLai_QC`` (and optionally ``Fpar_500m``) bands
    from an MCD15A3H zarr laid out like ``RawGAMODISSource``:
    ``patches/{band}/{date}/{grid_id}``, 500 m native MODIS-sinusoidal
    patches (256×256).

    ``Lai_500m`` / ``Fpar_500m`` are stored *already* scaled to physical
    units (m²/m² and fraction respectively) by the download pipeline
    (``EarthAccessDownloader``'s ``LAI_FPAR_BANDS`` branch applies each
    band's own ``scale_factor`` and masks fill/out-of-range/special QC codes
    — raw 249-254 — to NaN at write time, before scaling). This class must
    NOT re-apply a scale factor or a raw-value range check — NaN pixels are
    already the exclusion signal. QC-flagged pixels (see
    :func:`_fparlai_qc_bad`) are excluded via the returned ``lai_qc``/
    ``fpar_qc`` weights (1 = good, 0 = exclude); ``FparLai_QC`` is the single
    QC field MODIS defines for both bands jointly, so both weights start
    from the same QC decode and differ only in each band's own NaN pattern.

    ``fpar_band`` defaults to ``None`` (disabled) for backward compatibility
    with MCD15A3H zarrs downloaded before ``Fpar_500m`` support was added —
    pass ``fpar_band="Fpar_500m"`` explicitly to also load FAPAR.
    """

    PATCH_PX: int = 256

    def __init__(
        self,
        raw_zarr_path: Union[str, Path],
        band: str = "Lai_500m",
        qc_band: str = "FparLai_QC",
        fpar_band: str | None = None,
    ) -> None:
        self._raw_path  = str(raw_zarr_path)
        self._band      = band
        self._qc_band   = qc_band
        self._fpar_band = fpar_band
        self._local     = threading.local()

    @property
    def _raw_store(self):
        if not hasattr(self._local, "raw_store"):
            self._local.raw_store = zarr.open(self._raw_path, mode="r")
        return self._local.raw_store

    @property
    def has_fpar(self) -> bool:
        return self._fpar_band is not None

    def available_dates(self) -> set[str]:
        return set(self._raw_store["patches"][self._band].keys())

    def available_grid_ids(self, date: str) -> set[str]:
        try:
            p = self._raw_store["patches"]
            ids = set(p[self._band][date].keys()) & set(p[self._qc_band][date].keys())
            if self._fpar_band:
                ids &= set(p[self._fpar_band][date].keys())
            return ids
        except KeyError:
            return set()

    def has(self, grid_id: str, date: str) -> bool:
        """True if a real MCD15A3H composite exists for this (grid_id, date).

        Called per-sample by ``AlignedPatchDataset`` — cheap zarr key lookups,
        no array reads — to decide whether to load real data or zero-fill.
        """
        try:
            p = self._raw_store["patches"]
            ok = grid_id in p[self._band][date] and grid_id in p[self._qc_band][date]
            if self._fpar_band:
                ok = ok and grid_id in p[self._fpar_band][date]
            return ok
        except KeyError:
            return False

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        p   = self._raw_store["patches"]
        lai = p[self._band][date][grid_id][:].astype(np.float32)
        qc  = p[self._qc_band][date][grid_id][:].astype(np.uint8)

        # NaN already marks fill/out-of-range pixels (masked at download time).
        qc_bad = _fparlai_qc_bad(qc)
        lai_qc = (~(qc_bad | ~np.isfinite(lai))).astype(np.float32)

        out = {"lai": lai, "lai_qc": lai_qc}

        if self._fpar_band:
            fpar = p[self._fpar_band][date][grid_id][:].astype(np.float32)
            out["fpar"] = fpar
            out["fpar_qc"] = (~(qc_bad | ~np.isfinite(fpar))).astype(np.float32)

        return out

    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        return modis_patch_corner_for_cell(lat, lon, patch_px=self.PATCH_PX)

    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid: TargetGrid,
    ) -> dict[str, np.ndarray]:
        lat, lon = _aeqd_to_latlon(target_grid)
        h, w = next(iter(bands.values())).shape[-2:]
        frac_row, frac_col = _latlon_to_modis_px(lat, lon, row0, col0, h, w)
        return {k: _sample_array(v, frac_row, frac_col) for k, v in bands.items()}


# ---------------------------------------------------------------------------
# 6.  AlignedPatchDataset
# ---------------------------------------------------------------------------

class AlignedPatchDataset(Dataset):
    """Spatially-aligned MTG FCI + MODIS PyTorch Dataset.

    Each sample returns two tensors on a shared AEQD grid:

    ``X`` : (18, H, W) float32
        Multi-temporal FCI conditioning tensor.
        Channels 0–14: 5 timestamps x ``[vis06, vis08, cos_sza]`` (chronological;
        zero-padded if fewer than 5 timestamps are available).
        Channels 15–17: ``[NDVI75, NDVI_std, CloudScore]`` — temporal
        reliability composites derived across the 5 timestamps.

    ``target_ndvi`` : (1, H, W) float32
        Blended NDVI target: ``soft_score * ndvi_observed + (1−soft_score) * ndvi_whittaker``,
        clamped to [−1, 1].  NaN pixels fall back to ``ndvi_whittaker``; then 0.
    ``loss_weight`` : (1, H, W) float32
        Per-pixel training weight: ``soft_score + alpha * (1−soft_score)``.
        Full weight for clear pixels, reduced (alpha-scaled) for cloudy/gap-filled pixels.
    ``meta_mask`` : (1, H, W) float32
        Reliable-observation flag: ``soft_score`` where ``soft_score ≥ 0.5``, else 0.

    ``target_lai`` / ``lai_mask`` : (1, H, W) float32, only present when ``lai_source`` is set
        Sparse LAI/FAPAR supervision (see ``LAISource``). ``lai_mask`` is 0
        everywhere for samples whose date has no real MCD15A3H composite (and
        per-pixel 0 for QC-bad pixels within a valid composite); ``target_lai``
        is 0 wherever ``lai_mask`` is 0. Not densified/interpolated — this is
        deliberately sparse-in-time supervision, see project memory.
    ``target_fpar`` / ``fpar_mask`` : (1, H, W) float32, only present when
        ``lai_source.has_fpar`` is True (i.e. constructed with
        ``fpar_band="Fpar_500m"``). Same sparse-supervision semantics as
        ``target_lai``/``lai_mask``, independently masked (FAPAR NaN pattern
        can differ slightly from LAI's even though both share ``FparLai_QC``).

    Parameters
    ----------
    fci_store_path :
        Path to the MTG FCI zarr store (MajorTOM format).
    ancillary_sources :
        Dict mapping source name → ``AncillarySource``.
        Must include ``"modis": MODISSource(...)``.
    samples :
        ``[(grid_id, date), ...]`` index — build with :func:`build_sample_index`.
    target_n_pixels :
        Output spatial resolution (square, default 256).
    alpha :
        Weight floor for cloud-contaminated pixels (default 0.2).
    lai_source :
        Optional ``LAISource`` for sparse LAI/FAPAR supervision. Deliberately
        kept out of ``ancillary_sources`` — see ``LAISource`` docstring for why
        it must not participate in ``build_sample_index`` filtering.
    """

    def __init__(
        self,
        fci_store_path: Union[str, Path],
        ancillary_sources: dict[str, AncillarySource],
        samples: list[tuple[str, str]],
        target_n_pixels: int = 256,
        alpha: float = 0.2,
        whittaker_target: bool = False,
        lai_source: "LAISource | None" = None,
        mod13a3_source: "MOD13A3Source | None" = None,
    ) -> None:
        if "modis" not in ancillary_sources:
            raise ValueError("ancillary_sources must include a 'modis' key (MODISSource)")
        self._fci_store_path   = str(fci_store_path)
        self.ancillary_sources = ancillary_sources
        self.samples           = list(samples)
        self.target_n_pixels   = target_n_pixels
        self._alpha            = alpha
        self._whittaker_target = whittaker_target
        self.lai_source        = lai_source
        self.mod13a3_source    = mod13a3_source
        self._local            = threading.local()

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def _fci_store(self):
        if not hasattr(self._local, "fci_store"):
            self._local.fci_store = zarr.open(self._fci_store_path, mode="r")
        return self._local.fci_store

    def __getitem__(self, idx: int) -> dict:
        grid_id, date = self.samples[idx]
        lat, lon = grid_id_to_latlon(grid_id)

        fci_store = self._fci_store
        fci_index = _build_fci_index(fci_store, date)
        if grid_id not in fci_index:
            raise KeyError(f"No FCI data for {grid_id!r} on {date!r}")
        ts_rows = fci_index[grid_id]  # {timestamp: row_index}

        # Shared AEQD grid — same extent as the MODIS patch
        target_grid = TargetGrid(
            cell_lat=lat,
            cell_lon=lon,
            extent_m=MODISGeometry.PATCH_EXTENT_M / 2,
            n_pixels=self.target_n_pixels,
        )

        # X: multi-temporal FCI conditioning tensor (18, H, W)
        X = self._build_mtg_conditioning(fci_store, ts_rows, target_grid)

        # Load and reproject three MODIS NDVI-related arrays onto the shared AEQD grid
        modis_src      = self.ancillary_sources["modis"]
        modis_bands    = modis_src.load(grid_id, date)
        mod_r0, mod_c0 = modis_src.patch_origin(lat, lon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligned = modis_src.reproject(modis_bands, mod_r0, mod_c0, target_grid)

        ndvi_obs      = aligned["ndvi_observed"]   # (H, W) float32
        ndvi_whi      = aligned["ndvi_whittaker"]  # (H, W) float32
        soft          = aligned["soft_score"]      # (H, W) float32
        gap_cloud_mask = aligned.get("gap_cloud_mask")  # (H, W) float32 | None

        # Clamp soft_score to [0, 1]; treat NaN as fully uncertain (0)
        soft = np.clip(np.nan_to_num(soft, nan=0.0), 0.0, 1.0)

        if self._whittaker_target:
            # Pure Whittaker reconstruction as target (cloud-free by construction)
            target_ndvi = np.clip(ndvi_whi, -1.0, 1.0)
            nan_mask = np.isnan(target_ndvi)
            if nan_mask.any():
                target_ndvi[nan_mask] = 0.0
        else:
            # Blend: confident-clear pixels use observed NDVI; uncertain pixels use Whittaker
            target_ndvi = soft * ndvi_obs + (1.0 - soft) * ndvi_whi
            target_ndvi = np.clip(target_ndvi, -1.0, 1.0)

            # Fill any NaN that survived blending (both ndvi_obs and ndvi_whi were NaN)
            nan_mask = np.isnan(target_ndvi)
            if nan_mask.any():
                target_ndvi[nan_mask] = ndvi_whi[nan_mask]
                still_nan = np.isnan(target_ndvi)
                if still_nan.any():
                    target_ndvi[still_nan] = 0.0

        # Full weight for clear observations; alpha-scaled floor for cloudy/gap-filled pixels
        loss_weight = soft + self._alpha * (1.0 - soft)

        # Conditioning mask: pass soft_score through only where it signals reliable obs
        meta_mask = np.where(soft >= 0.5, soft, 0.0).astype(np.float32)

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[np.newaxis].astype(np.float32))  # (1, H, W)

        sample: dict = {
            "X":           X,
            "target_ndvi": _to_tensor(target_ndvi),
            "loss_weight": _to_tensor(loss_weight),
            "meta_mask":   _to_tensor(meta_mask),
            "grid_id":     grid_id,
            "date":        date,
            "cell_lat":    lat,
            "cell_lon":    lon,
        }
        if gap_cloud_mask is not None:
            sample["whittaker_cloud_mask"] = _to_tensor(gap_cloud_mask)

        # ERA5 ancillary — optional; shape (N_vars, n_days, H, W)
        era5_src = self.ancillary_sources.get("era5")
        if era5_src is not None:
            era5_bands = era5_src.load(grid_id, date)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                era5_aligned = era5_src.reproject(era5_bands, 0, 0, target_grid)
            era5_stack = np.stack(
                [era5_aligned[v] for v in era5_src.variables], axis=0
            )  # (N_vars, n_days, H, W)
            sample["era5"] = torch.from_numpy(era5_stack.astype(np.float32))

        # Land cover ancillary — optional; static per grid_id, date ignored
        # Shape: (N_lc_vars, H, W) float32 with IGBP class indices 0-17
        lc_src = self.ancillary_sources.get("land_cover")
        if lc_src is not None:
            lc_bands   = lc_src.load(grid_id, date)
            lc_aligned = lc_src.reproject(lc_bands, 0, 0, target_grid)
            lc_stack   = np.stack([lc_aligned[v] for v in lc_src.variables], axis=0)
            sample["land_cover"] = torch.from_numpy(lc_stack.astype(np.float32))

        # LAI/FAPAR ancillary — optional, sparse (MCD15A3H 4-day composite calendar).
        # Unlike era5/land_cover, real data only exists on ~1-in-4 dates; when this
        # sample's date isn't a real composite date (or the grid_id is missing that
        # date), target_lai/lai_mask are zero-filled so the LAI head gets no gradient
        # for this sample — see LAISource and EDMDiffusion._lai_loss.
        if self.lai_source is not None:
            H = W = self.target_n_pixels
            if self.lai_source.has(grid_id, date):
                lai_bands      = self.lai_source.load(grid_id, date)
                lai_r0, lai_c0 = self.lai_source.patch_origin(lat, lon)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lai_aligned = self.lai_source.reproject(lai_bands, lai_r0, lai_c0, target_grid)
                lai_arr    = lai_aligned["lai"]     # (H, W), NaN where fill/out-of-range
                lai_qc     = lai_aligned["lai_qc"]  # (H, W), 1=good quality, 0=excluded
                lai_valid  = np.isfinite(lai_arr) & (lai_qc > 0.5)
                target_lai = np.where(lai_valid, lai_arr, 0.0).astype(np.float32)
                lai_mask   = lai_valid.astype(np.float32)

                if self.lai_source.has_fpar:
                    fpar_arr     = lai_aligned["fpar"]
                    fpar_qc      = lai_aligned["fpar_qc"]
                    fpar_valid   = np.isfinite(fpar_arr) & (fpar_qc > 0.5)
                    target_fpar  = np.where(fpar_valid, fpar_arr, 0.0).astype(np.float32)
                    fpar_mask    = fpar_valid.astype(np.float32)
            else:
                target_lai = np.zeros((H, W), dtype=np.float32)
                lai_mask   = np.zeros((H, W), dtype=np.float32)
                if self.lai_source.has_fpar:
                    target_fpar = np.zeros((H, W), dtype=np.float32)
                    fpar_mask   = np.zeros((H, W), dtype=np.float32)

            sample["target_lai"] = _to_tensor(target_lai)
            sample["lai_mask"]   = _to_tensor(lai_mask)
            if self.lai_source.has_fpar:
                sample["target_fpar"] = _to_tensor(target_fpar)
                sample["fpar_mask"]   = _to_tensor(fpar_mask)

        # MOD13A3 monthly NDVI background — optional; (1, 128, 128) float32
        if self.mod13a3_source is not None:
            monthly = self.mod13a3_source.load(grid_id, date)
            sample["ndvi_monthly"] = torch.from_numpy(
                monthly["ndvi_monthly"][np.newaxis].astype(np.float32)
            )

        return sample

    def _build_mtg_conditioning(
        self,
        fci_store,
        ts_rows: dict[str, int],
        target_grid: TargetGrid,
    ) -> torch.Tensor:
        """Build the multi-temporal FCI conditioning tensor.

        Returns
        -------
        mtg_conditioning : torch.Tensor, shape (18, H, W), float32
            Channels 0–14 : 5 timestamps x [vis06, vis08, cos_sza] per observation.
            Channels 15–17: [NDVI75, NDVI_std, CloudScore] composites.
        """
        H = W = target_grid.n_px
        timestamps = sorted(ts_rows.keys())

        # Per-pixel lat/lon on the AEQD grid — used for cos_sza computation
        lat_deg, lon_deg = _aeqd_to_latlon(target_grid)

        # FCI fractional coords — patch_origins are invariant across timestamps
        # for the same grid_id (fixed geographic footprint); read once.
        first_ts  = timestamps[0]
        first_row = ts_rows[first_ts]
        r0, c0 = fci_store["patches"]["vis_06"][first_ts]["patch_origins"][first_row]
        frac_row_fci, frac_col_fci = _latlon_to_fci_px(
            lat_deg, lon_deg, int(r0), int(c0),
            fci_patch_h=128, fci_patch_w=128,
        )

        obs_frames: list[np.ndarray]  = []
        ndvi_frames: list[np.ndarray] = []

        for ts in timestamps[:5]:
            row = ts_rows[ts]

            vis06_raw = fci_store["patches"]["vis_06"][ts]["data"][row].astype(np.float32)
            vis08_raw = fci_store["patches"]["vis_08"][ts]["data"][row].astype(np.float32)

            vis06 = _sample_array(vis06_raw, frac_row_fci, frac_col_fci)
            vis08 = _sample_array(vis08_raw, frac_row_fci, frac_col_fci)

            # NDVI: FCI stores reflectances in percent (0–100+ scale).
            # Require both channels ≥ 2% to mask near-zero-red artifacts
            # (dark pixels where atmospheric absorption of red >> NIR produce
            # spuriously high NDVI values when vis06 is 0.02–1%).
            valid = (vis06 >= 2.0) & (vis08 >= 2.0)
            denom = vis08 + vis06
            ndvi = np.where(
                valid & (denom > 0.0),
                np.clip((vis08 - vis06) / denom, -1.0, 1.0),
                0.0,
            ).astype(np.float32)
            ndvi_frames.append(ndvi)

            dt_utc  = datetime.fromisoformat(ts)
            cos_sza = _compute_cos_sza(lat_deg, lon_deg, dt_utc)
            cos_sza = np.where(
                np.isfinite(cos_sza), np.clip(cos_sza, 0.0, 1.0), 0.0
            ).astype(np.float32)

            obs_frames.append(np.stack([vis06, vis08, cos_sza], axis=0))

        zero_frame = np.zeros((3, H, W), dtype=np.float32)
        while len(obs_frames) < 5:
            obs_frames.append(zero_frame)
            ndvi_frames.append(np.zeros((H, W), dtype=np.float32))

        mtg_stack   = np.stack(obs_frames, axis=0)   # (5, 3, H, W)
        ndvi_stack  = np.stack(ndvi_frames, axis=0)  # (5, H, W)
        vis08_stack = mtg_stack[:, 1]                # (5, H, W)

        raw_channels = mtg_stack.reshape(5 * 3, H, W)              # (15, H, W)
        composites   = compute_mtg_composites(ndvi_stack, vis08_stack)  # (3, H, W)

        mtg_conditioning = np.concatenate(
            [raw_channels, composites], axis=0
        )  # (18, H, W)

        return torch.from_numpy(np.ascontiguousarray(mtg_conditioning))


# ---------------------------------------------------------------------------
# 7.  build_sample_index
# ---------------------------------------------------------------------------

def _fci_coverage_filter(
    fci_store,
    max_nan_fraction: float,
    sample_band: str = "vis_06",
) -> set[str]:
    """Return grid IDs whose MTG FCI NaN fraction ≤ max_nan_fraction.

    Evaluates coverage from the first available timestamp (the geostationary
    disk edge is fixed, so NaN patterns are stable across time).
    Loads data in chunks to avoid a single large allocation.
    """
    timestamps = sorted(fci_store["patches"][sample_band].keys())
    if not timestamps:
        return set()
    ts   = timestamps[0]
    data = fci_store["patches"][sample_band][ts]["data"]   # zarr (N, H, W)
    gids = [str(g) for g in fci_store["patches"][sample_band][ts]["grid_ids"][:]]
    N    = data.shape[0]

    valid: set[str] = set()
    chunk = 2000
    for start in range(0, N, chunk):
        end        = min(start + chunk, N)
        block      = np.array(data[start:end])              # (chunk, H, W)
        nan_fracs  = np.isnan(block).mean(axis=(1, 2))
        for i, nan_frac in enumerate(nan_fracs):
            if nan_frac <= max_nan_fraction:
                valid.add(gids[start + i])

    n_removed = N - len(valid)
    _LOG.info(
        "MTG FCI coverage filter (max_nan=%.0f%%): kept %d / %d grid IDs "
        "(removed %d = %.1f%%)",
        max_nan_fraction * 100,
        len(valid), N, n_removed, 100.0 * n_removed / max(N, 1),
    )
    return valid


def build_sample_index(
    fci_store_path: Union[str, Path],
    ancillary_sources: dict[str, AncillarySource],
    dates: list[str] | None = None,
    max_fci_nan_fraction: float = 1.0,
    max_ndvi_lap_var: float = 0.0,
) -> list[tuple[str, str]]:
    """Return all ``(grid_id, date)`` pairs for which every source has data.

    Parameters
    ----------
    fci_store_path :
        Path to the MTG FCI zarr store.
    ancillary_sources :
        Same dict passed to ``AlignedPatchDataset``.
    dates :
        Restrict to these ISO date strings.  If None, use all FCI dates.
    max_fci_nan_fraction :
        Drop grid IDs where the MTG FCI NaN fraction exceeds this value.
        Default 1.0 = no filtering (backward compatible).
        Recommended: 0.10 to remove tiles outside the geostationary disk
        (typically ~13% of tiles, mostly Latin America / far east Pacific).
    max_ndvi_lap_var :
        Drop patches where ``var(laplace(ndvi_envelope[mask]))`` exceeds this
        value (mask = soft_score > 0.5).  Computed at full 256×256 resolution
        so pixel-level salt-and-pepper noise is caught (unlike downsampled
        variants).  0.0 = disabled (default, backward compatible).
        Recommended: 0.05 — cleanly separates coherent scenes (0.017–0.022)
        from noise patches (≥ 0.099).

    Notes
    -----
    Sources that return a non-empty ``available_dates()`` set are used to
    further restrict the date range (e.g. ERA5Source needs n_days of
    look-back history before the first sample date).  Sources that return
    a non-empty ``available_grid_ids(date)`` set restrict which grid IDs
    are included for that date.
    """
    fci_store = zarr.open(str(fci_store_path), mode="r")

    # Build coverage filter (computed once; stable across timestamps)
    fci_valid_gids: set[str] | None = None
    if max_fci_nan_fraction < 1.0:
        fci_valid_gids = _fci_coverage_filter(fci_store, max_fci_nan_fraction)

    # Collect date restrictions from all sources
    allowed_dates: set[str] | None = None
    for source in ancillary_sources.values():
        src_dates = source.available_dates()
        if src_dates:
            allowed_dates = src_dates if allowed_dates is None else (allowed_dates & src_dates)

    fci_by_date: dict[str, set[str]] = {}
    for ts_key in fci_store["patches"]["vis_06"].keys():
        date = ts_key[:10]
        if dates is not None and date not in dates:
            continue
        if allowed_dates is not None and date not in allowed_dates:
            continue
        gids = [str(g) for g in fci_store["patches"]["vis_06"][ts_key]["grid_ids"][:]]
        fci_by_date.setdefault(date, set()).update(gids)

    samples: list[tuple[str, str]] = []
    for date, fci_gids in sorted(fci_by_date.items()):
        available = fci_gids
        if fci_valid_gids is not None:
            available = available & fci_valid_gids
        for source in ancillary_sources.values():
            src_gids = source.available_grid_ids(date)
            if src_gids:
                available = available & src_gids
        for gid in sorted(available):
            samples.append((gid, date))

    if max_ndvi_lap_var > 0.0:
        # Find the first MODISSource with a processed zarr (ndvi_envelope + soft_score)
        proc_store = None
        for source in ancillary_sources.values():
            if hasattr(source, "_processed_path"):
                proc_store = zarr.open(source._processed_path, mode="r")
                break
        if proc_store is not None:
            nd_grp = proc_store["patches"]["ndvi_envelope"]
            ss_grp = proc_store["patches"]["soft_score"]
            kept, n_dropped = [], 0
            for gid, date in samples:
                try:
                    ndvi = nd_grp[date][gid][:].astype(np.float32)
                    soft = ss_grp[date][gid][:]
                    mask = soft > 0.5
                    if mask.sum() < 100:
                        kept.append((gid, date))
                        continue
                    lap_var = float(np.var(_ndimage_laplace(ndvi)[mask]))
                    if lap_var <= max_ndvi_lap_var:
                        kept.append((gid, date))
                    else:
                        n_dropped += 1
                except Exception:
                    kept.append((gid, date))
            _LOG.info(
                "Laplacian filter (max_ndvi_lap_var=%.4f): dropped %d / %d samples",
                max_ndvi_lap_var, n_dropped, len(samples),
            )
            samples = kept

    return samples
