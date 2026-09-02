"""
dataset.py
==========
Ancillary data source loaders for MTG FCI + MODIS spatial alignment.

Defines the ``AncillarySource`` interface and the concrete numpy/zarr/xarray
loaders (``MODISSource``, ``RawGAMODISSource``, ``MOD13A3Source``,
``ERA5Source``, ``LAISource``) used to build spatially-aligned training
samples. These loaders have no torch dependency so they can be imported from
a lightweight ``modis-majortom`` install.

The torch-dependent consumer of these sources — ``AlignedPatchDataset`` (a
``torch.utils.data.Dataset``) and its companion ``build_sample_index`` — now
lives in the ``ndvi-diffusion`` sibling package at
``ndvi_diffusion.datasets.patch_dataset``, which imports the classes below
as an external ``modis_majortom`` dependency.

Pixel [i, j] in the aligned tensors produced downstream refers to the same
point on the ground. Geometry and reprojection are handled by
``eumetsearch.transform``.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

from eumetsearch.transform import (
    TargetGrid,
    modis_patch_corner_for_cell,
    grid_id_to_latlon,
)
from eumetsearch.transform.fci_modis_align import (
    _aeqd_to_latlon,
    _latlon_to_modis_px,
    _sample_array,
)
from ..utils import compute_ndvi
from .cloud_adapter import upsample_500_to_250


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
# 6.  LAISource
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
