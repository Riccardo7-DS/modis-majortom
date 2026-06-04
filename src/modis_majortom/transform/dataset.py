"""
dataset.py
==========
Spatially-aligned multi-source PyTorch Dataset for MTG FCI + MODIS.

Each sample exposes two tensors on a common Earth-fixed AEQD grid centred on
the MajorTOM cell:

    X : (6, 3, H, W) float32 — MTG FCI multi-temporal conditioning tensor
        Frames 0-4 : per-observation [vis06, vis08, cos_sza]
        Frame 5    : [ndvi_max, ndvi_max, ndvi_max] — per-pixel max NDVI
                     across all intra-day observations

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
from scipy.ndimage import map_coordinates
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

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0.  Solar geometry helper
# ---------------------------------------------------------------------------

def _compute_cos_sza(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    dt: datetime,
) -> np.ndarray:
    """Vectorised cos(solar zenith angle) via the Spencer approximation (~0.5° accuracy)."""
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
    ``patches/sur_refl_b02`` (MOD09GQ bands).  ``ndvi_observed`` is computed
    on-the-fly from these so it never needs to be pre-stored.

    ``processed_zarr_path`` must contain the Whittaker pipeline outputs:
    ``patches/ndvi_smoothed`` and ``patches/soft_score`` (written by
    ``WhittakerPipeline.process_zarr``).

    Both stores are opened once per thread and cached in thread-local storage.
    """

    PATCH_PX: int = 512

    def __init__(
        self,
        raw_zarr_path: Union[str, Path],
        processed_zarr_path: Union[str, Path],
    ) -> None:
        self._raw_path       = str(raw_zarr_path)
        self._processed_path = str(processed_zarr_path)
        self._local          = threading.local()

    @property
    def _raw_store(self):
        if not hasattr(self._local, "raw_store"):
            self._local.raw_store = zarr.open(self._raw_path, mode="r")
        return self._local.raw_store

    @property
    def _processed_store(self):
        if not hasattr(self._local, "processed_store"):
            self._local.processed_store = zarr.open(self._processed_path, mode="r")
        return self._local.processed_store

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        raw  = self._raw_store["patches"]
        proc = self._processed_store["patches"]

        # ndvi_observed: computed on-the-fly — no need to pre-store a derived quantity
        b01      = raw["sur_refl_b01"][date][grid_id][:].astype(np.float32)
        b02      = raw["sur_refl_b02"][date][grid_id][:].astype(np.float32)
        ndvi_obs = np.asarray(compute_ndvi(b02, b01, fill_below=-0.05), dtype=np.float32)

        ndvi_whi = proc["ndvi_envelope"][date][grid_id][:].astype(np.float32)
        soft     = proc["soft_score"][date][grid_id][:].astype(np.float32)
        return {"ndvi_observed": ndvi_obs, "ndvi_whittaker": ndvi_whi, "soft_score": soft}

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
        return set(self._processed_store["patches"]["ndvi_smoothed"].keys())

    def available_grid_ids(self, date: str) -> set[str]:
        try:
            return set(self._processed_store["patches"]["ndvi_smoothed"][date].keys())
        except KeyError:
            return set()


# ---------------------------------------------------------------------------
# 3.  ERA5Source  (skeleton — implement when ERA5 zarr is available)
# ---------------------------------------------------------------------------

class ERA5Source(AncillarySource):
    """Wraps an ERA5 zarr on a regular WGS-84 lat/lon grid."""

    def __init__(
        self,
        zarr_path: Union[str, Path],
        variables: list[str],
        lat_origin: float = 90.0,
        lon_origin: float = -180.0,
        lat_spacing: float = 0.25,
        lon_spacing: float = 0.25,
    ) -> None:
        self._zarr_path  = str(zarr_path)
        self.variables   = variables
        self.lat_origin  = lat_origin
        self.lon_origin  = lon_origin
        self.lat_spacing = lat_spacing
        self.lon_spacing = lon_spacing

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        store = zarr.open(self._zarr_path, mode="r")
        return {v: store[v][date][:].astype(np.float32) for v in self.variables}

    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        return (0, 0)

    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid: TargetGrid,
    ) -> dict[str, np.ndarray]:
        lat, lon = _aeqd_to_latlon(target_grid)
        frac_row = (self.lat_origin - lat) / self.lat_spacing
        frac_col = (lon - self.lon_origin) / self.lon_spacing

        out: dict[str, np.ndarray] = {}
        for name, arr in bands.items():
            result = np.full(frac_row.shape, np.nan, dtype=np.float32)
            valid = (
                np.isfinite(frac_row) & np.isfinite(frac_col)
                & (frac_row >= 0) & (frac_row <= arr.shape[0] - 1)
                & (frac_col >= 0) & (frac_col <= arr.shape[1] - 1)
            )
            if valid.any():
                result[valid] = map_coordinates(
                    arr.astype(np.float32),
                    [frac_row[valid], frac_col[valid]],
                    order=1, mode="constant", cval=np.nan, prefilter=False,
                )
            out[name] = result
        return out

    def available_dates(self) -> set[str]:
        store = zarr.open(self._zarr_path, mode="r")
        return set(store[self.variables[0]].keys()) if self.variables else set()

    def available_grid_ids(self, date: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# 4.  AlignedPatchDataset
# ---------------------------------------------------------------------------

class AlignedPatchDataset(Dataset):
    """Spatially-aligned MTG FCI + MODIS PyTorch Dataset.

    Each sample returns two tensors on a shared AEQD grid:

    ``X`` : (6, 3, H, W) float32
        Multi-temporal FCI conditioning tensor.
        Frames 0–4: per-observation ``[vis06, vis08, cos_sza]`` (chronological;
        zero-padded if fewer than 5 timestamps are available).
        Frame 5:    ``[ndvi_max, ndvi_max, ndvi_max]`` — per-pixel max NDVI
        across all observations, replicated across 3 channel slots.

    ``target_ndvi`` : (1, H, W) float32
        Blended NDVI target: ``soft_score * ndvi_observed + (1−soft_score) * ndvi_whittaker``,
        clamped to [−1, 1].  NaN pixels fall back to ``ndvi_whittaker``; then 0.
    ``loss_weight`` : (1, H, W) float32
        Per-pixel training weight: ``soft_score + alpha * (1−soft_score)``.
        Full weight for clear pixels, reduced (alpha-scaled) for cloudy/gap-filled pixels.
    ``meta_mask`` : (1, H, W) float32
        Reliable-observation flag: ``soft_score`` where ``soft_score ≥ 0.5``, else 0.

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
    """

    def __init__(
        self,
        fci_store_path: Union[str, Path],
        ancillary_sources: dict[str, AncillarySource],
        samples: list[tuple[str, str]],
        target_n_pixels: int = 256,
        alpha: float = 0.2,
    ) -> None:
        if "modis" not in ancillary_sources:
            raise ValueError("ancillary_sources must include a 'modis' key (MODISSource)")
        self._fci_store_path   = str(fci_store_path)
        self.ancillary_sources = ancillary_sources
        self.samples           = list(samples)
        self.target_n_pixels   = target_n_pixels
        self._alpha            = alpha
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

        # X: multi-temporal FCI conditioning tensor (6, 3, H, W)
        X = self._build_mtg_conditioning(fci_store, ts_rows, target_grid)

        # Load and reproject three MODIS NDVI-related arrays onto the shared AEQD grid
        modis_src      = self.ancillary_sources["modis"]
        modis_bands    = modis_src.load(grid_id, date)
        mod_r0, mod_c0 = modis_src.patch_origin(lat, lon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligned = modis_src.reproject(modis_bands, mod_r0, mod_c0, target_grid)

        ndvi_obs = aligned["ndvi_observed"]   # (H, W) float32
        ndvi_whi = aligned["ndvi_whittaker"]  # (H, W) float32
        soft     = aligned["soft_score"]      # (H, W) float32

        # Clamp soft_score to [0, 1]; treat NaN as fully uncertain (0)
        soft = np.clip(np.nan_to_num(soft, nan=0.0), 0.0, 1.0)

        # Blend: confident-clear pixels use observed NDVI; uncertain pixels use Whittaker
        target_ndvi = soft * ndvi_obs + (1.0 - soft) * ndvi_whi
        target_ndvi = np.clip(target_ndvi, -1.0, 1.0)

        # Fill any NaN that survived blending (both ndvi_obs and ndvi_whi were NaN)
        nan_mask = np.isnan(target_ndvi)
        if nan_mask.any():
            target_ndvi[nan_mask] = ndvi_whi[nan_mask]
            still_nan = np.isnan(target_ndvi)
            if still_nan.any():
                n_nan = int(still_nan.sum())
                _LOG.warning(
                    "target_ndvi has %d NaN pixels after Whittaker fallback "
                    "(grid_id=%r, date=%r); setting to 0",
                    n_nan, grid_id, date,
                )
                target_ndvi[still_nan] = 0.0

        # Full weight for clear observations; alpha-scaled floor for cloudy/gap-filled pixels
        loss_weight = soft + self._alpha * (1.0 - soft)

        # Conditioning mask: pass soft_score through only where it signals reliable obs
        meta_mask = np.where(soft >= 0.5, soft, 0.0).astype(np.float32)

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[np.newaxis].astype(np.float32))  # (1, H, W)

        return {
            "X":           X,
            "target_ndvi": _to_tensor(target_ndvi),
            "loss_weight": _to_tensor(loss_weight),
            "meta_mask":   _to_tensor(meta_mask),
            "grid_id":     grid_id,
            "date":        date,
            "cell_lat":    lat,
            "cell_lon":    lon,
        }

    def _build_mtg_conditioning(
        self,
        fci_store,
        ts_rows: dict[str, int],
        target_grid: TargetGrid,
    ) -> torch.Tensor:
        """Build the multi-temporal FCI conditioning tensor.

        Returns
        -------
        mtg_conditioning : torch.Tensor, shape (6, 3, H, W), float32
            Frames 0–4 : [vis06, vis08, cos_sza] per observation.
            Frame 5    : [ndvi_max, ndvi_max, ndvi_max].
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

            # NDVI: guard near-zero denominators (cos(SZA)-normalised reflectances
            # produce large extremes at dawn/dusk); clip to [-1, 1].
            denom = vis08 + vis06
            ndvi = np.where(
                np.abs(denom) > 1e-3,
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

        mtg_stack = np.stack(obs_frames, axis=0)  # (5, 3, H, W)
        ndvi_max  = np.stack(ndvi_frames, axis=0).max(axis=0, keepdims=True)  # (1, H, W)

        ndvi_max_frame = np.broadcast_to(
            ndvi_max[:, np.newaxis, :, :], (1, 3, H, W)
        )
        mtg_conditioning = np.concatenate(
            [mtg_stack, np.ascontiguousarray(ndvi_max_frame)], axis=0
        )  # (6, 3, H, W)

        return torch.from_numpy(mtg_conditioning)


# ---------------------------------------------------------------------------
# 5.  build_sample_index
# ---------------------------------------------------------------------------

def build_sample_index(
    fci_store_path: Union[str, Path],
    ancillary_sources: dict[str, AncillarySource],
    dates: list[str] | None = None,
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
    """
    fci_store = zarr.open(str(fci_store_path), mode="r")

    fci_by_date: dict[str, set[str]] = {}
    for ts_key in fci_store["patches"]["vis_06"].keys():
        date = ts_key[:10]
        if dates is not None and date not in dates:
            continue
        gids = [str(g) for g in fci_store["patches"]["vis_06"][ts_key]["grid_ids"][:]]
        fci_by_date.setdefault(date, set()).update(gids)

    samples: list[tuple[str, str]] = []
    for date, fci_gids in sorted(fci_by_date.items()):
        available = fci_gids
        for source in ancillary_sources.values():
            src_gids = source.available_grid_ids(date)
            if src_gids:
                available = available & src_gids
        for gid in sorted(available):
            samples.append((gid, date))

    return samples
