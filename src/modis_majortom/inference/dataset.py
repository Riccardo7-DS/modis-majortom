"""Inference-time dataset and sample-index builder (FCI + ERA5, no MODIS required)."""
from __future__ import annotations

import threading
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from eumetsearch.transform import (
    TargetGrid,
    MODISGeometry,
    grid_id_to_latlon,
)
from eumetsearch.transform.fci_modis_align import (
    _aeqd_to_latlon,
    _latlon_to_fci_px,
    _sample_array,
)
from eumetsearch.transform.ndvi import _build_fci_index

from ..transform.dataset import ERA5Source
from ..transform.land_cover import LandCoverSource
from ..utils.solar import compute_cos_sza


class InferencePatchDataset(Dataset):
    """Like AlignedPatchDataset but does not require MODIS.

    Returns a dict with keys:
        ``X``           (6, 3, H, W) float32 — MTG FCI conditioning tensor
        ``era5``        (N_vars, n_days, H, W) float32 — ERA5 features
        ``land_cover``  (1, H, W) float32 — LC_Type1 if lc_source provided, else absent
        ``grid_id``     str
        ``date``        str  (YYYY-MM-DD)
        ``cell_lat``    float
        ``cell_lon``    float
    """

    def __init__(
        self,
        fci_store_path: str | Path,
        era5_source: ERA5Source,
        samples: list[tuple[str, str]],
        target_n_pixels: int = 256,
        lc_source: LandCoverSource | None = None,
    ) -> None:
        self._fci_store_path = str(fci_store_path)
        self.era5_source = era5_source
        self.samples = list(samples)
        self.target_n_pixels = target_n_pixels
        self.lc_source = lc_source
        self._local = threading.local()

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def _fci_store(self) -> zarr.Group:
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
        ts_rows = fci_index[grid_id]

        target_grid = TargetGrid(
            cell_lat=lat,
            cell_lon=lon,
            extent_m=MODISGeometry.PATCH_EXTENT_M / 2,
            n_pixels=self.target_n_pixels,
        )

        X = self._build_mtg_conditioning(fci_store, ts_rows, target_grid)

        era5_bands = self.era5_source.load(grid_id, date)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            era5_aligned = self.era5_source.reproject(era5_bands, 0, 0, target_grid)
        era5_stack = np.stack(
            [era5_aligned[v] for v in self.era5_source.variables], axis=0
        )  # (N_vars, n_days, H, W)

        sample = {
            "X": X,
            "era5": torch.from_numpy(era5_stack.astype(np.float32)),
            "grid_id": grid_id,
            "date": date,
            "cell_lat": lat,
            "cell_lon": lon,
        }

        if self.lc_source is not None:
            lc_bands = self.lc_source.load(grid_id, date)
            lc_aligned = self.lc_source.reproject(lc_bands, 0, 0, target_grid)
            lc_stack = np.stack([lc_aligned[v] for v in self.lc_source.variables], axis=0)
            sample["land_cover"] = torch.from_numpy(lc_stack.astype(np.float32))

        return sample

    def _build_mtg_conditioning(
        self,
        fci_store: zarr.Group,
        ts_rows: dict,
        target_grid: TargetGrid,
    ) -> torch.Tensor:
        """Build the (6, 3, H, W) MTG conditioning tensor for one cell/date."""
        H = W = target_grid.n_px
        timestamps = sorted(ts_rows.keys())
        lat_deg, lon_deg = _aeqd_to_latlon(target_grid)

        first_ts = timestamps[0]
        first_row = ts_rows[first_ts]
        r0, c0 = fci_store["patches"]["vis_06"][first_ts]["patch_origins"][first_row]
        frac_r, frac_c = _latlon_to_fci_px(
            lat_deg, lon_deg, int(r0), int(c0), fci_patch_h=128, fci_patch_w=128,
        )

        obs_frames: list[np.ndarray] = []
        ndvi_frames: list[np.ndarray] = []

        for ts in timestamps[:5]:
            row = ts_rows[ts]
            vis06 = _sample_array(
                fci_store["patches"]["vis_06"][ts]["data"][row].astype(np.float32),
                frac_r, frac_c,
            )
            vis08 = _sample_array(
                fci_store["patches"]["vis_08"][ts]["data"][row].astype(np.float32),
                frac_r, frac_c,
            )
            valid = (vis06 >= 2.0) & (vis08 >= 2.0)
            denom = vis08 + vis06
            ndvi = np.where(
                valid & (denom > 0), np.clip((vis08 - vis06) / denom, -1, 1), 0.0
            ).astype(np.float32)
            ndvi_frames.append(ndvi)

            c = compute_cos_sza(lat_deg, lon_deg, datetime.fromisoformat(ts))
            cos_sza = np.where(np.isfinite(c), np.clip(c, 0, 1), 0.0).astype(np.float32)
            obs_frames.append(np.stack([vis06, vis08, cos_sza], axis=0))

        zero = np.zeros((3, H, W), dtype=np.float32)
        while len(obs_frames) < 5:
            obs_frames.append(zero)
            ndvi_frames.append(np.zeros((H, W), dtype=np.float32))

        mtg_stack = np.stack(obs_frames, axis=0)  # (5, 3, H, W)
        ndvi_max = np.percentile(np.stack(ndvi_frames), 75, axis=0, keepdims=True)  # (1, H, W)
        ndvi_frame = np.broadcast_to(ndvi_max[:, np.newaxis], (1, 3, H, W))
        return torch.from_numpy(
            np.concatenate([mtg_stack, np.ascontiguousarray(ndvi_frame)], axis=0)
        )  # (6, 3, H, W)


def build_inference_index(
    fci_store_path: str | Path,
    era5_source: ERA5Source,
    skip_set: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Return (grid_id, date) pairs where both FCI and ERA5 have data.

    Args:
        fci_store_path: Path to the MTG FCI zarr store.
        era5_source: ERA5Source instance; used to get the set of available dates.
        skip_set: Optional set of (grid_id, date) pairs to exclude (e.g. already written).

    Returns:
        Sorted list of (grid_id, date) tuples.
    """
    fci_store = zarr.open(str(fci_store_path), mode="r")
    allowed_dates = era5_source.available_dates()

    fci_by_date: dict[str, set[str]] = {}
    for ts_key in fci_store["patches"]["vis_06"].keys():
        date = ts_key[:10]
        if date not in allowed_dates:
            continue
        gids = [str(g) for g in fci_store["patches"]["vis_06"][ts_key]["grid_ids"][:]]
        fci_by_date.setdefault(date, set()).update(gids)

    samples = []
    for date, gids in sorted(fci_by_date.items()):
        for gid in sorted(gids):
            if skip_set and (gid, date) in skip_set:
                continue
            samples.append((gid, date))
    return samples
