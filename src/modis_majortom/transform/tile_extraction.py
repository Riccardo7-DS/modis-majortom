"""
Tile extraction and synchronous dataset for MTG FCI + MODIS MOD09GQ.

Provides two main components:

1.  ``extract_clean_tiles`` — iterates over every Major TOM grid cell and every
    time step stored in a MODGA zarr archive, applies a cloud-fraction filter,
    and returns lightweight ``TileDescriptor`` records consumed later by a loader.

2.  ``MTGMODISDataset`` — a map-style dataset that pairs a daily MTG max-value
    composite with the same-day MOD09GQ patch for every (grid_id, date) pair
    present in both stores.

Expected MODGA zarr layout
--------------------------
modga.zarr/
    patches/
        cloud_mask/
            <grid_id>   →  zarr array  (T, H₅₀₀, W₅₀₀)  float32
                           soft cloud score in [0, 1]
                           1 = confident clear, 0 = confident cloud
        blue_band/
            <grid_id>   →  zarr array  (T, H₅₀₀, W₅₀₀)  float32
    dates               →  zarr array  (T,)  str "YYYY-MM-DD"

MOD09GQ zarr layout (250 m, used by MTGMODISDataset)
-----------------------------------------------------
modgq.zarr/
    patches/
        sur_refl_b01/
            <date>/              "YYYY-MM-DD"
                <grid_id>  →  zarr array  (H_mod, W_mod)  float16
        sur_refl_b02/  (same layout)
        QC_250m/       (same layout, uint8)

MTG FCI zarr layout (used by MTGMODISDataset)
---------------------------------------------
mtg.zarr/
    patches/
        vis_06/
            <ISO timestamp>/     e.g. "2025-06-01T09:00:00"
                data:          (N, H_mtg, W_mtg)  float16
                grid_ids:      (N,)                str  MajorTOM ids
                patch_origins: (N, 2)               int32
        vis_08/  (same layout as vis_06)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import zarr

logger = logging.getLogger(__name__)


# ── Descriptor ────────────────────────────────────────────────────────────────

@dataclass
class TileDescriptor:
    """
    Lightweight, array-free record for one cloud-filtered (tile, date) pair.

    Attributes
    ----------
    row, col:
        Integer Major TOM grid indices parsed from ``grid_id``.
    grid_id:
        Original string key used in the zarr store (e.g. ``"266D_764L"``).
        Pass this directly to ``zarr_store["patches"]["cloud_mask"][grid_id]``
        to retrieve the corresponding (T, H, W) array.
    date:
        ISO date string ``"YYYY-MM-DD"`` for this observation.
    time_index:
        Integer position along the T axis of the (T, H, W) zarr arrays.
        Use ``tile_array[time_index]`` to load the (H, W) slice.
    cloud_fraction:
        Fraction of pixels in the tile where ``soft_score < 0.5``.
        Guaranteed to be ≤ the ``max_cloud_fraction`` threshold passed to
        :func:`extract_clean_tiles`.
    month:
        Calendar month (1–12) of ``date``, pre-computed for stratified sampling.
    """
    row:            int
    col:            int
    grid_id:        str
    date:           str
    time_index:     int
    cloud_fraction: float
    month:          int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_row_col(grid_id: str) -> tuple[int, int]:
    """
    Extract (row, col) integer indices from a Major TOM grid_id string.

    Handles formats such as:
      ``"266D_764L"``  →  (266, 764)
      ``"0266_0764"``  →  (266, 764)
      ``"266_764"``    →  (266, 764)

    The leading integer in the first segment is taken as row; the leading
    integer in the second segment is taken as col.  Any letter suffixes (e.g.
    ``D``, ``L``) are stripped.

    Raises
    ------
    ValueError
        If the grid_id cannot be split into at least two ``_``-separated parts,
        or if either part contains no digits.
    """
    parts = grid_id.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot parse (row, col) from grid_id {grid_id!r}. "
            "Expected a string with at least one '_' separator, e.g. '266D_764L'."
        )
    m_row = re.search(r"\d+", parts[0])
    m_col = re.search(r"\d+", parts[1])
    if m_row is None or m_col is None:
        raise ValueError(
            f"Could not find integer digits in grid_id segments "
            f"{parts[0]!r} / {parts[1]!r}."
        )
    return int(m_row.group()), int(m_col.group())


def _normalise_date_array(raw: np.ndarray) -> list[str]:
    """
    Convert a zarr-loaded dates array to a list of plain ``"YYYY-MM-DD"`` strings.

    Handles:
      - ``numpy.bytes_`` / ``bytes`` elements (zarr variable-length string dtype)
      - ``numpy.str_`` / ``str`` elements
      - ``numpy.datetime64`` elements (truncated to day precision)
    """
    dates: list[str] = []
    for d in raw:
        if isinstance(d, (bytes, np.bytes_)):
            dates.append(d.decode())
        elif isinstance(d, np.datetime64):
            # numpy.datetime64 → "YYYY-MM-DD" (first 10 chars of ISO repr)
            dates.append(str(d)[:10])
        else:
            dates.append(str(d)[:10])
    return dates


# ── Main function ─────────────────────────────────────────────────────────────

def extract_clean_tiles(
    modga_path: Union[str, Path],
    modgq_path: Union[str, Path],
    mtg_path: Union[str, Path],
    era5_path: Union[str, Path],
    max_cloud_fraction: float = 0.3,
) -> list[TileDescriptor]:
    """
    Iterate over all Major TOM tiles and time steps; retain those where the
    fraction of soft-cloudy pixels does not exceed ``max_cloud_fraction``.

    Cloud fraction is defined as the fraction of pixels in a tile where the
    soft cloud-confidence score is below 0.5.  The soft score is stored in the
    MODGA ``cloud_mask`` array with 1 = confident clear and 0 = confident cloud.
    Pixels with NaN values (fill / no-data) are treated as clear and do **not**
    contribute to the cloud fraction.

    Only the MODGA zarr is read.  MODGQ, MTG, and ERA5 paths are accepted for
    API completeness; the DataLoader will use them via the :class:`TileDescriptor`
    records returned here.

    Parameters
    ----------
    modga_path:
        Path to the MODGA 500 m zarr store (see module docstring for layout).
    modgq_path:
        Path to the MODGQ 250 m zarr store.  Not read here; the same
        ``grid_id`` present in each descriptor can be used to access the
        corresponding (T, 2·H, 2·W) NDVI array in this store.
    mtg_path:
        Path to the MTG FCI zarr store (not used at this stage).
    era5_path:
        Path to the ERA5 zarr store (not used at this stage).
    max_cloud_fraction:
        Upper bound on the fraction of pixels with ``soft_score < 0.5``.
        Tiles exceeding this threshold are discarded.  Default 0.3.

    Returns
    -------
    list[TileDescriptor]
        One entry per (tile, date) pair that passes the cloud fraction filter,
        ordered by grid_id then time index.  Empty if every pair is rejected.
    """
    # ── Open the MODGA zarr store (read-only, lazy) ───────────────────────────
    # Accept both filesystem paths (str / Path) and zarr Store objects directly.
    store_arg = modga_path if not isinstance(modga_path, Path) else str(modga_path)
    modga_store = zarr.open(store_arg, mode="r")

    # Navigate to the cloud_mask group: grid_id → (T, H, W) array
    try:
        cloud_grp = modga_store["patches"]["cloud_mask"]
    except KeyError as exc:
        raise KeyError(
            f"Expected 'patches/cloud_mask' group inside {modga_path!r}. "
            f"Got: {exc}.  Check that the MODGA zarr layout matches the "
            "documented structure."
        ) from exc

    # ── Load the shared dates coordinate (small, safe to materialise) ─────────
    try:
        raw_dates = modga_store["dates"][:]
    except KeyError as exc:
        raise KeyError(
            f"Expected a top-level 'dates' array inside {modga_path!r}. "
            f"Got: {exc}."
        ) from exc

    dates: list[str] = _normalise_date_array(raw_dates)
    T = len(dates)

    # Pre-compute calendar month per time index to avoid string parsing in the loop
    months: list[int] = [int(d[5:7]) for d in dates]  # "YYYY-MM-DD"[5:7] = "MM"

    # ── Enumerate all grid cells ───────────────────────────────────────────────
    grid_ids: list[str] = sorted(cloud_grp.keys())
    n_grids = len(grid_ids)
    n_total = n_grids * T

    logger.info(
        "extract_clean_tiles: %d grid cells × %d time steps = %d (tile, date) pairs",
        n_grids, T, n_total,
    )
    logger.info(
        "Cloud fraction threshold: %.2f  (soft_score < 0.5 treated as cloudy)",
        max_cloud_fraction,
    )

    # ── Main filtering loop ────────────────────────────────────────────────────
    descriptors: list[TileDescriptor] = []

    # Per-month counters for the summary log
    month_total:   defaultdict[int, int] = defaultdict(int)
    month_passing: defaultdict[int, int] = defaultdict(int)

    for grid_idx, grid_id in enumerate(grid_ids):

        # Parse the row/col integers from the grid_id string once per tile
        try:
            row, col = _parse_row_col(grid_id)
        except ValueError:
            logger.warning("Skipping grid_id %r: cannot parse (row, col).", grid_id)
            continue

        # Reference the per-tile (T, H, W) cloud_mask array lazily — no I/O yet
        try:
            tile_cloud: zarr.Array = cloud_grp[grid_id]
        except KeyError:
            logger.warning(
                "grid_id %r not found in cloud_mask group; skipping.", grid_id
            )
            continue

        if (grid_idx + 1) % max(1, n_grids // 10) == 0:
            logger.info(
                "  Progress: %d / %d grid cells processed (%.0f%%)",
                grid_idx + 1, n_grids, 100 * (grid_idx + 1) / n_grids,
            )

        for t in range(T):
            date  = dates[t]
            month = months[t]
            month_total[month] += 1

            # Load only this single time step: (H₅₀₀, W₅₀₀) float32.
            # Using integer indexing on axis 0 avoids materialising the full
            # (T, H, W) tile — zarr reads exactly one chunk slice.
            soft_score: np.ndarray = tile_cloud[t]  # shape (H, W)

            # Cloud fraction: proportion of pixels below the clear-sky threshold.
            # np.mean(x < 0.5) treats NaN as False (NaN < 0.5 == False in numpy),
            # so fill / no-data pixels are counted as clear — conservative choice.
            cloud_fraction = float(np.mean(soft_score < 0.5))

            if cloud_fraction > max_cloud_fraction:
                # Too cloudy — discard this (tile, date) pair
                continue

            month_passing[month] += 1
            descriptors.append(
                TileDescriptor(
                    row=row,
                    col=col,
                    grid_id=grid_id,
                    date=date,
                    time_index=t,
                    cloud_fraction=cloud_fraction,
                    month=month,
                )
            )

    # ── Summary log ───────────────────────────────────────────────────────────
    n_passing  = len(descriptors)
    n_rejected = n_total - n_passing
    rejection_rate = n_rejected / max(n_total, 1)

    logger.info("=" * 60)
    logger.info("Tile extraction complete")
    logger.info("  Total (tile, date) pairs evaluated : %8d", n_total)
    logger.info("  Pairs passing cloud filter         : %8d", n_passing)
    logger.info("  Pairs rejected                     : %8d", n_rejected)
    logger.info("  Overall rejection rate             : %8.1f%%", rejection_rate * 100)
    logger.info("  Threshold (max_cloud_fraction)     :   %.2f", max_cloud_fraction)
    logger.info("")
    logger.info("  Month | Total  | Passing | Rejected | Reject %%")
    logger.info("  ------+--------+---------+----------+---------")
    for m in sorted(month_total):
        n_m   = month_total[m]
        n_p   = month_passing.get(m, 0)
        n_r   = n_m - n_p
        r_pct = 100 * n_r / max(n_m, 1)
        logger.info(
            "  %5d | %6d | %7d | %8d | %7.1f%%",
            m, n_m, n_p, n_r, r_pct,
        )
    logger.info("=" * 60)

    return descriptors


# ── Synchronous MTG + MOD09GQ dataset ─────────────────────────────────────────

@dataclass
class SyncSample:
    """
    One (grid_id, date) sample pairing an MTG daily max composite with MOD09GQ.

    Attributes
    ----------
    grid_id:
        MajorTOM grid identifier shared by both stores (e.g. ``"35D_55L"``).
    date:
        ISO date string ``"YYYY-MM-DD"`` of the observation.
    cell_lat, cell_lon:
        Decimal-degree centre coordinates of the MajorTOM cell.  Use as
        the ``cell_lat`` / ``cell_lon`` arguments of
        ``eumetsearch.transform.align_patches_multiband``.
    fci_patch_origins:
        ``(row0, col0)`` of the FCI patch in full-disk pixel coordinates,
        as stored in the ``patch_origins`` zarr array.  Consistent across
        all intra-day timestamps for the same grid_id.  Pass directly as
        ``fci_patch_row0`` / ``fci_patch_col0`` to ``align_patches_multiband``.
    mtg_composite:
        ``(C, H_mtg, W_mtg)`` float32 array.  C = number of MTG bands
        requested (default 2: vis_06, vis_08).  Each band is a pixel-wise
        maximum across all intra-day timestamps.  Missing timestamps are
        excluded; if *no* timestamp is available for a band the band slice
        is filled with NaN.
    modis:
        Dict mapping each requested MOD09GQ variable name to its
        ``(H_mod, W_mod)`` array.  Float variables are cast to float32 and
        pixels with value < −0.05 are replaced with NaN (fill / noise).
        Integer variables (e.g. ``QC_250m``) keep their original dtype.
        Missing variables are filled with a NaN float32 array.
    ancillary:
        Placeholder for ancillary sources such as ERA5 reanalysis fields.
        Always ``None`` until implemented.
    """

    grid_id:            str
    date:               str
    cell_lat:           float             # MajorTOM cell centre latitude
    cell_lon:           float             # MajorTOM cell centre longitude
    fci_patch_origins:  tuple[int, int]   # FCI full-disk (row0, col0) for align_patches_multiband
    mtg_composite:      np.ndarray        # (C, H_mtg, W_mtg) float32 max composite
    modis:              dict[str, np.ndarray]
    ancillary:          dict | None = None   # TODO: ERA5 and other ancillary sources


class MTGMODISDataset:
    """
    Synchronous dataset pairing MTG daily max-value composites with MOD09GQ.

    For every (grid_id, date) pair present in **both** stores the dataset:

    1. Collects all intra-day MTG timestamps for that date.
    2. Stacks the patches and reduces with ``np.nanmax`` along the time axis
       (max-value composite, one per band).
    3. Loads the corresponding MOD09GQ bands for the same grid cell and date.

    The result is a :class:`SyncSample` that can be used directly or fed into
    a ``torch.utils.data.DataLoader`` (duck-type compatible via ``__len__`` /
    ``__getitem__``).

    Parameters
    ----------
    mtg_path:
        Path to the MTG FCI zarr store (see module docstring for layout).
    modgq_path:
        Path to the MOD09GQ zarr store.
    mtg_bands:
        MTG band names to include in the composite.
        Defaults to ``("vis_06", "vis_08")``.
    modis_vars:
        MOD09GQ variable names to load.
        Defaults to ``("sur_refl_b01", "sur_refl_b02", "QC_250m")``.
    """

    MTG_BANDS:  tuple[str, ...] = ("vis_06", "vis_08")
    MODIS_VARS: tuple[str, ...] = ("sur_refl_b01", "sur_refl_b02", "QC_250m")

    def __init__(
        self,
        mtg_path:   Union[str, Path],
        modgq_path: Union[str, Path],
        mtg_bands:  tuple[str, ...] = MTG_BANDS,
        modis_vars: tuple[str, ...] = MODIS_VARS,
    ) -> None:
        self.mtg_path   = str(mtg_path)
        self.modgq_path = str(modgq_path)
        self.mtg_bands  = mtg_bands
        self.modis_vars = modis_vars

        self._mtg_store   = zarr.open(self.mtg_path,   mode="r")
        self._modgq_store = zarr.open(self.modgq_path, mode="r")

        # band → date → grid_id → [(ts_key, patch_idx), ...]
        # Built once in _build_index; used by _load_mtg_composite.
        self._mtg_lookup: dict[str, dict[str, dict[str, list[tuple[str, int]]]]] = {}

        # grid_id → (row0, col0) in FCI full-disk coords; populated by _build_index.
        self._patch_origins_lookup: dict[str, tuple[int, int]] = {}
        # grid_id → (lat, lon) of MajorTOM cell centre; populated by _build_index.
        self._cell_coords: dict[str, tuple[float, float]] = {}

        self._index: list[tuple[str, str]] = self._build_index()
        logger.info(
            "MTGMODISDataset ready: %d synchronous (grid_id, date) pairs "
            "| MTG bands: %s | MODIS vars: %s",
            len(self._index), self.mtg_bands, self.modis_vars,
        )

    # ── Index construction ─────────────────────────────────────────────────────

    def _build_index(self) -> list[tuple[str, str]]:
        """
        Scan both stores; return (grid_id, date) pairs present in both.

        Also populates ``self._mtg_lookup`` for O(1) patch retrieval in
        ``_load_mtg_composite``.
        """
        # ── MTG: build lookup and collect (date → grid_id set) ────────────────
        mtg_date_grids: dict[str, set[str]] = {}

        for band in self.mtg_bands:
            self._mtg_lookup[band] = {}
            try:
                band_grp = self._mtg_store["patches"][band]
            except KeyError:
                logger.warning("MTG band %r not found in store; skipping.", band)
                continue

            for ts_key in sorted(band_grp.keys()):
                date = ts_key[:10]              # "YYYY-MM-DD"
                ts_grp = band_grp[ts_key]
                gids: list[str] = ts_grp["grid_ids"][:].tolist()
                origins = ts_grp["patch_origins"][:]  # (N, 2) int32
                day_dict = self._mtg_lookup[band].setdefault(date, {})
                for patch_idx, gid in enumerate(gids):
                    day_dict.setdefault(gid, []).append((ts_key, patch_idx))
                    if gid not in self._patch_origins_lookup:
                        r0, c0 = int(origins[patch_idx, 0]), int(origins[patch_idx, 1])
                        self._patch_origins_lookup[gid] = (r0, c0)
                mtg_date_grids.setdefault(date, set()).update(gids)

        # ── MODIS: collect (date → grid_id set) ───────────────────────────────
        modis_date_grids: dict[str, set[str]] = {}
        ref_var = self.modis_vars[0] if self.modis_vars else None
        if ref_var is not None:
            try:
                var_grp = self._modgq_store["patches"][ref_var]
                for date_key in var_grp.keys():
                    modis_date_grids[date_key] = set(var_grp[date_key].keys())
            except KeyError:
                logger.warning("MODIS variable %r not found in store.", ref_var)

        # ── Intersect ─────────────────────────────────────────────────────────
        common: list[tuple[str, str]] = []
        for date in sorted(mtg_date_grids):
            if date not in modis_date_grids:
                continue
            for gid in sorted(mtg_date_grids[date] & modis_date_grids[date]):
                common.append((gid, date))

        logger.info(
            "_build_index: %d MTG dates, %d MODIS dates → %d common pairs",
            len(mtg_date_grids), len(modis_date_grids), len(common),
        )

        # ── Cell lat/lon for all common grid_ids ──────────────────────────
        unique_gids = {gid for gid, _ in common}
        try:
            from eumetsearch.transform import grid_id_to_latlon as _g2ll
            self._cell_coords = {gid: _g2ll(gid) for gid in unique_gids}
        except ImportError:
            self._cell_coords = {
                gid: (90.0 - row * 0.9, -180.0 + col * 0.9)
                for gid in unique_gids
                for row, col in [_parse_row_col(gid)]
            }

        return common

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> SyncSample:
        grid_id, date = self._index[idx]
        lat, lon = self._cell_coords[grid_id]
        return SyncSample(
            grid_id=grid_id,
            date=date,
            cell_lat=lat,
            cell_lon=lon,
            fci_patch_origins=self._patch_origins_lookup.get(grid_id, (0, 0)),
            mtg_composite=self._load_mtg_composite(grid_id, date),
            modis=self._load_modis(grid_id, date),
            ancillary=None,  # TODO: load ERA5 and other ancillary sources here
        )

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_mtg_composite(self, grid_id: str, date: str) -> np.ndarray:
        """
        Pixel-wise max composite across all intra-day MTG timestamps.

        Returns ``(C, H, W)`` float32.  C = len(self.mtg_bands).
        Bands with no coverage for this grid_id/date are filled with NaN.
        """
        band_composites: list[np.ndarray] = []

        for band in self.mtg_bands:
            entries = self._mtg_lookup.get(band, {}).get(date, {}).get(grid_id, [])

            if not entries:
                band_composites.append(np.full((128, 128), np.nan, dtype=np.float32))
                continue

            band_grp = self._mtg_store["patches"][band]
            frames: list[np.ndarray] = [
                band_grp[ts_key]["data"][patch_idx].astype(np.float32)
                for ts_key, patch_idx in entries
            ]

            # Suppress the all-NaN slice warning — expected when a pixel is
            # masked in every timestamp.
            with np.errstate(all="ignore"):
                composite = np.nanmax(np.stack(frames, axis=0), axis=0)  # (H, W)

            band_composites.append(composite)

        return np.stack(band_composites, axis=0)  # (C, H, W)

    def compute_mtg_ndvi_composite(self, grid_id: str, date: str) -> np.ndarray:
        """
        Daily max-NDVI composite for one grid cell and date.

        For each intra-day MTG timestamp:
          1. Load vis_06 (red) and vis_08 (NIR) patches.
          2. Compute per-pixel NDVI = (NIR - RED) / (NIR + RED).
             Pixels where NIR + RED < 0.01 (near-zero denominator) are masked as NaN.
          3. Take the pixel-wise maximum across all timestamps.

        Returns ``(H, W)`` float32 with NaN where no valid observation exists.
        """
        vis06_grp = self._mtg_store["patches"]["vis_06"]
        vis08_grp = self._mtg_store["patches"]["vis_08"]

        entries06 = self._mtg_lookup.get("vis_06", {}).get(date, {}).get(grid_id, [])
        entries08 = self._mtg_lookup.get("vis_08", {}).get(date, {}).get(grid_id, [])

        # Build a common set of timestamps available in both bands
        ts06 = {ts: idx for ts, idx in entries06}
        ts08 = {ts: idx for ts, idx in entries08}
        common_ts = sorted(set(ts06) & set(ts08))

        if not common_ts:
            return np.full((128, 128), np.nan, dtype=np.float32)

        ndvi_frames: list[np.ndarray] = []
        for ts in common_ts:
            red = vis06_grp[ts]["data"][ts06[ts]].astype(np.float32)
            nir = vis08_grp[ts]["data"][ts08[ts]].astype(np.float32)
            denom = nir + red
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = np.where(denom > 0.01, (nir - red) / denom, np.nan).astype(np.float32)
            ndvi_frames.append(ndvi)

        with np.errstate(all="ignore"):
            return np.nanmax(np.stack(ndvi_frames, axis=0), axis=0)

    def _load_modis(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        """
        Load all requested MOD09GQ variables for ``grid_id`` on ``date``.

        Missing variables are filled with NaN float32 of shape (512, 512).
        """
        result: dict[str, np.ndarray] = {}
        for var in self.modis_vars:
            try:
                raw = self._modgq_store["patches"][var][date][grid_id][:]
                if raw.dtype.kind == "f":
                    arr = raw.astype(np.float32)
                    arr[arr < -0.05] = np.nan  # mask fill/noise (negative reflectance)
                    result[var] = arr
                else:
                    result[var] = raw
            except KeyError:
                result[var] = np.full((512, 512), np.nan, dtype=np.float32)
        return result
