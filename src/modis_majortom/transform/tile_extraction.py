"""
Tile extraction with cloud-fraction filtering for the diffusion model dataset.

Iterates over every Major TOM grid cell and every time step stored in the MODGA
zarr archive, computes a per-tile cloud fraction from the soft cloud-confidence
score, and returns lightweight descriptors for the (tile, date) pairs that pass
the threshold.  No array data is written into the descriptors — they are pure
metadata records consumed later by the DataLoader.

Expected MODGA zarr layout
--------------------------
modga.zarr/
    patches/
        cloud_mask/
            <grid_id>   →  zarr array  (T, H₅₀₀, W₅₀₀)  float32
                           soft cloud score in [0, 1]
                           1 = confident clear, 0 = confident cloud
                           (produced by Whittaker residual + MOD35 fusion)
        blue_band/
            <grid_id>   →  zarr array  (T, H₅₀₀, W₅₀₀)  float32
    dates               →  zarr array  (T,)  str "YYYY-MM-DD"

MODGQ zarr layout (250 m target, accessed later by the DataLoader)
-------------------------------------------------------------------
modgq.zarr/
    patches/
        ndvi/
            <grid_id>   →  zarr array  (T, 2·H₅₀₀, 2·W₅₀₀)  float32
    dates               →  zarr array  (T,)  str

The same grid_id key used in MODGA maps directly to the corresponding MODGQ
tile.  The DataLoader does not need separate 250 m spatial indices — it uses
the grid_id stored in each TileDescriptor and slices the MODGQ array with
the same time_index.

MTG and ERA5 zarrs are accepted as arguments for API completeness but are not
read at this stage.
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
