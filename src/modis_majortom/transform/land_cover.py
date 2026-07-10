"""
MCD12Q1 land cover tile extraction and AncillarySource for AlignedPatchDataset.

Two components:

1.  ``LandCoverTileExtractor``
    Reads the raw HDF files downloaded by ``MCD12Q1Downloader`` and extracts
    one AEQD tile per MajorTOM grid cell into a zarr store.

    Reprojection is done in a **single step**: MODIS sinusoidal → AEQD.
    For each cell the AEQD pixel centres are converted to sinusoidal (x, y)
    via the closed-form MODIS formula, then looked up with integer indexing
    directly in the HDF tile array — no intermediate EPSG:4326 or mosaic.
    This preserves the ~463 m native resolution at each cell centre and avoids
    the latitude-dependent distortion of EPSG:4326.

2.  ``LandCoverSource``
    Implements the ``AncillarySource`` ABC from ``transform.dataset`` so that
    land cover patches slot directly into ``AlignedPatchDataset``.  Because the
    patches are already in AEQD, ``reproject()`` is identity.

Zarr layout
-----------
    <zarr_out>/
        patches/
            LC_Type1/ {grid_id} → (256, 256) uint8
            LC_Type2/ {grid_id} → (256, 256) uint8
            QC/       {grid_id} → (256, 256) uint8
        attrs:
            product, year, pixel_size_m, n_pixels, bbox_wsen

AEQD grid parameters
--------------------
    n_pixels = 256, pixel_size_m = 500
    half_extent_m = 64 000 m

These match ``TargetGrid(extent_m = MODISGeometry.PATCH_EXTENT_M / 2,
n_pixels = 256)`` used everywhere else in the pipeline.
"""
from __future__ import annotations

import glob
import logging
import threading
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pyproj
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from .dataset import AncillarySource
from ..eo_data.land_cover import hv_from_filename, read_hdf_variable, _R, _TILE_SIZE, _PIXELS_PER_TILE, _PIXEL_SIZE

logger = logging.getLogger(__name__)

# ── AEQD grid parameters ─────────────────────────────────────────────────────
_N_PIXELS   = 256
_PIXEL_M    = 500.0
_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _aeqd_pixel_centres(
    cell_lat: float,
    cell_lon: float,
    n_pixels: int = _N_PIXELS,
    pixel_m: float = _PIXEL_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return *(lat, lon)* arrays (shape n×n) for an AEQD patch centred on the cell.

    Row 0 is northernmost; col 0 is westernmost.  Same convention as
    ``TargetGrid`` / ``_aeqd_to_latlon`` in eumetsearch.
    """
    half = n_pixels / 2 * pixel_m
    aeqd = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={cell_lat} +lon_0={cell_lon} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    xs = (np.arange(n_pixels) - n_pixels / 2 + 0.5) * pixel_m   # W → E
    ys = (n_pixels / 2 - np.arange(n_pixels) - 0.5) * pixel_m   # N → S
    XX, YY = np.meshgrid(xs, ys)

    t = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
    lon_flat, lat_flat = t.transform(XX.ravel(), YY.ravel())
    return (
        lat_flat.reshape(n_pixels, n_pixels).astype(np.float64),
        lon_flat.reshape(n_pixels, n_pixels).astype(np.float64),
    )


def _latlon_to_sinu(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert geographic (lat, lon) in degrees to MODIS sinusoidal (x, y) in metres."""
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    x = _R * lon_r * np.cos(lat_r)
    y = _R * lat_r
    return x, y


def _sinu_to_hv_pixel(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map sinusoidal (x, y) to MODIS tile (h, v) and integer pixel (row, col).

    Returns h, v, row, col — all integer arrays with the same shape as x/y.
    Values are clamped to valid tile-pixel range; callers should mask using
    the (h, v) validity check against the available HDF files.
    """
    h = ((x + 20_015_109) / _TILE_SIZE).astype(int)
    v = ((10_007_555 - y) / _TILE_SIZE).astype(int)

    x0 = -20_015_109 + h * _TILE_SIZE
    y0 =  10_007_555 - v * _TILE_SIZE

    col = np.clip(((x - x0) / _PIXEL_SIZE).astype(int), 0, _PIXELS_PER_TILE - 1)
    row = np.clip(((y0 - y) / _PIXEL_SIZE).astype(int), 0, _PIXELS_PER_TILE - 1)
    return h, v, row, col


# ── Tile extractor ────────────────────────────────────────────────────────────

class LandCoverTileExtractor:
    """Extract per-MajorTOM-cell AEQD land cover tiles directly from HDF files.

    For each MajorTOM grid cell the extractor:
      1. Builds an AEQD pixel grid (n×n at pixel_m resolution) centred on the cell.
      2. Converts pixel-centre (lat, lon) → MODIS sinusoidal (x, y).
      3. Looks up the integer pixel index in the MODIS tile array — **one
         nearest-neighbour step, sinusoidal → AEQD, no intermediate CRS**.
      4. Writes the result to zarr as uint8.

    Parameters
    ----------
    hdf_dir:
        Directory produced by ``MCD12Q1Downloader.download()``, containing
        ``*.hdf`` files for a single year.
    zarr_out:
        Output zarr store path.
    majortom_dist_km:
        MajorTOM grid spacing in km (default 100).
    bbox_filter:
        (lon_min, lat_min, lon_max, lat_max) — restrict to cells inside this box.
        Defaults to the Africa + EU region.
    variables:
        Variables to extract from the HDF files.
    n_pixels:
        Output patch size in pixels (square, default 256).
        Must match the ``target_n_pixels`` used by ``AlignedPatchDataset``.
    pixel_m:
        Output pixel spacing in metres (default 500, MCD12Q1 native resolution).
    """

    BBOX_AFRICA_EU: Tuple[float, float, float, float] = (-30.0, -40.0, 60.0, 80.0)

    def __init__(
        self,
        hdf_dir: Union[str, Path],
        zarr_out: Union[str, Path],
        majortom_dist_km: float = 100.0,
        bbox_filter: Tuple[float, float, float, float] | None = None,
        variables: Sequence[str] = ("LC_Type1", "LC_Type2", "QC"),
        n_pixels: int = _N_PIXELS,
        pixel_m: float = _PIXEL_M,
    ) -> None:
        self.hdf_dir          = Path(hdf_dir)
        self.zarr_out         = Path(zarr_out)
        self.majortom_dist_km = majortom_dist_km
        self.bbox_filter      = bbox_filter or self.BBOX_AFRICA_EU
        self.variables        = list(variables)
        self.n_pixels         = n_pixels
        self.pixel_m          = pixel_m

    def run(self) -> Path:
        """Extract all cells and write zarr.  Returns path to the zarr store.

        Resumes from partial runs: cells already present in the first variable
        group are skipped.
        """
        from majortom import Grid  # lazy import — not installed system-wide

        # ── Build (h, v) → hdf_path lookup from downloaded files ─────────────
        hdf_files = sorted(self.hdf_dir.glob("*.hdf"))
        if not hdf_files:
            raise FileNotFoundError(f"No HDF files found in {self.hdf_dir}")

        tile_index: dict[tuple[int, int], Path] = {}
        for f in hdf_files:
            h, v = hv_from_filename(f)
            if h is not None:
                tile_index[(h, v)] = f

        logger.info(
            "Found %d HDF tiles: h ∈ [%d,%d], v ∈ [%d,%d]",
            len(tile_index),
            min(k[0] for k in tile_index), max(k[0] for k in tile_index),
            min(k[1] for k in tile_index), max(k[1] for k in tile_index),
        )

        # ── Open / create zarr store ──────────────────────────────────────────
        store   = zarr.open_group(str(self.zarr_out), mode="a", zarr_format=2)
        patches = store.require_group("patches")
        for var in self.variables:
            patches.require_group(var)

        store.attrs.update({
            "product":      "MCD12Q1.061",
            "pixel_size_m": self.pixel_m,
            "n_pixels":     self.n_pixels,
            "bbox_wsen":    list(self.bbox_filter),
        })

        # ── Filter MajorTOM cells to bbox ─────────────────────────────────────
        lon_min, lat_min, lon_max, lat_max = self.bbox_filter
        logger.info("Loading MajorTOM grid (dist=%g km) …", self.majortom_dist_km)
        grid   = Grid(dist=self.majortom_dist_km)
        points = grid.points
        mask   = (
            (points.geometry.x >= lon_min) & (points.geometry.x <= lon_max) &
            (points.geometry.y >= lat_min) & (points.geometry.y <= lat_max)
        )
        cells   = points[mask]
        n_cells = len(cells)
        logger.info("%d grid cells to extract", n_cells)

        first_var = self.variables[0]
        skipped = written = 0

        for _, row in tqdm(cells.iterrows(), total=n_cells, desc="LC tiles"):
            grid_id  = row["name"]
            cell_lat = float(row.geometry.y)
            cell_lon = float(row.geometry.x)

            # Skip already-written cells (resume support)
            if first_var in patches and grid_id in patches[first_var]:
                skipped += 1
                continue

            # 1. AEQD pixel centres → geographic (lat, lon)
            lat_q, lon_q = _aeqd_pixel_centres(cell_lat, cell_lon, self.n_pixels, self.pixel_m)

            # 2. Geographic → MODIS sinusoidal
            x_sinu, y_sinu = _latlon_to_sinu(lat_q, lon_q)

            # 3. Sinusoidal → tile (h, v) and integer pixel (row, col)
            h_arr, v_arr, row_arr, col_arr = _sinu_to_hv_pixel(x_sinu, y_sinu)

            # Identify which tiles are needed for this cell
            needed_tiles = set(zip(h_arr.ravel().tolist(), v_arr.ravel().tolist()))
            available    = needed_tiles & set(tile_index.keys())

            if not available:
                skipped += 1
                continue

            for var in self.variables:
                output    = np.zeros((self.n_pixels, self.n_pixels), dtype=np.uint8)
                filled    = np.zeros((self.n_pixels, self.n_pixels), dtype=bool)

                for (h, v) in available:
                    px_mask = (h_arr == h) & (v_arr == v)
                    if not px_mask.any():
                        continue

                    tile_data, fill_value = read_hdf_variable(tile_index[(h, v)], var)
                    if tile_data is None:
                        continue

                    rows_idx = row_arr[px_mask]
                    cols_idx = col_arr[px_mask]
                    values   = tile_data[rows_idx, cols_idx]

                    if fill_value is not None:
                        values = np.where(values == fill_value, 0, values)

                    output[px_mask] = values.astype(np.uint8)
                    filled[px_mask] = True

                # Write to zarr
                var_grp = patches[var]
                if grid_id in var_grp:
                    var_grp[grid_id][:] = output
                else:
                    arr = var_grp.create(
                        name=grid_id,
                        shape=(self.n_pixels, self.n_pixels),
                        chunks=(self.n_pixels, self.n_pixels),
                        dtype="uint8",
                        compressor=_COMPRESSOR,
                    )
                    arr[:] = output

            written += 1

        logger.info(
            "Done — wrote %d tiles, skipped %d. Zarr: %s",
            written, skipped, self.zarr_out,
        )
        return self.zarr_out


# ── AncillarySource for AlignedPatchDataset ───────────────────────────────────

class LandCoverSource(AncillarySource):
    """MCD12Q1 land cover as an ancillary source for ``AlignedPatchDataset``.

    Reads pre-extracted AEQD tiles from the zarr produced by
    ``LandCoverTileExtractor.run()``.  Tiles are already on the shared AEQD
    grid, so ``reproject()`` is an identity pass-through.

    Parameters
    ----------
    zarr_path:
        Path to the zarr store written by ``LandCoverTileExtractor``.
    variables:
        Variables to expose.  Default: ``["LC_Type1"]``.
    """

    def __init__(
        self,
        zarr_path: Union[str, Path],
        variables: Sequence[str] = ("LC_Type1",),
    ) -> None:
        self._zarr_path = str(zarr_path)
        self.variables  = list(variables)
        self._local     = threading.local()

    @property
    def _store(self):
        if not hasattr(self._local, "store"):
            self._local.store = zarr.open(self._zarr_path, mode="r", zarr_format=2)
        return self._local.store

    def available_dates(self) -> set[str]:
        return set()  # annual — no date restriction

    def available_grid_ids(self, date: str) -> set[str]:
        try:
            return set(self._store["patches"][self.variables[0]].keys())
        except (KeyError, Exception):
            return set()

    def load(self, grid_id: str, date: str) -> dict[str, np.ndarray]:
        n = int(self._store.attrs.get("n_pixels", _N_PIXELS))
        bands: dict[str, np.ndarray] = {}
        for var in self.variables:
            try:
                bands[var] = self._store["patches"][var][grid_id][:].astype(np.float32)
            except KeyError:
                bands[var] = np.zeros((n, n), dtype=np.float32)
        return bands

    def patch_origin(self, lat: float, lon: float) -> tuple[int, int]:
        return (0, 0)

    def reproject(
        self,
        bands: dict[str, np.ndarray],
        row0: int,
        col0: int,
        target_grid,
    ) -> dict[str, np.ndarray]:
        # Tiles are pre-extracted in AEQD — identity pass-through
        return {k: v.astype(np.float32) for k, v in bands.items()}
