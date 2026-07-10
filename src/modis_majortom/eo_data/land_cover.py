"""
MCD12Q1 (MODIS Land Cover Type) download utilities.

``MCD12Q1Downloader`` fetches all HDF granules for a region and year from NASA
Earthdata via earthaccess and saves them to a persistent local directory.  No
reprojection is done here — the HDF files stay in their native MODIS sinusoidal
CRS.  Tile extraction and reprojection to AEQD are handled by
``LandCoverTileExtractor`` in ``transform.land_cover``.

Typical usage
-------------
    from modis_majortom.eo_data.land_cover import MCD12Q1Downloader

    dl = MCD12Q1Downloader(year=2024)
    hdf_dir = dl.download()   # data/modis/MCD12Q1/hdf/2024/
"""
from __future__ import annotations

import glob
import logging
import re
import time
from pathlib import Path
from typing import Tuple

import earthaccess
import numpy as np
from dotenv import load_dotenv
from pyhdf.SD import SD, SDC

from ..definitions import DATA_PATH

load_dotenv()
logger = logging.getLogger(__name__)


class MCD12Q1Downloader:
    """Download MCD12Q1.061 HDF tiles for a geographic region and year.

    Files are saved to ``out_dir / "hdf" / str(year)`` and kept on disk so
    that ``LandCoverTileExtractor`` can read them directly (sinusoidal → AEQD
    in one reprojection step, no intermediate EPSG:4326 mosaic).

    Parameters
    ----------
    year:
        Target year (MCD12Q1 is annual).
    bbox:
        (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
        Defaults to a box covering Africa and Europe.
    out_dir:
        Root directory under which ``hdf/<year>/`` is created.
        Defaults to ``DATA_PATH / "modis" / "MCD12Q1"``.
    """

    BBOX_AFRICA_EU: Tuple[float, float, float, float] = (-30.0, -40.0, 60.0, 80.0)

    def __init__(
        self,
        year: int,
        bbox: Tuple[float, float, float, float] = BBOX_AFRICA_EU,
        out_dir: Path | str | None = None,
    ) -> None:
        self.year    = int(year)
        self.bbox    = tuple(bbox)
        self.out_dir = Path(out_dir) if out_dir else DATA_PATH / "modis" / "MCD12Q1"
        self.hdf_dir = self.out_dir / "hdf" / str(self.year)
        self.hdf_dir.mkdir(parents=True, exist_ok=True)
        earthaccess.login(strategy="environment")

    def download(self) -> Path:
        """Download all HDF tiles for the year+bbox into ``self.hdf_dir``.

        Already-present files (matched by filename stem) are skipped so the
        method is safe to re-run after an interrupted download.

        Returns
        -------
        Path
            ``self.hdf_dir`` — the directory containing the HDF files.
        """
        existing = {Path(f).name for f in glob.glob(str(self.hdf_dir / "*.hdf"))}

        logger.info(
            "%d: searching MCD12Q1 granules for bbox %s …", self.year, self.bbox
        )
        results = self._search(self.year)
        if not results:
            raise RuntimeError(
                f"No MCD12Q1 granules found for {self.year} in bbox {self.bbox}"
            )

        # Filter to granules we don't already have
        to_download = [
            r for r in results
            if not any(
                Path(link.get("href", "")).name in existing
                for link in getattr(r, "links", [])
            )
        ]

        if not to_download:
            logger.info(
                "%d: all %d granules already present in %s",
                self.year, len(results), self.hdf_dir,
            )
            return self.hdf_dir

        logger.info(
            "%d: downloading %d / %d granules → %s",
            self.year, len(to_download), len(results), self.hdf_dir,
        )
        earthaccess.download(to_download, local_path=str(self.hdf_dir))

        n = len(glob.glob(str(self.hdf_dir / "*.hdf")))
        logger.info("%d: %d HDF files now in %s", self.year, n, self.hdf_dir)
        return self.hdf_dir

    # ── Internal ──────────────────────────────────────────────────────────────

    def _search(self, year: int, max_attempts: int = 5):
        for attempt in range(1, max_attempts + 1):
            try:
                return earthaccess.search_data(
                    short_name="MCD12Q1",
                    version="061",
                    temporal=(f"{year}-01-01", f"{year}-12-31"),
                    bounding_box=self.bbox,
                )
            except Exception as exc:
                wait = 30 * attempt
                logger.warning(
                    "Search attempt %d failed: %s. Retrying in %ds …", attempt, exc, wait
                )
                time.sleep(wait)
        return []


# ── Low-level HDF reader (shared with LandCoverTileExtractor) ─────────────────

# MODIS sinusoidal grid constants
_R              = 6_371_007.181        # Earth radius used by MODIS sinu
_TILE_SIZE      = 1_111_950.0          # metres per tile edge
_PIXELS_PER_TILE = 2_400               # MCD12Q1 500 m → 2400 px/tile
_PIXEL_SIZE     = _TILE_SIZE / _PIXELS_PER_TILE   # ≈ 463.3125 m


def hv_from_filename(path: str | Path) -> tuple[int | None, int | None]:
    """Parse MODIS tile indices h, v from a filename like ``.h18v07.``."""
    m = re.search(r"\.h(\d+)v(\d+)\.", str(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def read_hdf_variable(hdf_path: str | Path, variable: str) -> tuple[np.ndarray | None, float | None]:
    """Read one SDS from a MCD12Q1 HDF4 tile via pyhdf.

    Returns *(data, fill_value)* or *(None, None)* on failure.
    """
    try:
        hdf = SD(str(hdf_path), SDC.READ)
    except Exception as exc:
        logger.warning("Cannot open %s: %s", hdf_path, exc)
        return None, None

    sds_names = list(hdf.datasets().keys())
    sds_name  = next((n for n in sds_names if variable.lower() in n.lower()), None)
    if sds_name is None:
        hdf.end()
        return None, None

    sds        = hdf.select(sds_name)
    data       = sds.get()
    fill_value = sds.attributes().get("_FillValue")
    sds.endaccess()
    hdf.end()
    return data, fill_value
