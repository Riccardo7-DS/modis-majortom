"""
NDVI multi-date loader, Whittaker smoother, and temporal cloud detector.

Loads NDVI time-series from the sparse MajorTOM zarr stores (MOD09GQ for
reflectance, MOD35 for cloud masking), smooths each pixel with the Whittaker
filter from modape, and detects cloud-contaminated observations via an
asymmetric residual thresholding approach.

Typical usage
-------------
from modis_majortom.transform.ndvi_whittaker import WhittakerPipeline, CloudDetectionResult

pipeline = WhittakerPipeline()

# Load + smooth a spatial stack
smoothed, dates = pipeline.smooth_ndvi_series("266D_764L", ["2025-06-01", ...])

# Detect clouds on a single pixel time-series
result = pipeline.detect_clouds(ndvi_ts, doy, mod35_weights)
"""

from __future__ import annotations

import array as _array
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr
from joblib import Parallel, delayed
import xarray as xr
from ..definitions import DATA_PATH
from ..utils import compute_ndvi
from .cloud_adapter import upsample_500_to_250

log = logging.getLogger(__name__)


# ── Module-level helper for multiprocessing (must be picklable) ────────────────

def _detect_pixel(
    ndvi_ts, doy, mod_weights, landcover_class,
    n_iter, min_clear_obs, max_pos_residual, min_abs_residual, floor_residual,
    lc_k, lc_k_default, min_clear_obs_default, max_pos_residual_default,
    min_abs_residual_default, floor_residual_default,
    llas, half_normal_corr,
):
    """Standalone per-pixel cloud detection callable for process-pool workers.

    Mirrors ``WhittakerPipeline.detect_clouds`` but captures all configuration
    as plain arguments so it can be pickled by joblib's process backend.
    """
    from modape.whittaker import ws2d, ws2doptv

    if min_clear_obs    is None: min_clear_obs    = min_clear_obs_default
    if max_pos_residual is None: max_pos_residual = max_pos_residual_default
    if min_abs_residual is None: min_abs_residual = min_abs_residual_default
    if floor_residual   is None: floor_residual   = floor_residual_default

    T = len(ndvi_ts)
    month = np.clip((doy - 1) // 30, 0, 11).astype(np.int32) + 1

    w = mod_weights.astype(np.float64).copy()
    w[np.isnan(ndvi_ts)] = 0.0
    y = np.where(np.isnan(ndvi_ts), 0.0, ndvi_ts).astype(np.float64)

    if w.sum() == 0:
        return (
            np.zeros(T, dtype=bool),
            np.ones(T, dtype=bool),
            np.full(T, np.nan, dtype=np.float32),
            np.full(T, np.nan, dtype=np.float32),
        )

    z, lmda = ws2doptv(y, w, llas)
    z = np.array(z, dtype=np.float64)
    for _ in range(n_iter):
        residual_iter = y - z
        w_new = w.copy()
        w_new[residual_iter < 0] *= 0.1
        z = np.array(ws2d(y, lmda, w_new), dtype=np.float64)
        w = w_new
    ndvi_smooth = z.astype(np.float32)

    residuals = ndvi_ts.astype(np.float32) - ndvi_smooth
    sigma_by_month: dict[int, float] = {}
    for m in range(1, 13):
        mask_m = month == m
        pos_res = residuals[mask_m & ~np.isnan(residuals) & (residuals > 0)]
        pos_res = pos_res[pos_res <= max_pos_residual]
        sigma_by_month[m] = float(pos_res.std()) * half_normal_corr if pos_res.size > 1 else 1e-3

    k = lc_k.get(landcover_class, lc_k_default)
    threshold = np.array([-k * sigma_by_month[m] for m in month], dtype=np.float32)
    valid       = ~np.isnan(residuals)
    stat_cloud  = valid & (residuals < threshold) & (np.abs(residuals) >= min_abs_residual)
    floor_cloud = valid & (residuals < -floor_residual)
    cloud_mask  = stat_cloud | floor_cloud

    uncertain_mask = np.zeros(T, dtype=bool)
    confident_clear = mod_weights > 0.5
    for m in range(1, 13):
        mask_m = month == m
        if int(confident_clear[mask_m].sum()) < min_clear_obs:
            uncertain_mask[mask_m] = True
    cloud_mask[uncertain_mask] = False

    return cloud_mask, uncertain_mask, ndvi_smooth, residuals


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class CloudDetectionResult:
    """Per-pixel outputs of :meth:`WhittakerPipeline.detect_clouds`."""
    cloud_mask:     np.ndarray  # bool     (T,) — True = flagged as cloud
    uncertain_mask: np.ndarray  # bool     (T,) — True = too few clear obs in month
    ndvi_smooth:    np.ndarray  # float32  (T,) — asymmetric Whittaker envelope
    residuals:      np.ndarray  # float32  (T,) — ndvi_ts - ndvi_smooth


# ── Pipeline class ─────────────────────────────────────────────────────────────

class WhittakerPipeline:
    """Load, smooth, and cloud-detect MODIS NDVI time-series from zarr stores.

    Parameters
    ----------
    product:
        MODIS product for NDVI: ``"MOD09GQ"`` (250 m) or ``"MOD09GA"`` (500 m).
    path_250:
        Path to the MOD09GQ 250 m zarr store.
    path_cloud:
        Path to the MOD35 cloud-mask zarr store.
    state1km_algorithm:
        Algorithm passed to ``MODISQCMask.state_1km_cloud_mask`` when decoding
        the ``state_1km`` QA band from the MOD09GA store.  Options:
        ``"strict"`` (cloud | mixed | shadow | aerosol≥2 | cirrus | internal),
        ``"cloud_state"`` (bits 0-1 only: 01=cloudy, 10=mixed),
        ``"internal"`` (bit 10 internal flag only),
        ``"binned"`` (bits 0-1: 10=mixed, 11=not-set — MODIS science team convention).
        Default ``"binned"``.
    weight_source:
        Source used to generate per-pixel weights for Whittaker smoothing and
        cloud detection.  Also controls which pixels are NaN-masked in the
        returned ``ndvi_masked`` stack.

        * ``"mod35"`` (default) — MOD35_L2 quality flag (0=confident cloudy …
          3=confident clear, 99=fill); the binned algorithm flags quality < 2.
        * ``"state1km"`` — MOD09GA ``state_1km`` QA bits decoded with
          ``self.state1km_algorithm`` (default ``"strict"`` is recommended here
          for tighter initial masking before smoothing).

        Regardless of this setting, the ``cloud_mask_mod35`` variable in the
        output dataset always reflects the true MOD35 mask.
    k:
        Per-observation k-sigma multiplier.  Pass a float to override the
        default for all land-cover classes, or a dict keyed by LC class index
        to set per-class values.

    Cloud detection thresholds
    --------------------------
    Three parameters shape the sensitivity of :meth:`detect_clouds`.  They can
    be passed directly to any ``detect_clouds*`` or ``smooth_ndvi_series`` call,
    or changed globally by overriding the class constants.

    ``max_pos_residual`` (default 0.1) — tightens the k·σ threshold
        Per-month sigma is estimated from the std of *positive* residuals
        (observations above the smooth envelope), scaled by √(π/2) to recover
        the full-distribution σ.  Thin cirrus or partially-suppressed
        observations can land slightly above the envelope and inflate sigma,
        raising the detection threshold and making the detector miss subtle
        cloud dips.  Positive residuals larger than ``max_pos_residual`` are
        excluded before computing std, trimming that right tail.

        * Lower value → smaller sigma → **more sensitive** detector
        * ``np.inf`` → no clipping, full right tail included (permissive)

    ``min_abs_residual`` (default 0.02) — suppresses tiny false positives
        Even when a dip satisfies the k·σ criterion, it must also exceed this
        absolute size.  Prevents flagging dips of −0.01 or −0.02 that are
        statistically significant only because sigma is very small (e.g., on a
        stable desert pixel in a dry month).

        * Higher value → fewer flags, may miss subtle cloud dips
        * 0.0 → no absolute gate; pure sigma test

    ``floor_residual`` (default 0.05) — guarantees large dips are always flagged
        Any observation with ``residual < −floor_residual`` is flagged as cloud
        unconditionally, bypassing the sigma test.  Guards against high-variance
        pixels (dense forest, irrigated cropland) where k·σ grows so large that
        even real cloud dips of −0.10 fail the statistical test.

        * Lower value → more aggressive unconditional flagging
        * ``np.inf`` → disables the floor; pure sigma test

        Default 0.10: conservative enough to avoid false positives from haze or
        atmospheric correction errors on vegetated surfaces (typical noise dips
        < 0.05), while still catching real cloud contamination (usually ≥ 0.10
        on vegetated pixels).  For arid/sparse pixels the k·σ threshold is
        already tight enough that the floor rarely fires.

    The combined decision rule is::

        flag if (residual < −k·σ  AND  |residual| ≥ min_abs_residual)
                OR  residual < −floor_residual

    Typical tuning by pixel type:

    * **Stable desert / low NDVI**: very small sigma → many false positives.
      Raise ``min_abs_residual``.
    * **Dense forest / high variability**: large sigma → misses real clouds.
      Lower ``floor_residual``.
    * **Frequent thin cirrus**: inflated sigma from the cirrus tail.
      Lower ``max_pos_residual``.
    """

    # ── Default zarr paths ────────────────────────────────────────────────────
    DEFAULT_250   = DATA_PATH / "MOD09GQ_dataset.zarr"
    DEFAULT_500   = DATA_PATH / "MOD09GA_dataset.zarr"
    DEFAULT_CLOUD = DATA_PATH / "MOD35_L2_dataset.zarr"

    # ── Cloud-detection constants ─────────────────────────────────────────────

    # k-sigma thresholds per broad land-cover category.
    # Tighter k → more aggressive cloud flagging.
    LC_K: dict[int, float] = {
        1: 2.5,   # dense forest  — large seasonal amplitude, looser threshold
        2: 1.8,   # cropland      — sharp phenology, low natural variability
        3: 2.0,   # savanna       — intermediate
        4: 2.0,   # arid / sparse — MIN_ABS_RESIDUAL already guards noise-level dips;
                  #                  k=1.5 added no real sensitivity here
    }
    
    LC_K_DEFAULT = 2.0
    
    # Minimum MOD35-confident-clear obs per month to trust the sigma estimate.
    MIN_CLEAR_OBS = 3

    # Upper clip for positive residuals used in per-month sigma estimation.
    # Residuals above this are likely thin-cirrus artifacts, not clean noise.
    MAX_POS_RESIDUAL: float = 0.1

    # Minimum absolute residual to trigger the statistical cloud flag.
    # Dips smaller than this are within measurement noise — skip them even if
    # they satisfy the k·σ threshold.
    MIN_ABS_RESIDUAL: float = 0.02

    # Unconditional cloud floor: flag without a sigma test if residual < −this.
    # Guards against high-variance pixels where k·σ grows too permissive.
    # 0.10 chosen to avoid false positives from haze / atmospheric correction
    # errors on vegetated surfaces (typical dip < 0.05); real cloud dips are
    # almost always ≥ 0.10 on vegetated pixels.
    FLOOR_RESIDUAL: float = 0.10

    # Invert half-normal to recover full-distribution σ from positive residuals:
    # std(positive half) × √(π/2) ≈ full σ  (for X ~ N(0, σ²))
    _HALF_NORMAL_CORR = math.sqrt(math.pi / 2)

    # ── Batch processing constants ────────────────────────────────────────────
    AVAILABLE_VARIABLES: frozenset = frozenset({
        "ndvi_raw", "ndvi_smoothed", "ndvi_envelope",
        "cloud_mask_algo", "uncertain_mask", "residuals",
        "cloud_mask_mod35", "cloud_mask_state1km",
        "soft_score",
    })
    DEFAULT_VARIABLES: tuple = ("ndvi_envelope", "soft_score")

    # Default log₁₀(λ) search grid for ws2doptv / ws2doptvp.
    # Both functions expect log10-transformed lambda values (not actual lambdas).
    # Range -1…5 covers λ = 0.1 … 100 000, sufficient for near-daily NDVI.
    # Step 0.2 gives fine enough resolution for the V-curve to find a good minimum.
    # Must be a Python array.array('d', ...) — ws2doptv rejects numpy arrays.
    _LLAS = _array.array('d', np.arange(-1, 5, 0.2).round(2).tolist())

    def __init__(
        self,
        product: str = "MOD09GQ",
        path_250: Path | str | None = None,
        path_500: Path | str | None = None,
        path_cloud: Path | str | None = None,
        state1km_algorithm: str = "binned",
        weight_source: str = "mod35",
        k: float | dict[int, float] | None = None,
    ) -> None:
        self.product = product          # "MOD09GQ" or "MOD09GA"
        self.path_250   = Path(path_250)   if path_250   else self.DEFAULT_250
        self.path_500   = Path(path_500)   if path_500   else self.DEFAULT_500
        self.path_cloud = Path(path_cloud) if path_cloud else self.DEFAULT_CLOUD
        self.state1km_algorithm = state1km_algorithm
        if weight_source not in ("mod35", "state1km"):
            raise ValueError(f"weight_source must be 'mod35' or 'state1km', got {weight_source!r}")
        self.weight_source = weight_source
        if isinstance(k, dict):
            self.LC_K = k           # per-class lookup table
        elif k is not None:
            self.LC_K_DEFAULT = float(k)   # single value overrides the default
    # ── Private zarr helpers ──────────────────────────────────────────────────

    @staticmethod
    def _open(path: Path) -> zarr.Group:
        return zarr.open(path, mode="r")

    @staticmethod
    def _load_ndvi_patch(
        patches_250: zarr.Group,
        date: str,
        grid_id: str,
    ) -> np.ndarray | None:
        """Return float32 NDVI (H, W) for one tile/date, or None if missing."""
        try:
            b01 = patches_250["sur_refl_b01"][date][grid_id][:].astype(np.float32)
            b02 = patches_250["sur_refl_b02"][date][grid_id][:].astype(np.float32)
        except (KeyError, Exception):
            return None
        # fill_below=-0.05 masks MODIS fill values before the NIR/Red ratio;
        # clip guards residual outliers from near-zero (but non-zero) denominators.
        ndvi = compute_ndvi(b02, b01, fill_below=-0.05)
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def _load_cloud_mask(
        patches_cloud: zarr.Group,
        date: str,
        grid_id: str,
    ) -> np.ndarray | None:
        """Return bool MOD35 cloud mask (H, W): True = cloudy/fill.  None if missing."""
        try:
            raw = patches_cloud["cloud_mask"][date][grid_id][:].astype(np.uint8)
        except (KeyError, Exception):
            return None
        # stored: 0=confident cloudy, 1=probably cloudy, 2=probably clear,
        #         3=confident clear, 99=fill
        return (raw < 2) | (raw == 99)

    def _load_state1km_mask(
        self,
        date: str,
        grid_id: str,
    ) -> np.ndarray | None:
        """Return bool state_1km cloud mask from MOD09GA, upsampled if product is MOD09GQ.

        Returns None when the date or grid_id is absent in the MOD09GA store.
        True = cloudy, False = clear (binned algorithm: cloud_state bits 2|3).
        """
        try:
            store = self._open(self.path_500)
            raw = store["patches"]["state_1km"][date][grid_id][:].astype(np.uint16)
        except KeyError:
            log.debug("state_1km not found for %s / %s", date, grid_id)
            return None
        except Exception as exc:
            log.warning("Failed to load state_1km for %s / %s: %s", date, grid_id, exc)
            return None

        from ..eo_data.modis import MODISQCMask
        mask = MODISQCMask.state_1km_cloud_mask(raw, algorithm=self.state1km_algorithm)

        if self.product == "MOD09GQ":
            # state_1km is 500 m; double resolution to match 250 m NDVI patches
            mask = upsample_500_to_250(mask)

        return mask.astype(bool)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_ndvi_stack(
        self,
        grid_id: str,
        dates: Sequence[str],
        cloud_mask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load a cloud-masked NDVI stack for one grid cell across many dates.

        Parameters
        ----------
        grid_id:
            MajorTOM grid cell identifier, e.g. ``"266D_764L"``.
        dates:
            Sequence of ``"YYYY-MM-DD"`` strings to load.
        cloud_mask:
            If True (default), set cloudy/fill pixels to NaN and compute
            per-pixel weights using ``self.weight_source``
            (``"mod35"`` or ``"state1km"``).

        Returns
        -------
        ndvi_masked : np.ndarray  shape (T, H, W), float32 — NaN where cloud/fill
        ndvi_raw    : np.ndarray  shape (T, H, W), float32 — original values
        weights     : np.ndarray  shape (T, H, W), float32 — 1=clear, 0=cloud/fill
                      Source determined by ``self.weight_source``.
        valid_dates : list[str]   dates with at least one valid NDVI patch
        """
        store_250   = self._open(self.path_250)
        patches_250 = store_250["patches"]

        # Open the MOD35 store only when needed.
        patches_cloud: zarr.Group | None = None
        if cloud_mask and self.weight_source == "mod35":
            patches_cloud = self._open(self.path_cloud)["patches"]

        raw_layers:    list[np.ndarray] = []
        masked_layers: list[np.ndarray] = []
        weight_layers: list[np.ndarray] = []
        valid_dates:   list[str]        = []

        for date in dates:
            ndvi = self._load_ndvi_patch(patches_250, date, grid_id)
            if ndvi is None:
                log.debug("No 250 m data for %s / %s — skipping", grid_id, date)
                continue

            raw_layers.append(ndvi.copy())
            w = np.ones(ndvi.shape, dtype=np.float32)   # default: all clear

            if cloud_mask:
                if self.weight_source == "state1km":
                    cloud = self._load_state1km_mask(date, grid_id)
                else:
                    cloud = self._load_cloud_mask(patches_cloud, date, grid_id)

                if cloud is not None:
                    ndvi = np.where(cloud, np.nan, ndvi)
                    w    = (~cloud).astype(np.float32)
                else:
                    log.debug("No %s mask for %s / %s — keeping all pixels",
                              self.weight_source, grid_id, date)

            masked_layers.append(ndvi)
            weight_layers.append(w)
            valid_dates.append(date)

        if not raw_layers:
            raise ValueError(f"No valid data found for grid_id={grid_id!r} in the requested dates.")

        ndvi_masked = np.stack(masked_layers, axis=0).astype(np.float32)   # (T, H, W)
        ndvi_raw    = np.stack(raw_layers,    axis=0).astype(np.float32)   # (T, H, W)
        weights     = np.stack(weight_layers, axis=0).astype(np.float32)   # (T, H, W)
        return ndvi_masked, ndvi_raw, weights, valid_dates

    def smooth_xarray(
        self,
        da: "xr.DataArray",
        p: float | None = None,
        llas: "_array.array | None" = None,
        lambda_min: float = -1.0,
        lambda_max: float = 5.0,
        lambda_step: float = 0.2,
        time_dim: str = "time",
    ) -> "xr.DataArray":
        """Apply Whittaker smoothing to an xarray DataArray along the time dimension.

        Uses ``xr.apply_ufunc`` for vectorized execution over spatial dimensions.
        Supports dask-backed arrays transparently.

        Parameters
        ----------
        da:
            DataArray with a ``time`` (or ``time_dim``) dimension.
            NaN values are treated as missing (weight = 0).
        p:
            Asymmetry parameter for ``ws2doptvp`` (0 < p < 1).

            * ``p < 0.5``  — upper envelope; suppresses cloud dips (p ≈ 0.1).
            * ``p > 0.5``  — lower envelope.
            * ``p = 0.5``  — symmetric (equivalent to ``ws2doptv``).
            * ``None``     — use ``ws2doptv`` (symmetric V-curve, default).
        llas:
            Explicit log₁₀(λ) search grid as ``array.array('d', ...)``.
            If provided, overrides ``lambda_min/max/step``.
            Defaults to ``np.arange(lambda_min, lambda_max, lambda_step)``.
        lambda_min:
            Lower bound of the log₁₀(λ) search range.  Default ``-1``
            (λ = 0.1 — essentially no smoothing).
        lambda_max:
            Upper bound of the log₁₀(λ) search range.  Default ``5``
            (λ = 100 000 — very heavy smoothing).
        lambda_step:
            Step size in log₁₀ space.  Finer steps improve λ selection at
            the cost of runtime.  Default ``0.2``.
        time_dim:
            Name of the time dimension (default ``"time"``).

        Returns
        -------
        xr.DataArray
            Smoothed values with the same dimensions and coordinates as ``da``.

        Examples
        --------
        >>> pipeline = WhittakerPipeline()
        >>> smoothed = pipeline.smooth_xarray(ndvi_da)                    # symmetric
        >>> envelope = pipeline.smooth_xarray(ndvi_da, p=0.1)             # upper envelope
        >>> fine    = pipeline.smooth_xarray(ndvi_da, lambda_min=-2, lambda_max=4)
        """
        import xarray as xr

        if llas is None:
            llas = _array.array(
                'd',
                np.arange(lambda_min, lambda_max, lambda_step).round(4).tolist(),
            )

        w_da = xr.where(da.isnull(), 0.0, 1.0).astype(np.float32)
        y_da = da.fillna(0.0).astype(np.float32)

        # Capture in local names so the closure works inside apply_ufunc.
        _p, _llas = p, llas

        def _pixel_smooth(y: np.ndarray, w: np.ndarray) -> np.ndarray:
            y_d = y.astype(np.float64)
            w_d = w.astype(np.float64)
            if w_d.sum() == 0:
                return np.full_like(y, np.nan, dtype=np.float32)
            if _p is not None:
                from modape.whittaker import ws2doptvp
                z, _ = ws2doptvp(y_d, w_d, _llas, p=_p)
            else:
                from modape.whittaker import ws2doptv
                z, _ = ws2doptv(y_d, w_d, _llas)
            return np.array(z, dtype=np.float32)

        result = xr.apply_ufunc(
            _pixel_smooth,
            y_da,
            w_da,
            input_core_dims=[[time_dim], [time_dim]],
            output_core_dims=[[time_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        # apply_ufunc moves the core dim to the last axis; restore original order.
        return result.transpose(*da.dims)

    def smooth_ndvi_series(
        self,
        grid_id: str,
        dates: Sequence[str],
        cloud_mask: bool = True,
        llas: "_array.array | None" = None,
        p: float | None = None,
        as_dataset: bool = False,
        landcover_class: int | None = None,
        max_pos_residual: float | None = None,
        min_abs_residual: float | None = None,
        floor_residual: float | None = None,
        n_jobs: int = 1,
        variables: list[str] | None = None,
    ):
        """Load and Whittaker-smooth a multi-date NDVI stack for one grid cell.

        Uses :meth:`smooth_xarray` to apply the smoother spatially via
        ``xr.apply_ufunc``.  By default uses ``ws2doptv`` (symmetric V-curve
        auto-λ).  Pass ``p`` to switch to ``ws2doptvp`` (asymmetric).

        Parameters
        ----------
        grid_id:
            MajorTOM grid cell identifier.
        dates:
            Sequence of ``"YYYY-MM-DD"`` strings to load.
        cloud_mask:
            Apply MOD35 cloud masking before smoothing (default True).
        llas:
            λ candidates for V-curve search.  Defaults to ``[-1, 0, 1, 2, 3, 4, 5]``.
        p:
            Asymmetry parameter forwarded to :meth:`smooth_xarray`.
            ``None`` (default) → symmetric ``ws2doptv``.
            ``p < 0.5`` (e.g. ``0.1``) → upper envelope via ``ws2doptvp``,
            useful when you want a cloud-suppressed smooth in one pass.
        as_dataset:
            When True, return an ``xr.Dataset`` instead of a bare numpy array.
        landcover_class:
            Forwarded to :meth:`detect_clouds_xarray` when ``as_dataset=True``.
        n_jobs:
            Parallel workers for :meth:`detect_clouds_xarray` (``-1`` = all
            CPUs).  Ignored when ``as_dataset=False``.

        Returns
        -------
        (smoothed, dates) : (np.ndarray shape (T, H, W), list[str])
            Default return when ``as_dataset=False``.
        xr.Dataset
            When ``as_dataset=True``.
        """
        import xarray as xr

        ndvi_masked, ndvi_raw, weights, valid_dates = self.load_ndvi_stack(
            grid_id, dates, cloud_mask=cloud_mask
        )

        T, H, W = ndvi_masked.shape

        # Build a temporary DataArray so smooth_xarray can use apply_ufunc.
        # NaN positions in ndvi_masked → weight=0 inside smooth_xarray.
        t_idx = np.arange(T, dtype=np.int32)
        ndvi_da_tmp = xr.DataArray(
            ndvi_masked,
            dims=("time", "y", "x"),
            coords={"time": t_idx, "y": np.arange(H), "x": np.arange(W)},
        )
        # Only run the symmetric Whittaker smooth when explicitly requested.
        # The detection step already produces ndvi_envelope (asymmetric smooth),
        # which is a better gap-fill for ML targets and costs no extra compute.
        _vars = set(variables) if variables is not None else set(self.DEFAULT_VARIABLES)
        if "ndvi_smoothed" in _vars:
            smoothed = self.smooth_xarray(ndvi_da_tmp, p=p, llas=llas).values
        else:
            smoothed = None

        log.info("NDVI stack: shape=%s, dates=%s  (smooth_skipped=%s)",
                 ndvi_masked.shape, valid_dates, smoothed is None)

        if not as_dataset:
            return smoothed, valid_dates

        return self._to_dataset(
            grid_id=grid_id,
            ndvi_raw=ndvi_raw,
            ndvi_masked=ndvi_masked,
            ndvi_smoothed=smoothed,
            weights=weights,
            valid_dates=valid_dates,
            landcover_class=landcover_class,
            max_pos_residual=max_pos_residual,
            min_abs_residual=min_abs_residual,
            floor_residual=floor_residual,
            n_jobs=n_jobs,
        )

    def _to_dataset(
        self,
        grid_id: str,
        ndvi_raw: np.ndarray,
        ndvi_masked: np.ndarray,
        ndvi_smoothed: np.ndarray,
        weights: np.ndarray,
        valid_dates: list[str],
        landcover_class: int | None = None,
        max_pos_residual: float | None = None,
        min_abs_residual: float | None = None,
        floor_residual: float | None = None,
        n_jobs: int = 1,
    ):
        """Build the full output xr.Dataset.

        Called internally by :meth:`smooth_ndvi_series` when ``as_dataset=True``.

        The y-axis is flipped (axis 1 of every array) so that the dataset
        displays correctly with xarray's default plot (y increasing upward),
        matching the visual convention used by ``imshow(origin='upper')`` in
        the notebook.
        """
        import pandas as pd
        import xarray as xr

        T, H, W = ndvi_raw.shape
        times = pd.to_datetime(valid_dates)

        # Decreasing y coordinates → xarray plots row-0 at the top,
        # matching the imshow(origin='upper') convention used in the notebook.
        y_coords = np.arange(H - 1, -1, -1, dtype=np.int32)
        x_coords = np.arange(W, dtype=np.int32)
        dims   = ("time", "y", "x")
        coords = {"time": times, "y": y_coords, "x": x_coords}

        # ── Cloud detection on RAW (unmasked) NDVI + MOD35 weights ───────────
        # Passing raw NDVI (cloud dips still present as actual values, not NaN)
        # lets the asymmetric Whittaker smoother down-weight them iteratively.
        # If we passed the NaN-masked stack the smoother would have nothing to
        # suppress, causing the envelope to sit above all clear obs and flag
        # nearly everything as cloud.
        ndvi_raw_da  = xr.DataArray(ndvi_raw, dims=dims, coords=coords)
        weights_da   = xr.DataArray(weights,  dims=dims, coords=coords)

        detection = self.detect_clouds_xarray(
            ndvi_raw_da, weights_da,
            landcover_class=landcover_class,
            max_pos_residual=max_pos_residual,
            min_abs_residual=min_abs_residual,
            floor_residual=floor_residual,
            n_jobs=n_jobs,
        )

        # ── state_1km mask — load per date, fallback to NaN ──────────────────
        s1km_layers: list[np.ndarray] = []
        for date in valid_dates:
            mask = self._load_state1km_mask(date, grid_id)
            if mask is None:
                s1km_layers.append(np.full((H, W), np.nan, dtype=np.float32))
            else:
                mask = mask[:H, :W]
                s1km_layers.append(mask.astype(np.float32))
        state1km_stack = np.stack(s1km_layers, axis=0)

        # ── MOD35 binary mask — always loaded from the MOD35 store ───────────
        # When weight_source="state1km" the smoothing weights aren't from MOD35,
        # so we load MOD35 independently here to keep the dataset variable correct.
        if self.weight_source == "mod35":
            mod35_mask = (1.0 - weights).astype(np.float32)   # fast path: reuse weights
        else:
            patches_cloud = self._open(self.path_cloud)["patches"]
            mod35_layers: list[np.ndarray] = []
            for date in valid_dates:
                m35 = self._load_cloud_mask(patches_cloud, date, grid_id)
                if m35 is None:
                    mod35_layers.append(np.full((H, W), np.nan, dtype=np.float32))
                else:
                    mod35_layers.append(m35.astype(np.float32))
            mod35_mask = np.stack(mod35_layers, axis=0)

        # ── Soft score: continuous cloud confidence for ML loss weighting ────
        # 1.0 = confident clear, 0.5 = uncertain (too few clear obs in month),
        # 0.0 = cloud (flagged by primary mask or Whittaker residual detector).
        # Missing primary mask pixels (NaN) are treated as uncertain (0.5).
        algo_cloud     = detection["cloud_mask"].values      # bool (T, H, W)
        algo_uncertain = detection["uncertain_mask"].values  # bool (T, H, W)
        if self.weight_source == "state1km":
            primary_cloud = state1km_stack > 0.5   # NaN > 0.5 → False in numpy
            primary_nan   = np.isnan(state1km_stack)
        else:
            primary_cloud = mod35_mask > 0.5
            primary_nan   = np.isnan(mod35_mask)
        is_cloud     = primary_cloud | algo_cloud
        is_uncertain = algo_uncertain | primary_nan
        soft_score   = np.where(is_cloud, 0.0,
                       np.where(is_uncertain, 0.5, 1.0)).astype(np.float32)

        # ── Assemble dataset ─────────────────────────────────────────────────
        def _da(arr, long_name=""):
            return xr.DataArray(arr, dims=dims, coords=coords,
                                attrs={"long_name": long_name} if long_name else {})

        dataset_vars = {
            "ndvi_raw":  _da(ndvi_raw, "Raw NDVI (no cloud masking)"),
        }
        if ndvi_smoothed is not None:
            dataset_vars["ndvi_smoothed"] = _da(
                ndvi_smoothed, "Whittaker-smoothed NDVI (ws2doptv, V-curve λ)"
            )
        dataset_vars.update({
            "ndvi_envelope":       detection["ndvi_smooth"].assign_attrs(
                                       {"long_name": "Asymmetric Whittaker upper envelope (ws2d, fixed λ)"}),
            "cloud_mask_algo":     detection["cloud_mask"].assign_attrs(
                                       {"long_name": "Temporal cloud flag (Whittaker residual, True=cloud)"}),
            "uncertain_mask":      detection["uncertain_mask"].assign_attrs(
                                       {"long_name": "Uncertain flag (too few clear obs in month)"}),
            "residuals":           detection["residuals"].assign_attrs(
                                       {"long_name": "NDVI residuals (raw − envelope)"}),
            "cloud_mask_mod35":    _da(mod35_mask,    "MOD35 cloud mask (True=cloud, from weights)"),
            "cloud_mask_state1km": _da(state1km_stack,
                                       f"state_1km cloud mask [{self.state1km_algorithm}] (True=cloud; NaN=not available)"),
            "soft_score":          _da(soft_score,
                                       "Cloud soft score (1=confident clear, 0.5=uncertain, 0=cloud)"),
        })
        return xr.Dataset(dataset_vars)

    def detect_clouds(
        self,
        ndvi_ts: np.ndarray,
        doy: np.ndarray,
        mod_weights: np.ndarray,
        landcover_class: int | None = None,
        n_iter: int = 5,
        min_clear_obs: int | None = None,
        max_pos_residual: float | None = None,
        min_abs_residual: float | None = None,
        floor_residual: float | None = None,
    ) -> CloudDetectionResult:
        """Detect cloud-contaminated observations in a single-pixel NDVI series.

        Fits a smooth upper envelope via an asymmetric iterative Whittaker
        filter (using ``self.lmda``), then flags observations whose negative
        deviation exceeds a land-cover-dependent multiple of the per-month
        clean-noise standard deviation.

        Parameters
        ----------
        ndvi_ts:
            Raw NDVI observations, shape (T,).  NaN treated as missing.
        doy:
            Day-of-year for each observation, shape (T,), dtype int.
        mod_weights:
            Per-observation cloud confidence weights in [0, 1] (0=cloudy,
            1=clear).  Source depends on ``self.weight_source`` (MOD35 or
            state_1km).  Used as initial weights for the smoother and for the
            uncertain flag.
        landcover_class:
            Optional land-cover index (1=forest, 2=cropland, 3=savanna,
            4=arid).  None → default k=2.0.
        n_iter:
            Asymmetric re-weighting iterations.
        min_clear_obs:
            Months with fewer confident-clear obs than this are flagged as
            uncertain.  Defaults to ``self.MIN_CLEAR_OBS``.
        max_pos_residual:
            Upper bound applied to positive residuals before computing per-month
            sigma.  Trims the thin-cirrus right tail that would otherwise inflate
            the noise floor and desensitise the detector.
            ``None`` → ``self.MAX_POS_RESIDUAL`` (default 0.1 NDVI units).
            Pass ``np.inf`` to disable clipping entirely.
        min_abs_residual:
            Minimum absolute residual magnitude to trigger the statistical cloud
            flag.  Prevents flagging tiny dips (e.g. −0.02) that are within
            measurement noise even when they satisfy the k·σ criterion.
            ``None`` → ``self.MIN_ABS_RESIDUAL`` (default 0.02 NDVI units).
        floor_residual:
            Unconditional flag threshold: any observation with
            ``residual < −floor_residual`` is marked as cloud regardless of the
            per-month sigma.  Guards against high-variance pixels where k·σ
            grows too permissive.
            ``None`` → ``self.FLOOR_RESIDUAL`` (default 0.05 NDVI units).
            Set to ``np.inf`` to disable the floor entirely.

        Returns
        -------
        CloudDetectionResult
            cloud_mask, uncertain_mask, ndvi_smooth, residuals.
        """
        from modape.whittaker import ws2d, ws2doptv  # lazy import — optional dependency

        if min_clear_obs is None:
            min_clear_obs = self.MIN_CLEAR_OBS
        if max_pos_residual is None:
            max_pos_residual = self.MAX_POS_RESIDUAL
        if min_abs_residual is None:
            min_abs_residual = self.MIN_ABS_RESIDUAL
        if floor_residual is None:
            floor_residual = self.FLOOR_RESIDUAL

        T = len(ndvi_ts)

        # Convert doy → approximate calendar month (1–12).
        # doy 1–31 → month 1, …, 335–365 → month 12.
        month = np.clip((doy - 1) // 30, 0, 11).astype(np.int32) + 1  # (T,)

        # ── Step 1: asymmetric iterative Whittaker smoother ───────────────────
        # Start from the supplied weights; zero out NaN positions.
        w = mod_weights.astype(np.float64).copy()
        w[np.isnan(ndvi_ts)] = 0.0

        y = np.where(np.isnan(ndvi_ts), 0.0, ndvi_ts).astype(np.float64)

        if w.sum() == 0:
            return CloudDetectionResult(
                cloud_mask=np.zeros(T, dtype=bool),
                uncertain_mask=np.ones(T, dtype=bool),
                ndvi_smooth=np.full(T, np.nan, dtype=np.float32),
                residuals=np.full(T, np.nan, dtype=np.float32),
            )

        # Find the per-pixel optimal λ via V-curve, then use it for all
        # asymmetric iterations — no global fixed λ required.
        # ws2doptv returns sopt as the actual optimal λ value (not log10),
        # so it can be passed directly to ws2d.
        z, lmda = ws2doptv(y, w, self._LLAS)
        z = np.array(z, dtype=np.float64)

        for _ in range(n_iter):
            residual_iter = y - z
            # Cloud dips sit below the envelope (negative residuals).
            # Soft-penalise them so the smoother tracks the clear-sky surface.
            w_new = w.copy()
            w_new[residual_iter < 0] *= 0.1
            z = np.array(ws2d(y, lmda, w_new), dtype=np.float64)
            w = w_new

        ndvi_smooth = z.astype(np.float32)

        # ── Step 2: residuals ─────────────────────────────────────────────────
        residuals = ndvi_ts.astype(np.float32) - ndvi_smooth  # (T,)

        # ── Step 3: per-month clean-noise sigma (half-normal estimator) ───────
        # Using only positive residuals avoids the left tail being pulled down
        # by cloud dips, which would inflate sigma and desensitise the detector.
        sigma_by_month: dict[int, float] = {}
        for m in range(1, 13):
            mask_m = month == m
            pos_res = residuals[mask_m & ~np.isnan(residuals) & (residuals > 0)]
            pos_res = pos_res[pos_res <= max_pos_residual]   # trim thin-cirrus right tail
            if pos_res.size > 1:
                sigma_by_month[m] = float(pos_res.std()) * self._HALF_NORMAL_CORR
            else:
                sigma_by_month[m] = 1e-3   # non-zero fallback keeps threshold finite

        # ── Step 4: land-cover sensitivity k ─────────────────────────────────
        k = self.LC_K.get(landcover_class, self.LC_K_DEFAULT)

        # ── Step 5: cloud flag ────────────────────────────────────────────────
        # Statistical path: residual < −k·σ AND abs(residual) ≥ min_abs_residual.
        # The absolute guard suppresses noise-level dips that happen to clear the
        # k·σ bar when sigma is very small.
        # Floor path: always flag if residual < −floor_residual regardless of σ,
        # catching cases where k·σ grows too permissive on high-variance pixels.
        threshold = np.array(
            [-k * sigma_by_month[m] for m in month], dtype=np.float32
        )
        valid = ~np.isnan(residuals)
        stat_cloud  = valid & (residuals < threshold) & (np.abs(residuals) >= min_abs_residual)
        floor_cloud = valid & (residuals < -floor_residual)
        cloud_mask  = stat_cloud | floor_cloud

        # ── Step 6: uncertain-month flag ─────────────────────────────────────
        # If a month has too few high-confidence clear obs, the sigma estimate
        # is unreliable — mark all obs in that month as uncertain, not cloud.
        uncertain_mask = np.zeros(T, dtype=bool)
        confident_clear = mod_weights > 0.5

        for m in range(1, 13):
            mask_m = month == m
            if int(confident_clear[mask_m].sum()) < min_clear_obs:
                uncertain_mask[mask_m] = True

        cloud_mask[uncertain_mask] = False   # uncertain takes priority over cloudy

        return CloudDetectionResult(
            cloud_mask=cloud_mask,
            uncertain_mask=uncertain_mask,
            ndvi_smooth=ndvi_smooth,
            residuals=residuals,
        )

    def detect_clouds_spatial(
        self,
        ndvi_stack: np.ndarray,
        doy: np.ndarray,
        mod_stack: np.ndarray,
        landcover_class: int | np.ndarray | None = None,
        n_iter: int = 5,
        min_clear_obs: int | None = None,
        max_pos_residual: float | None = None,
        min_abs_residual: float | None = None,
        floor_residual: float | None = None,
        n_jobs: int = 1,
    ) -> CloudDetectionResult:
        """Apply :meth:`detect_clouds` to every pixel of a (T, H, W) stack.

        Pixels are processed independently; joblib parallelises over the
        flattened spatial dimension.

        Parameters
        ----------
        ndvi_stack:
            Shape (T, H, W), float32.  NaN = missing/cloud.
        doy:
            Day-of-year for each time step, shape (T,), dtype int.
        mod_stack:
            MOD confidence weights, shape (T, H, W), float32 in [0, 1].
        landcover_class:
            Either a scalar int (same class for all pixels) or an (H, W) int
            array with a per-pixel class.  None → default k=2.0 everywhere.
        n_iter:
            Asymmetric Whittaker iterations per pixel.
        min_clear_obs, max_pos_residual, min_abs_residual, floor_residual:
            Passed through to :meth:`detect_clouds`.
        n_jobs:
            Number of parallel workers (joblib).  -1 = all CPUs, 1 = serial.

        Returns
        -------
        CloudDetectionResult
            All four fields have shape (T, H, W).
        """
        T, H, W = ndvi_stack.shape
        N = H * W

        # Flatten spatial dims: (T, H, W) → (T, N)
        ndvi_flat   = ndvi_stack.reshape(T, N)
        mod_flat  = mod_stack.reshape(T, N)

        # Build per-pixel land-cover array: (N,)
        if landcover_class is None or np.isscalar(landcover_class):
            lc_flat = np.full(N, landcover_class if landcover_class is not None else -1, dtype=object)
        else:
            lc_flat = np.asarray(landcover_class).reshape(N)

        # prefer="processes": ws2doptv/ws2d hold the GIL so threads serialize;
        # separate processes each have their own GIL → true parallelism.
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_detect_pixel)(
                ndvi_flat[:, i], doy, mod_flat[:, i],
                int(lc_flat[i]) if lc_flat[i] != -1 else None,
                n_iter, min_clear_obs,
                max_pos_residual, min_abs_residual, floor_residual,
                self.LC_K, self.LC_K_DEFAULT,
                self.MIN_CLEAR_OBS, self.MAX_POS_RESIDUAL,
                self.MIN_ABS_RESIDUAL, self.FLOOR_RESIDUAL,
                self._LLAS, self._HALF_NORMAL_CORR,
            )
            for i in range(N)
        )

        # Unpack and reshape back to (T, H, W)
        cloud_mask     = np.stack([r[0] for r in results], axis=1).reshape(T, H, W)
        uncertain_mask = np.stack([r[1] for r in results], axis=1).reshape(T, H, W)
        ndvi_smooth    = np.stack([r[2] for r in results], axis=1).reshape(T, H, W)
        residuals      = np.stack([r[3] for r in results], axis=1).reshape(T, H, W)

        return CloudDetectionResult(
            cloud_mask=cloud_mask,
            uncertain_mask=uncertain_mask,
            ndvi_smooth=ndvi_smooth,
            residuals=residuals,
        )

    def detect_clouds_xarray(
        self,
        ndvi_da,
        weights_da,
        landcover_class: int | None = None,
        time_dim: str = "time",
        n_iter: int = 5,
        min_clear_obs: int | None = None,
        max_pos_residual: float | None = None,
        min_abs_residual: float | None = None,
        floor_residual: float | None = None,
        n_jobs: int = 1,
    ):
        """Detect clouds on an xarray DataArray with a time dimension.

        Extracts day-of-year from the ``time`` coordinate (must be
        ``datetime64``), calls :meth:`detect_clouds_spatial` on the underlying
        numpy arrays, and returns the four result fields as an ``xr.Dataset``
        with the same coordinates as the input.

        Parameters
        ----------
        ndvi_da:
            ``xr.DataArray`` with at least a ``time`` dimension.
        weights_da:
            Cloud confidence weights (0=cloudy, 1=clear) with the same shape
            and coords as ``ndvi_da``.  Source depends on ``self.weight_source``.
        landcover_class:
            Scalar land-cover class (or None for default k).  Per-pixel LC
            is not supported through this interface; use
            :meth:`detect_clouds_spatial` directly.
        time_dim:
            Name of the time dimension (default ``"time"``).
        n_iter, min_clear_obs, n_jobs:
            Forwarded to :meth:`detect_clouds_spatial`.

        Returns
        -------
        xr.Dataset
            Variables: ``cloud_mask``, ``uncertain_mask``, ``ndvi_smooth``,
            ``residuals``, all with the same dimensions and coordinates as
            ``ndvi_da``.
        """
        import xarray as xr

        # Extract day-of-year from the datetime coordinate.
        time_coord = ndvi_da[time_dim]
        doy = time_coord.dt.dayofyear.values.astype(np.int32)

        # Move time to axis-0 so the array is (T, ...).
        ndvi_arr    = ndvi_da.transpose(time_dim, ...).values.astype(np.float32)
        weights_arr = weights_da.transpose(time_dim, ...).values.astype(np.float32)

        # Ensure exactly 3-D: (T, H, W).
        if ndvi_arr.ndim == 2:
            ndvi_arr    = ndvi_arr[:, :, np.newaxis]
            weights_arr = weights_arr[:, :, np.newaxis]
            squeeze = True
        else:
            squeeze = False

        result = self.detect_clouds_spatial(
            ndvi_arr, doy, weights_arr,
            landcover_class=landcover_class,
            n_iter=n_iter,
            min_clear_obs=min_clear_obs,
            max_pos_residual=max_pos_residual,
            min_abs_residual=min_abs_residual,
            floor_residual=floor_residual,
            n_jobs=n_jobs,
        )

        if squeeze:
            cloud_mask     = result.cloud_mask[:, :, 0]
            uncertain_mask = result.uncertain_mask[:, :, 0]
            ndvi_smooth    = result.ndvi_smooth[:, :, 0]
            residuals      = result.residuals[:, :, 0]
        else:
            cloud_mask, uncertain_mask = result.cloud_mask, result.uncertain_mask
            ndvi_smooth, residuals     = result.ndvi_smooth, result.residuals

        # Rebuild as Dataset aligned to the original (time-first) coords.
        coords = ndvi_da.transpose(time_dim, ...).coords
        dims   = ndvi_da.transpose(time_dim, ...).dims

        return xr.Dataset({
            "cloud_mask":     xr.DataArray(cloud_mask,     dims=dims, coords=coords),
            "uncertain_mask": xr.DataArray(uncertain_mask, dims=dims, coords=coords),
            "ndvi_smooth":    xr.DataArray(ndvi_smooth,    dims=dims, coords=coords),
            "residuals":      xr.DataArray(residuals,      dims=dims, coords=coords),
        })

    # ── Batch zarr processing ─────────────────────────────────────────────────

    def process_zarr(
        self,
        output_zarr: Path | str,
        grid_ids: list[str] | None = None,
        dates: list[str] | None = None,
        variables: list[str] | None = None,
        n_jobs: int = 1,
        show_progress: bool = True,
        **detect_kwargs,
    ) -> None:
        """Run the full pipeline on every grid cell and write results to a zarr store.

        Iterates over all grid IDs in the source store, runs
        :meth:`smooth_ndvi_series` with ``as_dataset=True``, and writes the
        requested output bands back in the MajorTOM sparse layout::

            output_zarr/patches/{variable}/{date}/{grid_id}  →  (H, W) array

        The output store is created if it does not exist; existing arrays are
        overwritten so the method is safe to re-run after a partial failure.

        Parameters
        ----------
        output_zarr:
            Path to the output zarr store.
        grid_ids:
            Subset of grid IDs to process.  ``None`` → all IDs in the source.
        dates:
            Subset of dates to process.  ``None`` → all dates in the source.
        variables:
            Variables to write.  ``None`` → ``self.DEFAULT_VARIABLES``
            (``ndvi_smoothed``, ``cloud_mask_algo``, ``uncertain_mask``).
            Any subset of ``self.AVAILABLE_VARIABLES`` is accepted:

            * ``ndvi_raw`` — raw NDVI before cloud masking
            * ``ndvi_smoothed`` — symmetric Whittaker smooth (V-curve λ)
            * ``ndvi_envelope`` — asymmetric upper envelope
            * ``cloud_mask_algo`` — Whittaker residual cloud flag
            * ``uncertain_mask`` — months with too few clear obs
            * ``residuals`` — raw NDVI − envelope
            * ``cloud_mask_mod35`` — MOD35 binary cloud mask
            * ``cloud_mask_state1km`` — state_1km cloud mask
        n_jobs:
            Parallel grid workers (joblib threads).  When > 1 the pixel-level
            parallelism inside each grid is disabled (nested pools conflict);
            when == 1 the pixel loop uses all CPUs (``n_jobs=-1``).
        show_progress:
            Display a tqdm progress bar over grids.
        **detect_kwargs:
            Forwarded to :meth:`smooth_ndvi_series`
            (e.g. ``landcover_class``, ``max_pos_residual``, ``floor_residual``).
        """
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **_kw):   # type: ignore[misc]
                return it
        from zarr.codecs import BloscCodec

        output_zarr = Path(output_zarr)
        variables = list(variables) if variables is not None else list(self.DEFAULT_VARIABLES)
        unknown = set(variables) - self.AVAILABLE_VARIABLES
        if unknown:
            raise ValueError(
                f"Unknown variables: {unknown}.\n"
                f"Available: {sorted(self.AVAILABLE_VARIABLES)}"
            )

        # ── enumerate source dates and grid IDs ───────────────────────────
        src_path = self.path_250 if self.product == "MOD09GQ" else self.path_500
        src_patches = self._open(src_path)["patches"]
        ref_var = "sur_refl_b01"
        all_dates    = sorted(src_patches[ref_var].keys()) \
                       if dates    is None else list(dates)
        all_grid_ids = sorted(src_patches[ref_var][all_dates[0]].keys()) \
                       if grid_ids is None else list(grid_ids)

        log.info(
            "process_zarr: %d grids × %d dates → %s  variables=%s",
            len(all_grid_ids), len(all_dates), output_zarr, variables,
        )

        # ── open output store and pre-create all variable/date groups ─────
        # Pre-creation avoids TOCTOU races when the grid loop runs in parallel.
        compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
        out_store   = zarr.open(str(output_zarr), mode="a")
        out_patches = out_store.require_group("patches")
        for var in variables:
            vg = out_patches.require_group(var)
            for date in all_dates:
                vg.require_group(date)

        # ── per-pixel parallelism: disable when grid loop is parallel ─────
        pixel_jobs = 1 if n_jobs != 1 else -1

        def _process_grid(grid_id: str) -> None:
            try:
                ds = self.smooth_ndvi_series(
                    grid_id,
                    all_dates,
                    as_dataset=True,
                    n_jobs=pixel_jobs,
                    variables=variables,
                    **detect_kwargs,
                )
            except Exception as exc:
                log.warning("Grid %s skipped: %s", grid_id, exc)
                return

            actual_dates = ds.time.dt.strftime("%Y-%m-%d").values

            for var in variables:
                if var not in ds:
                    continue
                arr = ds[var].values   # (T, H, W) — numpy, no dask
                date_grp = out_patches[var]
                for i, date in enumerate(actual_dates):
                    patch = arr[i]     # (H, W)
                    dg = date_grp[date]
                    if grid_id in dg:
                        dg[grid_id][:] = patch
                    else:
                        dg.create_array(
                            name=grid_id,
                            shape=patch.shape,
                            chunks=patch.shape,
                            dtype=patch.dtype,
                            compressors=[compressor],
                        )
                        dg[grid_id][:] = patch

        # ── grid loop ─────────────────────────────────────────────────────
        grid_iter = _tqdm(all_grid_ids, desc="grids", disable=not show_progress)
        if n_jobs == 1:
            for gid in grid_iter:
                _process_grid(gid)
        else:
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_grid)(gid) for gid in grid_iter
            )

        zarr.consolidate_metadata(str(output_zarr))
        log.info("process_zarr: complete → %s", output_zarr)
