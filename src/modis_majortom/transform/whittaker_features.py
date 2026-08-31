"""
Diagnostic features derived from the ws2doptvp Whittaker smoother.

All public functions accept (T, H, W) numpy arrays (time-first) and a (T,)
days vector (float32, days from any fixed epoch).  No xarray dependency.

Typical usage
-------------
>>> from modis_majortom.transform.whittaker_features import (
...     weight_suppressed, run_lengths, suppressed_fraction_windowed,
...     forward_backward_gap, local_slope, high_risk_mask,
... )
>>> sup  = weight_suppressed(y, z)          # (T,H,W) uint8
>>> rl   = run_lengths(sup)                 # (T,H,W) int16
>>> swf  = suppressed_fraction_windowed(sup, days)
>>> fwd, bwd = forward_backward_gap(sup, days)
>>> sl   = local_slope(z, days)
>>> hr   = high_risk_mask(rl, sl)           # (T,H,W) uint8
>>> max_rl = rl.max(axis=0)                 # (H,W) — series-level scalar per pixel
"""
from __future__ import annotations

import numpy as np


def weight_suppressed(
    y: np.ndarray,
    z: np.ndarray,
    p: float = 0.99,
    threshold: float = 0.1,
) -> np.ndarray:
    """Binary suppression flag: 1 where ws2doptvp assigned low weight.

    ws2doptvp assigns weight ``p`` to observations at or above the smooth
    (residual ≥ 0, i.e. clear sky) and weight ``1−p`` to observations below
    (residual < 0, i.e. cloud-depressed).  Missing observations (NaN ``y``)
    receive weight 0.  With ``p=0.99`` the two clusters are 0.99 and 0.01;
    any ``threshold`` in (0.01, 0.99) separates them cleanly.

    Parameters
    ----------
    y : (T, H, W) float32
        Raw NDVI; NaN where missing / cloud-masked.
    z : (T, H, W) float32
        Smoothed NDVI (ws2doptvp upper envelope).
    p : float
        Asymmetry parameter used during smoothing (default 0.99).
    threshold : float
        Weight below which an observation is considered suppressed (default 0.1).

    Returns
    -------
    (T, H, W) uint8 — 1 = suppressed, 0 = clear / accepted.
    """
    residual = y - z
    w = np.where(
        np.isnan(y),
        np.float32(0.0),
        np.where(residual >= 0, np.float32(p), np.float32(1.0 - p)),
    )
    return (w < threshold).astype(np.uint8)


def run_lengths(suppressed: np.ndarray) -> np.ndarray:
    """For each timestep, the length (in timesteps) of the contiguous suppressed
    run it belongs to, or 0 if the timestep is not suppressed.

    Implemented as two vectorised scans over the time axis (T-length loop, each
    step a numpy broadcast over (H, W)) — no Python pixel loop.

    Parameters
    ----------
    suppressed : (T, H, W) uint8

    Returns
    -------
    (T, H, W) int16
        ``max_run_length`` per pixel is simply ``result.max(axis=0)``.
    """
    T, H, W = suppressed.shape
    sup = suppressed.astype(np.int16)

    # Forward pass: cum[t, h, w] = position within current run (1-indexed), 0 if clear.
    cum = np.zeros((H, W), dtype=np.int16)
    forward = np.zeros((T, H, W), dtype=np.int16)
    for t in range(T):
        m = sup[t].astype(bool)
        cum = np.where(m, (cum + np.int16(1)).astype(np.int16), np.int16(0))
        forward[t] = cum

    # Backward pass: propagate run length from each run-end position back through
    # the run.  ``fill`` tracks the run length to write for the current run.
    result = np.zeros((T, H, W), dtype=np.int16)
    fill = np.zeros((H, W), dtype=np.int16)
    for t in range(T - 1, -1, -1):
        m = sup[t].astype(bool)
        # A run ends at t when: suppressed at t AND (t is the last step OR not suppressed at t+1).
        is_run_end = m & (~sup[t + 1].astype(bool)) if t < T - 1 else m
        fill = np.where(is_run_end, forward[t], fill)
        result[t] = np.where(m, fill, np.int16(0))
        fill = np.where(~m, np.int16(0), fill)  # reset on clear timestep

    return result


def suppressed_fraction_windowed(
    suppressed: np.ndarray,
    days: np.ndarray,
    window_days: float = 20.0,
) -> np.ndarray:
    """Rolling mean of ``weight_suppressed`` over a centred calendar window.

    Uses actual day-spacing rather than fixed index steps, so irregular
    temporal gaps are handled correctly.

    Parameters
    ----------
    suppressed : (T, H, W) uint8
    days : (T,) float32 — ordinal days from any fixed epoch
    window_days : float
        Full width of the centred window in days (default 20).

    Returns
    -------
    (T, H, W) float32
    """
    T = suppressed.shape[0]
    diff = np.abs(days[:, None] - days[None, :])               # (T, T)
    in_window = (diff <= (window_days / 2.0)).astype(np.float32)  # (T, T)
    window_count = in_window.sum(axis=1)                        # (T,)

    sup_f32 = suppressed.astype(np.float32).reshape(T, -1)     # (T, H*W)
    rolling_sum = in_window @ sup_f32                           # (T, H*W)
    result = rolling_sum / window_count[:, None]
    return result.reshape(suppressed.shape).astype(np.float32)


def forward_backward_gap(
    suppressed: np.ndarray,
    days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Days to the nearest non-suppressed observation in each temporal direction.

    For non-suppressed timesteps the gap is 0.  For suppressed timesteps with
    no non-suppressed neighbour in that direction the gap is ``inf``.

    Parameters
    ----------
    suppressed : (T, H, W) uint8
    days : (T,) float32

    Returns
    -------
    forward_gap, backward_gap : each (T, H, W) float32
    """
    T, H, W = suppressed.shape
    sup = suppressed.astype(bool)

    # Backward gap — left-to-right scan, track last clear day per pixel.
    bwd = np.full((T, H, W), np.inf, dtype=np.float32)
    last_clear = np.full((H, W), -np.inf, dtype=np.float32)
    for t in range(T):
        clear = ~sup[t]
        gap = np.where(
            clear,
            np.float32(0.0),
            np.where(np.isfinite(last_clear), np.float32(days[t]) - last_clear, np.inf),
        )
        bwd[t] = gap
        last_clear = np.where(clear, np.float32(days[t]), last_clear)

    # Forward gap — right-to-left scan, track next clear day per pixel.
    fwd = np.full((T, H, W), np.inf, dtype=np.float32)
    next_clear = np.full((H, W), np.inf, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        clear = ~sup[t]
        gap = np.where(
            clear,
            np.float32(0.0),
            np.where(np.isfinite(next_clear), next_clear - np.float32(days[t]), np.inf),
        )
        fwd[t] = gap
        next_clear = np.where(clear, np.float32(days[t]), next_clear)

    return fwd, bwd


def local_slope(
    z: np.ndarray,
    days: np.ndarray,
    window_days: float = 5.0,
) -> np.ndarray:
    """Instantaneous slope of the smoothed NDVI (NDVI / day).

    At each timestep t, fits an ordinary least-squares line to all observations
    within ±(window_days/2) days and returns the slope.  Broadcasts efficiently
    over the spatial (H, W) dimension — no Python pixel loop.

    Parameters
    ----------
    z : (T, H, W) float32 — smoothed NDVI (envelope)
    days : (T,) float32
    window_days : float
        Full width of the local regression window in days (default 5).

    Returns
    -------
    (T, H, W) float32
    """
    T, H, W = z.shape
    result = np.zeros((T, H, W), dtype=np.float32)
    z_flat = z.reshape(T, -1)   # (T, N) where N = H*W

    half = window_days / 2.0
    for t in range(T):
        mask = np.abs(days - days[t]) <= half   # (T,) bool
        d = days[mask]                           # (K,)
        v = z_flat[mask]                         # (K, N)
        K = int(mask.sum())
        if K < 2:
            result[t] = 0.0
            continue
        d_c = d - d.mean()                       # (K,)
        v_c = v - v.mean(axis=0, keepdims=True)  # (K, N)
        den = float((d_c ** 2).sum())
        if den == 0.0:
            result[t] = 0.0
            continue
        num = (d_c[:, None] * v_c).sum(axis=0)  # (N,)
        result[t] = (num / den).reshape(H, W).astype(np.float32)

    return result


def high_risk_mask(
    run_len: np.ndarray,
    slope: np.ndarray,
    run_length_threshold: int = 4,
    slope_quantile: float = 0.90,
) -> np.ndarray:
    """High-risk suppression flag: long gap AND slope in the top quantile per pixel.

    A timestep is flagged when BOTH conditions hold:

    1. ``run_len[t] > run_length_threshold`` — the pixel is inside a long
       suppressed run (cloud gap spanning many observations).
    2. The local slope is in the extreme tail of that pixel's own distribution —
       either above the ``slope_quantile`` of positive slopes OR below the
       ``(1 − slope_quantile)`` of negative slopes, evaluated separately.

    Splitting the slope distribution by sign ensures that anomalously rapid
    senescence (steep negative slope) is judged against the pixel's own
    negative-slope history, not the typically steeper positive green-up ramps.
    Each pixel is judged against its own seasonal dynamic range.

    This is the actionable quality signal for deciding whether a pixel/series
    needs manual QC, a shorter smoothing λ, or exclusion from phenology metrics
    (SOS/POS/EOS dates are exactly what a mis-reconstructed green-up ramp corrupts).

    Parameters
    ----------
    run_len : (T, H, W) int16
        Output of :func:`run_lengths`.
    slope : (T, H, W) float32
        Output of :func:`local_slope`.
    run_length_threshold : int
        Minimum run length (timesteps) to trigger the run-length condition
        (default 4 — gaps longer than four consecutive suppressed observations).
    slope_quantile : float
        Per-pixel quantile of ``|slope|`` used as the adaptive slope threshold
        (default 0.90 → top 10 % of each pixel's slope magnitude series).

    Returns
    -------
    (T, H, W) uint8 — 1 = high-risk, 0 = not flagged.
    """
    # Compute separate quantile thresholds for positive and negative slopes so
    # that anomalous rapid senescence (steep negative) is judged against the
    # pixel's own negative-slope distribution, not the (typically steeper)
    # positive green-up ramps.
    pos = np.where(slope > 0, slope, np.nan)                                    # (T, H, W)
    neg = np.where(slope < 0, slope, np.nan)                                    # (T, H, W)
    pos_thresh = np.nanpercentile(pos, slope_quantile * 100,         axis=0)   # (H, W)
    neg_thresh = np.nanpercentile(neg, (1.0 - slope_quantile) * 100, axis=0)   # (H, W) — negative values

    slope_flag = (slope > pos_thresh[None, :, :]) | (slope < neg_thresh[None, :, :])

    run_flag = run_len > run_length_threshold                                   # (T, H, W)

    return (run_flag & slope_flag).astype(np.uint8)
