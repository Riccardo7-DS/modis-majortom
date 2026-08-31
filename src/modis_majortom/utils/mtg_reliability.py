"""Temporal-reliability composite channels derived from the 5-timestamp MTG sequence."""
from __future__ import annotations

import numpy as np


def compute_mtg_composites(
    ndvi_stack: np.ndarray,
    vis08_stack: np.ndarray,
    mad_epsilon: float = 1e-6,
    anomaly_threshold: float = 3.0,
) -> np.ndarray:
    """Derive 4 per-pixel temporal-reliability channels from a 5-timestamp MTG sequence.

    Cloud/anomaly flagging uses a MAD-based temporal-inconsistency test on VIS08
    (clouds move between timestamps; vegetation reflectance does not), per-pixel:
    ``anomaly_t = |VIS08_t - median_t(VIS08)| / (MAD_t(VIS08) + mad_epsilon)``,
    flagged where ``anomaly_t > anomaly_threshold``. Zero-padded timestamps (dates
    with fewer than 5 real MTG observations) participate in this computation as-is
    (their VIS08=0 typically deviates from the median and gets flagged).

    Parameters
    ----------
    ndvi_stack : np.ndarray, shape (5, H, W) float32
        Per-timestamp NDVI (zero-padded slots included as 0.0).
    vis08_stack : np.ndarray, shape (5, H, W) float32
        Per-timestamp raw VIS08 reflectance (zero-padded slots included as 0.0).
    mad_epsilon : float
        Stabiliser added to the MAD denominator.
    anomaly_threshold : float
        Anomaly-score threshold above which an observation is flagged suspicious.

    Returns
    -------
    composites : np.ndarray, shape (3, H, W) float32
        ``[NDVI75, NDVI_std, CloudScore]``.
        NDVI75     : 75th-percentile NDVI across the 5 timestamps.
        NDVI_std   : temporal std of NDVI across the 5 timestamps.
        CloudScore : fraction of flagged timestamps, range [0, 1].

    Note: a count-of-valid-observations channel (``N_valid``) was deliberately
    dropped — with a fixed 5-timestamp denominator it is an exact affine
    transform of CloudScore (``N_valid = 5 * (1 - CloudScore)``), so it adds
    no information beyond what CloudScore already encodes.
    """
    median  = np.median(vis08_stack, axis=0, keepdims=True)   # (1, H, W)
    abs_dev = np.abs(vis08_stack - median)                    # (5, H, W)
    mad     = np.median(abs_dev, axis=0, keepdims=True)       # (1, H, W)
    anomaly = abs_dev / (mad + mad_epsilon)                   # (5, H, W)
    flagged = anomaly > anomaly_threshold                     # (5, H, W) bool

    ndvi75      = np.percentile(ndvi_stack, 75, axis=0).astype(np.float32)
    ndvi_std    = ndvi_stack.std(axis=0).astype(np.float32)
    cloud_score = flagged.mean(axis=0).astype(np.float32)

    return np.stack([ndvi75, ndvi_std, cloud_score], axis=0)
