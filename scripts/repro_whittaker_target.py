"""Reproduce the Whittaker training-target failure modes on synthetic data.

Runs the real pipeline (``WhittakerPipeline.detect_clouds`` and the target
blend from ``transform/dataset.py``) on a synthetic daily NDVI year where the
cloud-free truth is known, so the training target's error is directly
measurable.

Failure modes demonstrated:

1. Detector hair-trigger: the five cumulative ``w *= 0.1`` iterations push the
   envelope to the top of the data, the positive residuals that feed the
   per-month sigma estimate become tiny, and the -k*sigma threshold collapses
   to the MIN_ABS_RESIDUAL guard - ordinary clear observations get flagged as
   cloud, so the target degenerates to the (upper-biased) envelope almost
   everywhere.
2. Uncertain-month leak: in a month with fewer than MIN_CLEAR_OBS
   confident-clear observations the cloud flags are cancelled, so a
   contaminated observation that slipped the primary mask enters the target
   at 50% (soft_score 0.5) with loss weight 0.6.
3. Run-splitting: contaminated observations that survive the primary mask
   break suppression runs, so run-length gap thresholds under-fire exactly
   where the artifacts are.

Usage (from the repo root, an environment with modape installed):

    python scripts/repro_whittaker_target.py [--seeds N] [--figure out.png]
"""

from __future__ import annotations

import argparse

import numpy as np

from modis_majortom.transform.ndvi_whittaker import WhittakerPipeline

T = 365
DAYS = np.arange(T)
DOY = DAYS + 1
# Window 1: long cloudy run, primary mask misses ~40% of it
W1 = np.arange(150, 176)
# Window 2: a full detector-month (doy 241..270) of cloud, primary mask
# catches all but two observations -> fewer than MIN_CLEAR_OBS clear obs
W2 = np.arange(240, 270)
W2_SLIPS = np.array([248, 259])


def double_logistic(t, vmin=0.18, vmax=0.85, sos=110, eos=280, rs=0.09, re=0.07):
    return vmin + (vmax - vmin) * (
        1 / (1 + np.exp(-rs * (t - sos))) - 1 / (1 + np.exp(-re * (t - eos)))
    )


def make_scene(seed: int):
    """Synthetic year: known truth, cloud contamination, imperfect primary mask.

    Returns (truth, y, weights) where weights plays the role of the MOD35
    weights passed to detect_clouds (1 = clear, 0 = cloud).
    """
    rng = np.random.default_rng(seed)
    truth = double_logistic(DAYS)
    y = truth + rng.normal(0, 0.02, T)

    protected = np.concatenate([W1, np.arange(235, 275)])
    idx_short = rng.choice(np.setdiff1d(DAYS, protected), 55, replace=False)
    y[idx_short] -= rng.uniform(0.15, 0.5, idx_short.size)
    weights = np.ones(T)
    weights[idx_short] = (rng.random(idx_short.size) < 0.15).astype(float)

    y[W1] = truth[W1] - rng.uniform(0.25, 0.55, W1.size)
    spikes = W1[[8, 16]]
    y[spikes] = truth[spikes] + rng.uniform(0.08, 0.15, 2)
    weights[W1] = (rng.random(W1.size) > 0.60).astype(float)
    weights[spikes] = 1.0

    y[W2] = truth[W2] - rng.uniform(0.25, 0.55, W2.size)
    weights[W2] = 0.0
    weights[W2_SLIPS] = 1.0

    return truth, y, weights


def build_target(pipeline: WhittakerPipeline, y, weights):
    """detect_clouds + the soft_score/target/loss_weight recipe of the dataset.

    soft_score follows _to_dataset (ndvi_whittaker.py): cloud -> 0,
    uncertain -> 0.5, clear -> 1. Target and loss weight follow dataset.py:
    soft*obs + (1-soft)*envelope and soft + 0.2*(1-soft).
    """
    res = pipeline.detect_clouds(y.astype(np.float32), DOY, weights)
    primary_cloud = weights < 0.5
    is_cloud = primary_cloud | res.cloud_mask
    soft = np.where(is_cloud, 0.0, np.where(res.uncertain_mask, 0.5, 1.0))
    target = soft * y + (1.0 - soft) * res.ndvi_smooth.astype(np.float64)
    loss_weight = soft + 0.2 * (1.0 - soft)
    return res, soft, target, loss_weight


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--figure", type=str, default=None,
                    help="optional path for a 2-panel PNG (seed 0)")
    args = ap.parse_args()

    pipeline = WhittakerPipeline()
    rows = []
    for seed in range(args.seeds):
        truth, y, weights = make_scene(seed)
        res, soft, target, _ = build_target(pipeline, y, weights)

        contaminated = np.zeros(T, bool)
        contaminated[W1] = True
        contaminated[W2] = True
        contaminated[np.abs(y - truth) > 0.08] = True
        clear = (weights > 0.5) & ~contaminated
        clear_flagged = int(((soft == 0) & clear).sum())

        rows.append({
            "seed": seed,
            "clear_obs": int(clear.sum()),
            "clear_flagged": clear_flagged,
            "uncertain_fired_w2": bool(res.uncertain_mask[W2].all()),
            "leak_err": float(np.abs(target[W2_SLIPS] - truth[W2_SLIPS]).mean()),
            "leak_soft": float(soft[W2_SLIPS].mean()),
            "mae_w1": float(np.abs(target[W1] - truth[W1]).mean()),
            "mae_year": float(np.abs(target - truth).mean()),
        })

    print(f"{'seed':>4} {'clear obs':>9} {'flagged':>8} {'W2 uncertain':>12} "
          f"{'leak soft':>9} {'leak err':>8} {'W1 MAE':>7} {'year MAE':>8}")
    for r in rows:
        print(f"{r['seed']:>4} {r['clear_obs']:>9} {r['clear_flagged']:>8} "
              f"{str(r['uncertain_fired_w2']):>12} {r['leak_soft']:>9.2f} "
              f"{r['leak_err']:>8.3f} {r['mae_w1']:>7.3f} {r['mae_year']:>8.3f}")

    n_flagged = sum(r["clear_flagged"] for r in rows)
    n_clear = sum(r["clear_obs"] for r in rows)
    print(f"\nclear obs flagged as cloud: {n_flagged}/{n_clear} "
          f"({100 * n_flagged / n_clear:.0f}%) - detector hair-trigger")
    print("uncertain-month leak fired in all seeds:",
          all(r["uncertain_fired_w2"] for r in rows))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        truth, y, weights = make_scene(0)
        res, soft, target, _ = build_target(pipeline, y, weights)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for ax in axes:
            ax.axvspan(W1[0], W1[-1], color="#69c", alpha=0.13)
            ax.axvspan(W2[0], W2[-1], color="#c96", alpha=0.13)
        kept = weights > 0.5
        axes[0].plot(DAYS, truth, color="#444", lw=1.3, label="true NDVI (known)")
        axes[0].scatter(DAYS[kept], y[kept], s=9, color="#2a7", alpha=0.65,
                        label="obs kept by primary mask")
        axes[0].scatter(DAYS[~kept], y[~kept], s=9, color="#c7c7c7", alpha=0.6,
                        label="obs removed by primary mask")
        axes[0].plot(DAYS, res.ndvi_smooth, color="#d33", lw=1.8,
                     label="detect_clouds envelope")
        axes[0].legend(fontsize=8, loc="upper left")
        axes[0].set_ylabel("NDVI")
        axes[1].plot(DAYS, truth, color="#444", lw=1.3, label="true NDVI")
        axes[1].plot(DAYS, target, color="#d33", lw=1.5, label="training target")
        axes[1].scatter(W2_SLIPS, target[W2_SLIPS], marker="x", s=60,
                        color="#a0f", zorder=6, label="uncertain-month 50% leaks")
        axes[1].legend(fontsize=8, loc="upper left")
        axes[1].set_ylabel("NDVI")
        axes[1].set_xlabel("day of year")
        plt.tight_layout()
        plt.savefig(args.figure, dpi=130)
        print("figure saved:", args.figure)


if __name__ == "__main__":
    main()
