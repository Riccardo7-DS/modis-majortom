"""Synthetic ground-truth tests for WhittakerPipeline.detect_clouds.

The scene generator lives in scripts/repro_whittaker_target.py; each test
documents one known failure mode of the current detector. The xfail markers
are strict: once a failure mode is fixed, the corresponding test XPASSes and
the marker must be removed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("modape.whittaker")

from modis_majortom.transform.ndvi_whittaker import WhittakerPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from repro_whittaker_target import (  # noqa: E402
    DOY, T, W1, W2, W2_SLIPS, build_target, make_scene,
)


@pytest.fixture(scope="module")
def pipeline():
    return WhittakerPipeline()


@pytest.fixture(scope="module")
def scene():
    return make_scene(0)


@pytest.mark.xfail(
    strict=True,
    reason="detector hair-trigger: top-riding envelope starves the "
    "positive-residual sigma estimate, so ordinary clear obs get flagged",
)
def test_clear_obs_mostly_unflagged(pipeline, scene):
    truth, y, weights = scene
    _, soft, _, _ = build_target(pipeline, y, weights)
    contaminated = np.zeros(T, bool)
    contaminated[W1] = True
    contaminated[W2] = True
    contaminated[np.abs(y - truth) > 0.08] = True
    clear = (weights > 0.5) & ~contaminated
    false_positive_rate = ((soft == 0) & clear).sum() / clear.sum()
    assert false_positive_rate < 0.20, (
        f"{false_positive_rate:.0%} of genuinely clear obs flagged as cloud"
    )


@pytest.mark.xfail(
    strict=True,
    reason="uncertain-month fallback cancels cloud flags, so contaminated "
    "obs that slipped the primary mask enter the target at 50%",
)
def test_no_contaminated_obs_in_target(pipeline, scene):
    truth, y, weights = scene
    _, soft, target, _ = build_target(pipeline, y, weights)
    # The two W2 observations that slipped the primary mask are ~0.4 NDVI
    # below the truth; a sound target keeps them out.
    leak_err = np.abs(target[W2_SLIPS] - truth[W2_SLIPS]).mean()
    assert leak_err < 0.10, (
        f"target off by {leak_err:.2f} NDVI at slipped cloudy obs "
        f"(soft_score={soft[W2_SLIPS].tolist()})"
    )


def test_uncertain_month_fires(pipeline, scene):
    """Precondition for the leak test: W2 really is an uncertain month."""
    truth, y, weights = scene
    res, _, _, _ = build_target(pipeline, y, weights)
    assert res.uncertain_mask[W2].all()


def test_envelope_is_upper_envelope(pipeline, scene):
    """Sanity: the envelope sits above most clear observations (by design)."""
    truth, y, weights = scene
    res, _, _, _ = build_target(pipeline, y, weights)
    clear = weights > 0.5
    above = (res.ndvi_smooth[clear] >= y[clear] - 1e-6).mean()
    assert above > 0.5
