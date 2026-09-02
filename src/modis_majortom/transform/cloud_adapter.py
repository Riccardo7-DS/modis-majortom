"""Core MODIS cloud-masking utilities and zarr-cube extraction helpers.

The torch/DINOv2-based cloud-segmentation model (``Adapter``, ``CloudAdapterModel``,
``CloudAdapterPipeline``, ``MODISZarrPatchDataset``, and their training helpers
``safe_collate`` / ``resize_to_multiple_of_patch``) has moved to the
``ndvi-diffusion`` sibling package at ``ndvi_diffusion.datasets.cloud_adapter_model``,
which imports the functions below (``generate_cloud_mask``, ``normalize_modis``,
``upsample_500_to_250``, ``extract_modis_cube``) as an external ``modis_majortom``
dependency. This module keeps the pure numpy/zarr cloud-mask generation and
cube-extraction logic — the README-documented core cloud-masking feature — free
of any torch dependency.
"""

import logging
import zarr
import numpy as np
from dataclasses import dataclass
from ..utils import compute_ndvi

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Cloud mask preprocessing
# ─────────────────────────────────────────────

@dataclass
class CloudMaskResult:
    """Outputs of generate_cloud_mask, one per spatial tile."""
    cleaned_mask: np.ndarray       # bool   (H, W) — True = cloudy
    soft_score: np.ndarray         # float32 (H, W) in [0, 1] — loss weight, 1 = confident clear
    meta_mask_channel: np.ndarray  # float32 (H, W) in [0, 1] — extra GFM input channel


def generate_cloud_mask(
    mod09qa_bits: np.ndarray,
    mod35_confidence: np.ndarray,
    blue_band: np.ndarray,
) -> CloudMaskResult:
    """
    Build a cleaned binary cloud mask, a per-pixel soft confidence score,
    and a meta-mask channel for GFM input from three co-registered arrays.

    Parameters
    ----------
    mod09qa_bits : np.ndarray (H, W), uint16
        ``state_1km`` QA bits from MOD09GA.  Bits 0-1 encode cloud state:
        0b00 = clear, 0b01 = cloudy, 0b10 = mixed, 0b11 = not set.
        Pixels with value 1 (cloudy) or 2 (mixed) are treated as cloud.
    mod35_confidence : np.ndarray (H, W), uint8
        MOD35 4-level unobstructed-FOV quality flag stored in zarr as
        ``cloud_mask``:
        0 = confident cloudy, 1 = probably cloudy,
        2 = probably clear,   3 = confident clear.
    blue_band : np.ndarray (H, W), float32
        Band 3 (blue, ~459–479 nm) surface reflectance in physical units
        (i.e. raw DN / 10000).  Used as a spectral veto and soft penalty.

    Returns
    -------
    CloudMaskResult
        cleaned_mask      — bool (H, W), True = cloudy
        soft_score        — float32 (H, W) in [0, 1], use directly as pixel
                            weight in a masked reconstruction loss
        meta_mask_channel — float32 (H, W) in [0, 1], concatenate as an
                            extra channel alongside the imagery fed to the GFM
    """
    from scipy.ndimage import binary_dilation, binary_closing

    # ------------------------------------------------------------------
    # Raw binary cloud flags from each product
    # ------------------------------------------------------------------

    # MOD09GA state_1km — bits 0-1: 1=cloudy, 2=mixed both indicate cloud.
    cloud_bits  = (mod09qa_bits & 0b11).astype(np.uint8)
    mod09_cloud = np.isin(cloud_bits, [1, 2])  # bool (H, W)

    # MOD35 — confidence 0 (confident cloudy) or 1 (probably cloudy) → cloud.
    mod35_cloud = mod35_confidence <= 1  # bool (H, W)

    # ------------------------------------------------------------------
    # Step 2 — Blue-band veto for disagreement pixels
    # Applied BEFORE dilation so corrected cloud info propagates outward.
    #
    # At cloud edges the two products often disagree (one says clear, the
    # other cloudy).  A blue-band threshold of 0.15 provides an independent
    # physical check: surface reflectance above this level in the blue is
    # inconsistent with bare-soil/vegetation and strongly indicates cloud
    # contamination.
    # ------------------------------------------------------------------
    disagree  = mod09_cloud ^ mod35_cloud  # True where masks differ
    blue_veto = disagree & (blue_band > 0.15)

    # Bake the veto into both raw masks so the subsequent dilation step
    # propagates from a more complete starting set of cloudy pixels.
    mod09_cloud = mod09_cloud | blue_veto
    mod35_cloud = mod35_cloud | blue_veto

    # ------------------------------------------------------------------
    # Step 1 — Dilation → union → morphological closing
    # ------------------------------------------------------------------
    kernel = np.ones((5, 5), dtype=bool)  # 5×5 square structuring element

    # Independently dilate each mask to buffer cloud edges by ~2.5 px and
    # absorb the ~1-2 pixel misregistration typical between MOD35 and MOD09.
    mod09_dilated = binary_dilation(mod09_cloud, structure=kernel)
    mod35_dilated = binary_dilation(mod35_cloud, structure=kernel)

    # Union: a pixel is marked cloudy if EITHER product (after dilation) says so.
    combined = mod09_dilated | mod35_dilated

    # Morphological closing (dilation then erosion with the same kernel):
    # fills small interior holes that arise where the two products disagree
    # in the cloud interior, without expanding the outer boundary further.
    cleaned_mask = binary_closing(combined, structure=kernel)  # bool (H, W)

    # ------------------------------------------------------------------
    # Step 3 — Soft cloud confidence score ∈ [0, 1]
    # Three complementary signals are blended; 1 = confident clear.
    # ------------------------------------------------------------------

    # Signal 1 (w=0.5): MOD35 4-level confidence normalised to [0, 1].
    mod35_score = mod35_confidence.astype(np.float32) / 3.0

    # Signal 2 (w=0.3): inverted MOD09 binary flag (pre-veto).
    # Uses the veto-corrected flag so the veto is also reflected in the score.
    mod09_score = (~mod09_cloud).astype(np.float32)

    # Signal 3 (w=0.2): blue-band spectral clarity score.
    # Clip to [0, 0.3], rescale to [0, 1], then invert so low blue → score near 1.
    blue_clipped = np.clip(blue_band, 0.0, 0.3)
    blue_penalty = 1.0 - (blue_clipped / 0.3)  # 1 = clear, 0 = very blue

    soft_score = (
        0.5 * mod35_score +
        0.3 * mod09_score +
        0.2 * blue_penalty
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4 — Meta-mask channel
    # Retain the soft score for clear pixels; zero out cloudy pixels so
    # the model sees a clean "confidence of clearness" signal.
    # ------------------------------------------------------------------
    meta_mask_channel = np.where(cleaned_mask, 0.0, soft_score).astype(np.float32)

    return CloudMaskResult(
        cleaned_mask=cleaned_mask,
        soft_score=soft_score,
        meta_mask_channel=meta_mask_channel,
    )


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class PatchConfig:
    patch_size: int = 256
    stride: int = 256
    min_clear_fraction: float = 0.05


BAND_CONFIG = {
    "mod09_250": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "qc": "QC_250m",
    },
    "mod09_500": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "blue": "sur_refl_b03",
        "state": "state_1km",
    },
}


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def normalize_modis(x, reverse=False):
    if not reverse:
        x[:, 0] = (x[:, 0] - 0.5) / 0.5
        x[:, 1] = (x[:, 1] - 0.5) / 0.5
        x[:, 2] = np.clip(x[:, 2], -1, 1)
        x = np.nan_to_num(x, nan=-1.0, posinf=1.0, neginf=-1.0)

    else:
        x[:, 0] = x[:, 0] * 0.5 + 0.5
        x[:, 1] = x[:, 1] * 0.5 + 0.5

    return x

def upsample_500_to_250(mask_500):
    return np.repeat(np.repeat(mask_500, 2, axis=0), 2, axis=1)


def extract_zarr_store(path, band_name, samples:int=None, patches=True):
    store = zarr.open(path, mode="r")
    if patches:
        patches = store["patches"]
    else:
        patches = store

    band = patches[band_name]
    dates = sorted(band.keys())
    grid_ids = sorted(band[dates[0]].keys())
    if samples is not None:
        grid_ids = grid_ids[:samples]

    return {
        "patches": patches,
        "dates": dates,
        "grid_ids": grid_ids,
    }

def extract_modis_cube(path, product: str, samples: int = None):
    store = zarr.open(path, mode="r")
    patches = store["patches"]
    cfg = BAND_CONFIG[product]
    dates = sorted(patches[cfg["red"]].keys())
    grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())
    if samples is not None:
        grid_ids = grid_ids[:samples]
    return {
        "store": store,
        "patches": patches,
        "cfg": cfg,
        "dates": dates,
        "grid_ids": grid_ids,
        "product": product,
    }


def extract_image(path, product,  t, g, normalize=False):
    store = zarr.open(path, mode="r")
    patches = store["patches"]
    cfg = BAND_CONFIG[product]
    dates = sorted(patches[cfg["red"]].keys())
    grid_ids = sorted(patches[cfg["red"]][dates[0]].keys())
    date = dates[t]
    grid_id = grid_ids[g]
    red = patches[cfg["red"]][date][grid_id][:].astype(np.float32) / 10000.0
    nir = patches[cfg["nir"]][date][grid_id][:].astype(np.float32) / 10000.0
    ndvi = compute_ndvi(nir, red)
    x = np.stack([red, nir, ndvi], axis=0)  #
    if normalize:
        x = normalize_modis(x)

    return x
