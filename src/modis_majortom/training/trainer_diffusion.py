"""DiffusionTrainer — encapsulates the EDMDiffusion training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import datetime
import logging

import numpy as np
import zarr
import torch
import pytorch_lightning as pl
from scipy.ndimage import laplace as _ndimage_laplace
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .callbacks import EMACallback
from ..model.diffusion import EDMDiffusion
from ..transform.dataset import (
    AlignedPatchDataset,
    ERA5Source,
    LAISource,
    MOD13A3Source,
    MODISSource,
    RawGAMODISSource,
    build_sample_index,
)
from ..transform.land_cover import LandCoverSource

log = logging.getLogger(__name__)


@dataclass
class DiffusionDataConfig:
    """Paths and sampling settings from the ``data:`` section of the YAML config."""
    fci_zarr: str
    modis_raw_zarr: str
    era5_path: str
    modis_proc_zarr: str | None = None
    lc_zarr: str | None = None
    mod13a3_zarr: str | None = None   # MOD13A3 monthly NDVI zarr; see MOD13A3Source.
    lai_zarr: str | None = None       # MCD15A3H raw zarr; see LAISource. Requires
                                       # model.add_lai=true to actually train the head.
    max_fci_nan_fraction: float = 1.0
    whittaker_target: bool = False
    max_lap_var: float = 0.0          # Laplacian-variance threshold; 0.0 = disabled
    start_date: str | None = None     # ISO date; restrict FCI dates to >= this value
    use_whittaker_cloud_mask: bool = False  # exclude ±7d gap pixels from loss
    ndvi_oversample: float = 0.0      # expand train set by this factor; 0 = disabled
    ndvi_oversample_mode: str = "mean"  # "mean" | "frac_above"
    ndvi_oversample_threshold: float = 0.5  # pixel threshold for "frac_above" mode
    ndvi_oversample_alpha: float = 1.0  # weight = raw_score ** alpha; >1 = more aggressive


@dataclass
class DiffusionTrainConfig:
    """Hyperparameters from the ``train:`` section of the YAML config."""
    batch_size: int
    num_workers: int = 4
    lr: float = 1e-4
    val_every: int = 5000
    max_steps: int = 200_000
    save_every: int = 5000
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"
    ema_decay: float = 0.9999
    seed: int = 42
    root_dir: str | None = None  # Lightning default_root_dir; set to scratch path on HPC

    @classmethod
    def from_dict(cls, d: dict) -> "DiffusionTrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def _filter_noisy_patches(
    samples: list[tuple[str, str]],
    modis_zarr_path: str,
    max_lap_var: float,
) -> list[tuple[str, str]]:
    """Drop patches whose Laplacian variance (at 32×32) exceeds *max_lap_var*.

    Downsamples to 32×32 via average pooling (reshape+mean), then computes
    scipy.ndimage.laplace — no cv2 dependency required.
    """
    store = zarr.open(modis_zarr_path, mode="r")
    ndvi_grp = store["patches"]["ndvi_envelope"]
    kept, dropped = [], 0
    for gid, date in samples:
        try:
            ndvi = ndvi_grp[date][gid][:].astype(np.float64)
        except Exception:
            kept.append((gid, date))
            continue
        h, w = ndvi.shape
        small = ndvi.reshape(32, h // 32, 32, w // 32).mean(axis=(1, 3))
        lap = _ndimage_laplace(small)
        if float(np.var(lap)) <= max_lap_var:
            kept.append((gid, date))
        else:
            dropped += 1
    log.info(
        "Laplacian filter (max_lap_var=%.3f): kept %d / %d (dropped %d)",
        max_lap_var, len(kept), len(samples), dropped,
    )
    return kept


def _compute_ndvi_weights(
    samples: list[tuple[str, str]],
    modis_proc_zarr_path: str,
    mode: str = "mean",
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute a per-sample NDVI score used for oversampling.

    Parameters
    ----------
    mode : "mean" — mean NDVI of the patch (ignoring negative/invalid values).
           "frac_above" — fraction of pixels with NDVI > threshold.
    alpha : raise the raw score to this power before normalizing (>1 = stronger
            oversampling of high-NDVI patches; 1 = linear, 2 = quadratic).

    Returns
    -------
    weights : (N,) float64 array, all values >= 1e-6 (floor so no sample vanishes).
    """
    store = zarr.open(modis_proc_zarr_path, mode="r")
    ndvi_grp = store["patches"]["ndvi_envelope"]
    scores = np.empty(len(samples), dtype=np.float64)
    for i, (gid, date) in enumerate(samples):
        try:
            ndvi = ndvi_grp[date][gid][:].astype(np.float32)
            valid = ndvi[np.isfinite(ndvi) & (ndvi > -1.0)]
            if valid.size == 0:
                scores[i] = 0.0
            elif mode == "frac_above":
                scores[i] = float((valid > threshold).mean())
            else:  # "mean"
                scores[i] = float(np.clip(valid.mean(), 0.0, 1.0))
        except Exception:
            scores[i] = 0.0
    scores = scores ** alpha
    scores = np.maximum(scores, 1e-6)  # floor: every sample has non-zero probability
    log.info(
        "NDVI weights (%s, alpha=%.1f): min=%.4f mean=%.4f max=%.4f",
        mode, alpha, scores.min(), scores.mean(), scores.max(),
    )
    return scores


def _expand_samples_by_ndvi(
    samples: list[tuple[str, str]],
    weights: np.ndarray,
    oversample_factor: float,
) -> list[tuple[str, str]]:
    """Repeat samples proportionally to their NDVI weight.

    High-weight samples appear more often; every sample appears at least once.
    The expanded list is approximately ``oversample_factor`` × longer than the
    input, and using it with ``shuffle=True`` in the DataLoader reproduces the
    intended NDVI distribution without needing a custom sampler — compatible
    with PL's DDP DistributedSampler.
    """
    target_total = max(len(samples), round(len(samples) * oversample_factor))
    w_norm = weights / weights.sum()
    counts = np.maximum(1, np.round(w_norm * target_total).astype(int))
    expanded = [s for s, c in zip(samples, counts) for _ in range(int(c))]
    # log per-quintile repeat stats
    quintiles = np.percentile(weights, [20, 40, 60, 80])
    log.info(
        "NDVI oversample: %d → %d samples (%.1f×); "
        "weight quintile repeats: Q20=%.1f Q40=%.1f Q60=%.1f Q80=%.1f",
        len(samples), len(expanded), len(expanded) / len(samples),
        *[counts[weights >= q].mean() if (weights >= q).any() else 0.0 for q in quintiles],
    )
    return expanded


class DiffusionTrainer:
    """Builds dataloaders, assembles EDMDiffusion, and runs training.

    Args:
        data_cfg: Dataset paths and filtering thresholds.
        train_cfg: Optimiser and training loop hyperparameters.
        model_cfg: Keyword arguments forwarded to ``EDMDiffusion``.
    """

    def __init__(
        self,
        data_cfg: DiffusionDataConfig,
        train_cfg: DiffusionTrainConfig,
        model_cfg: dict[str, Any],
    ) -> None:
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

    # ── data ────────────────────────────────────────────────────────────────────

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders.

        The train/val split is seeded with ``train_cfg.seed``.  When
        ``data_cfg.ndvi_oversample > 0``, the train sample list is expanded
        by repeating high-NDVI (grid_id, date) pairs proportionally to their
        NDVI score (see ``_expand_samples_by_ndvi``).  The val set is never
        oversampled.  The expansion is done at the sample-list level so the
        resulting DataLoader is compatible with PL's DDP DistributedSampler.
        """
        if self.data_cfg.modis_proc_zarr:
            modis_src = MODISSource(
                raw_zarr_path=self.data_cfg.modis_raw_zarr,
                processed_zarr_path=self.data_cfg.modis_proc_zarr,
                use_gap_mask=self.data_cfg.use_whittaker_cloud_mask,
            )
        else:
            modis_src = RawGAMODISSource(raw_zarr_path=self.data_cfg.modis_raw_zarr)

        era5_src = ERA5Source(
            nc_path=self.data_cfg.era5_path,
            n_days=self.model_cfg.get("era5_days", 15),
        )
        ancillary: dict = {"modis": modis_src, "era5": era5_src}
        if self.data_cfg.lc_zarr:
            ancillary["land_cover"] = LandCoverSource(
                zarr_path=self.data_cfg.lc_zarr,
                variables=["LC_Type1"],
            )

        mod13a3_source = (
            MOD13A3Source(zarr_path=self.data_cfg.mod13a3_zarr)
            if self.data_cfg.mod13a3_zarr else None
        )

        # LAISource is intentionally NOT added to `ancillary` — build_sample_index
        # intersects every source's available_dates()/available_grid_ids(), and
        # LAI is only real on ~1-in-4 dates. Keeping it out preserves the full
        # daily sample index for NDVI; AlignedPatchDataset queries it per-sample
        # and zero-fills on non-composite dates instead.
        # fpar_band is always set: the LAI head jointly predicts LAI+FAPAR
        # (EDMDiffusion._lai_loss requires both target_fpar/fpar_mask whenever
        # add_lai=True), so there's no config path that trains LAI without it.
        lai_source = (
            LAISource(raw_zarr_path=self.data_cfg.lai_zarr, fpar_band="Fpar_500m")
            if self.data_cfg.lai_zarr else None
        )

        fci_dates: list[str] | None = None
        if self.data_cfg.start_date:
            fci_store_tmp = zarr.open(str(self.data_cfg.fci_zarr), mode="r")
            all_fci_dates = sorted(set(k[:10] for k in fci_store_tmp["patches"]["vis_06"].keys()))
            fci_dates = [d for d in all_fci_dates if d >= self.data_cfg.start_date]
            log.info("start_date=%s → %d FCI dates retained", self.data_cfg.start_date, len(fci_dates))

        samples = build_sample_index(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            max_fci_nan_fraction=self.data_cfg.max_fci_nan_fraction,
            dates=fci_dates,
        )
        if self.data_cfg.max_lap_var > 0.0 and self.data_cfg.modis_proc_zarr:
            samples = _filter_noisy_patches(
                samples, self.data_cfg.modis_proc_zarr, self.data_cfg.max_lap_var
            )
        if not samples:
            raise RuntimeError("No samples found — check zarr/netCDF paths in config.")

        # Deterministic train/val split at the sample-list level.
        rng = np.random.default_rng(self.train_cfg.seed)
        idx = rng.permutation(len(samples))
        n_val = max(1, int(len(samples) * 0.05))
        val_samples   = [samples[i] for i in idx[:n_val]]
        train_samples = [samples[i] for i in idx[n_val:]]

        # Optional NDVI oversampling — expand the train list only.
        if self.data_cfg.ndvi_oversample > 0.0 and self.data_cfg.modis_proc_zarr:
            weights = _compute_ndvi_weights(
                train_samples,
                self.data_cfg.modis_proc_zarr,
                mode=self.data_cfg.ndvi_oversample_mode,
                threshold=self.data_cfg.ndvi_oversample_threshold,
                alpha=self.data_cfg.ndvi_oversample_alpha,
            )
            train_samples = _expand_samples_by_ndvi(
                train_samples, weights, self.data_cfg.ndvi_oversample,
            )

        ds_train = AlignedPatchDataset(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            samples=train_samples,
            whittaker_target=self.data_cfg.whittaker_target,
            lai_source=lai_source,
            mod13a3_source=mod13a3_source,
        )
        ds_val = AlignedPatchDataset(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            samples=val_samples,
            whittaker_target=self.data_cfg.whittaker_target,
            lai_source=lai_source,
            mod13a3_source=mod13a3_source,
        )

        dl_kwargs = dict(
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.train_cfg.num_workers > 0,
        )
        return (
            DataLoader(ds_train, shuffle=True, **dl_kwargs),
            DataLoader(ds_val, shuffle=False, **dl_kwargs),
        )

    # ── model ───────────────────────────────────────────────────────────────────

    def build_model(self) -> EDMDiffusion:
        """Instantiate EDMDiffusion from ``model_cfg``."""
        model = EDMDiffusion(**self.model_cfg)
        return model

    # ── quick-sample helper ─────────────────────────────────────────────────────

    def sample(
        self,
        ckpt_path: str,
        steps: int = 20,
        guidance: float = 3.0,
    ) -> None:
        """Load a checkpoint and run one validation batch through the sampler.

        Prints output shape and value ranges — useful for a quick sanity check.
        """
        model = self.build_model()
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.eval()

        _, dl_val = self.build_dataloaders()
        batch = next(iter(dl_val))
        with torch.no_grad():
            out = model.sample(batch, steps=steps, guidance=guidance)
        print(f"sample shape: {out.shape}")
        print(f"  ndvi   range: [{out[:, 0].min():.3f}, {out[:, 0].max():.3f}]")
        print(f"  uncert range: [{out[:, 1].min():.3f}, {out[:, 1].max():.3f}]")

    # ── training ────────────────────────────────────────────────────────────────

    def fit(self, ckpt_path: str | None = None) -> None:
        """Run training.

        Args:
            ckpt_path: Resume a full Lightning checkpoint (model + optimiser state).
        """
        torch.set_float32_matmul_precision("medium")
        dl_train, dl_val = self.build_dataloaders()
        model = self.build_model()

        val_every = self.train_cfg.val_every
        steps_per_epoch = len(dl_train)
        if isinstance(val_every, float):
            # Fraction of epoch (0.0–1.0) — pass directly, DDP-safe
            val_kwargs: dict = {"val_check_interval": val_every}
        else:
            # Integer step count → convert to epochs so DDP per-GPU splitting doesn't break it
            val_kwargs = {"check_val_every_n_epoch": max(1, val_every // steps_per_epoch)}

        trainer = pl.Trainer(
            max_steps=self.train_cfg.max_steps,
            accumulate_grad_batches=self.train_cfg.accumulate_grad_batches,
            default_root_dir=self.train_cfg.root_dir,
            **val_kwargs,
            log_every_n_steps=min(100, steps_per_epoch),
            gradient_clip_val=1.0,
            accelerator="auto",
            devices="auto",
            strategy=DDPStrategy(
                find_unused_parameters=True,
                timeout=datetime.timedelta(hours=2),
            ),
            precision=self.train_cfg.precision,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                    every_n_train_steps=self.train_cfg.save_every,
                    filename="diff-{step:06d}-val_loss={val_loss:.4f}",
                ),
                LearningRateMonitor("step"),
                EMACallback(decay=self.train_cfg.ema_decay),
            ],
        )
        trainer.fit(model, dl_train, dl_val, ckpt_path=ckpt_path)
