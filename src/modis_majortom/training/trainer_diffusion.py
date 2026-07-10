"""DiffusionTrainer — encapsulates the EDMDiffusion training loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, random_split

from .callbacks import EMACallback
from ..model.diffusion import EDMDiffusion
from ..transform.dataset import (
    AlignedPatchDataset,
    ERA5Source,
    MODISSource,
    RawGAMODISSource,
    build_sample_index,
)
from ..transform.land_cover import LandCoverSource


@dataclass
class DiffusionDataConfig:
    """Paths and sampling settings from the ``data:`` section of the YAML config."""
    fci_zarr: str
    modis_raw_zarr: str
    era5_path: str
    modis_proc_zarr: str | None = None
    lc_zarr: str | None = None
    max_fci_nan_fraction: float = 1.0
    whittaker_target: bool = False


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

    @classmethod
    def from_dict(cls, d: dict) -> "DiffusionTrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


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

        The train/val split is seeded with ``train_cfg.seed``.
        """
        if self.data_cfg.modis_proc_zarr:
            modis_src = MODISSource(
                raw_zarr_path=self.data_cfg.modis_raw_zarr,
                processed_zarr_path=self.data_cfg.modis_proc_zarr,
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

        samples = build_sample_index(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            max_fci_nan_fraction=self.data_cfg.max_fci_nan_fraction,
        )
        if not samples:
            raise RuntimeError("No samples found — check zarr/netCDF paths in config.")

        n_val = max(1, int(len(samples) * 0.05))
        n_train = len(samples) - n_val
        dataset = AlignedPatchDataset(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            samples=samples,
            whittaker_target=self.data_cfg.whittaker_target,
        )
        ds_train, ds_val = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.train_cfg.seed),
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
        return EDMDiffusion(**self.model_cfg)

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
                    monitor="val/loss",
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                    every_n_train_steps=self.train_cfg.save_every,
                    filename="diff-{step:06d}-val_loss={val/loss:.4f}",
                ),
                LearningRateMonitor("step"),
                EMACallback(decay=self.train_cfg.ema_decay),
            ],
        )
        trainer.fit(model, dl_train, dl_val, ckpt_path=ckpt_path)
