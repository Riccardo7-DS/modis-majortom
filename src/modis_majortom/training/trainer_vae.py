"""VaeTrainer — encapsulates the VAE training loop for NdviVAE v1 and v2."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from ..transform.dataset import (
    AlignedPatchDataset,
    MODISSource,
    RawGAMODISSource,
    build_sample_index,
)


@dataclass
class VaeDataConfig:
    """Paths and sampling settings read from the ``data:`` section of the YAML config."""
    fci_zarr: str
    modis_raw_zarr: str
    modis_proc_zarr: str | None = None
    max_fci_nan_fraction: float = 1.0


@dataclass
class VaeTrainConfig:
    """Hyperparameters read from the ``train:`` section of the YAML config."""
    batch_size: int
    num_workers: int = 4
    lr: float = 1e-3
    beta_final: float = 1e-3
    beta_anneal_steps: int = 5000
    val_every: int = 2000
    max_steps: int = 100_000
    precision: str = "bf16-mixed"
    seed: int = 42
    # v2-specific loss weights (ignored for v1)
    lambda_ssim: float = 0.15
    lambda_l1: float = 0.10

    @classmethod
    def from_dict(cls, d: dict) -> "VaeTrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class VaeTrainer:
    """Builds dataloaders, assembles the Lightning module, and runs training.

    Args:
        data_cfg: Dataset paths and filtering thresholds.
        train_cfg: Optimiser and training loop hyperparameters.
        model_cfg: Keyword arguments forwarded to the VAE constructor.
        model_version: ``"v1"`` (NdviVAE) or ``"v2"`` (NdviVAEv2).
    """

    def __init__(
        self,
        data_cfg: VaeDataConfig,
        train_cfg: VaeTrainConfig,
        model_cfg: dict[str, Any],
        model_version: str = "v1",
    ) -> None:
        if model_version not in ("v1", "v2"):
            raise ValueError(f"model_version must be 'v1' or 'v2', got {model_version!r}")
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.model_version = model_version

    # ── data ────────────────────────────────────────────────────────────────────

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders.

        The train/val split is seeded with ``train_cfg.seed`` to keep it
        reproducible across runs. Changing ``seed`` will produce a different split.
        """
        if self.data_cfg.modis_proc_zarr:
            modis_src = MODISSource(
                raw_zarr_path=self.data_cfg.modis_raw_zarr,
                processed_zarr_path=self.data_cfg.modis_proc_zarr,
            )
        else:
            modis_src = RawGAMODISSource(raw_zarr_path=self.data_cfg.modis_raw_zarr)
        ancillary = {"modis": modis_src}

        samples = build_sample_index(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            max_fci_nan_fraction=self.data_cfg.max_fci_nan_fraction,
        )
        if not samples:
            raise RuntimeError("No (grid_id, date) samples found — check zarr paths in config.")

        n_val = max(1, int(len(samples) * 0.05))
        n_train = len(samples) - n_val
        dataset = AlignedPatchDataset(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            samples=samples,
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

    def build_module(self) -> pl.LightningModule:
        """Instantiate the appropriate VAE Lightning module."""
        if self.model_version == "v1":
            from ..model.vae import NdviVAE, NdviVAEModule
            return NdviVAEModule(
                vae=NdviVAE(**self.model_cfg),
                lr=self.train_cfg.lr,
                beta_final=self.train_cfg.beta_final,
                beta_anneal_steps=self.train_cfg.beta_anneal_steps,
            )
        else:
            from ..model.vae_v2 import NdviVAEv2, NdviVAEv2Module
            return NdviVAEv2Module(
                vae=NdviVAEv2(**self.model_cfg),
                lr=self.train_cfg.lr,
                beta_final=self.train_cfg.beta_final,
                beta_anneal_steps=self.train_cfg.beta_anneal_steps,
                lambda_ssim=self.train_cfg.lambda_ssim,
                lambda_l1=self.train_cfg.lambda_l1,
            )

    # ── training ────────────────────────────────────────────────────────────────

    def fit(
        self,
        ckpt_path: str | None = None,
        init_weights: str | None = None,
    ) -> None:
        """Run training.

        Args:
            ckpt_path: Resume a full Lightning checkpoint (model + optimiser state).
            init_weights: Load model weights only, starting with a fresh optimiser.
        """
        dl_train, dl_val = self.build_dataloaders()
        module = self.build_module()

        if init_weights:
            raw = torch.load(init_weights, map_location="cpu")
            module.load_state_dict(raw.get("state_dict", raw), strict=True)
            print(f"Loaded model weights from {init_weights} (fresh optimizer)")

        ckpt_prefix = "vae" if self.model_version == "v1" else "vae2"
        val_every = self.train_cfg.val_every
        steps_per_epoch = len(dl_train)
        if isinstance(val_every, float) or val_every <= steps_per_epoch:
            val_kwargs: dict = {"val_check_interval": val_every}
        else:
            val_kwargs = {"check_val_every_n_epoch": max(1, val_every // steps_per_epoch)}

        trainer = pl.Trainer(
            max_steps=self.train_cfg.max_steps,
            **val_kwargs,
            log_every_n_steps=min(50, steps_per_epoch),
            gradient_clip_val=1.0,
            accelerator="auto",
            devices="auto",
            precision=self.train_cfg.precision,
            callbacks=[
                ModelCheckpoint(
                    monitor="val/recon",
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                    filename=f"{ckpt_prefix}-{{step:06d}}-{{val/recon:.4f}}",
                ),
                LearningRateMonitor("step"),
                EarlyStopping(
                    monitor="val/recon",
                    patience=10,
                    mode="min",
                    min_delta=1e-4,
                    verbose=True,
                ),
            ],
        )
        trainer.fit(module, dl_train, dl_val, ckpt_path=ckpt_path)
