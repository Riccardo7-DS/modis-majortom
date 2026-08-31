"""VaeTrainer — encapsulates the VAE training loop for NdviVAE v1 and v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from ..transform.dataset import (
    AlignedPatchDataset,
    MODISSource,
    RawGAMODISSource,
    build_sample_index,
)
from .trainer_diffusion import _compute_ndvi_weights, _expand_samples_by_ndvi


@dataclass
class VaeDataConfig:
    """Paths and sampling settings read from the ``data:`` section of the YAML config."""
    fci_zarr: str
    modis_raw_zarr: str
    modis_proc_zarr: str | None = None
    max_fci_nan_fraction: float = 1.0
    max_ndvi_lap_var: float = 0.0         # Laplacian-variance threshold; 0.0 = disabled
    ndvi_oversample: float = 0.0          # expand train set by this factor; 0 = disabled
    ndvi_oversample_mode: str = "mean"    # "mean" | "frac_above"
    ndvi_oversample_threshold: float = 0.5  # pixel threshold for "frac_above" mode
    ndvi_oversample_alpha: float = 1.0    # weight = raw_score ** alpha; >1 = more aggressive


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
    early_stopping_patience: int = 40
    precision: str = "bf16-mixed"
    seed: int = 42
    # v2-specific loss weights (ignored for v1)
    lambda_ssim: float = 0.15
    lambda_l1: float = 0.10
    lambda_sobel: float = 0.0
    lambda_amp: float = 0.0
    amp_weight_slope: float = 2.0
    ndvi_mean: float = 0.3
    ndvi_alpha: float = 0.0

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
        if model_version not in ("v1", "v2", "v3"):
            raise ValueError(f"model_version must be 'v1', 'v2', or 'v3', got {model_version!r}")
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.model_version = model_version

    # ── data ────────────────────────────────────────────────────────────────────

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders.

        The train/val split is seeded with ``train_cfg.seed`` to keep it
        reproducible across runs.  When ``data_cfg.ndvi_oversample > 0``, the
        train sample list is expanded by repeating high-NDVI patches proportionally
        to their NDVI score (see ``_expand_samples_by_ndvi``).  The val set is
        never oversampled.
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
            max_ndvi_lap_var=self.data_cfg.max_ndvi_lap_var,
        )
        if not samples:
            raise RuntimeError("No (grid_id, date) samples found — check zarr paths in config.")

        # Deterministic train/val split at the sample-list level so oversampling
        # only affects train without leaking into val.
        rng = np.random.default_rng(self.train_cfg.seed)
        idx = rng.permutation(len(samples))
        n_val = max(1, int(len(samples) * 0.05))
        val_samples   = [samples[i] for i in idx[:n_val]]
        train_samples = [samples[i] for i in idx[n_val:]]

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
        )
        ds_val = AlignedPatchDataset(
            fci_store_path=self.data_cfg.fci_zarr,
            ancillary_sources=ancillary,
            samples=val_samples,
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
        elif self.model_version == "v2":
            from ..model.vae_v2 import NdviVAEv2, NdviVAEv2Module
            return NdviVAEv2Module(
                vae=NdviVAEv2(**self.model_cfg),
                lr=self.train_cfg.lr,
                beta_final=self.train_cfg.beta_final,
                beta_anneal_steps=self.train_cfg.beta_anneal_steps,
                lambda_ssim=self.train_cfg.lambda_ssim,
                lambda_l1=self.train_cfg.lambda_l1,
                lambda_sobel=self.train_cfg.lambda_sobel,
                lambda_amp=self.train_cfg.lambda_amp,
                amp_weight_slope=self.train_cfg.amp_weight_slope,
                ndvi_mean=self.train_cfg.ndvi_mean,
                ndvi_alpha=self.train_cfg.ndvi_alpha,
            )
        else:  # v3
            from ..model.vae_v2 import NdviVAEv3, NdviVAEv2Module
            return NdviVAEv2Module(
                vae=NdviVAEv3(**self.model_cfg),  # use_output_norm passed via model_cfg if set
                lr=self.train_cfg.lr,
                beta_final=self.train_cfg.beta_final,
                beta_anneal_steps=self.train_cfg.beta_anneal_steps,
                lambda_ssim=self.train_cfg.lambda_ssim,
                lambda_l1=self.train_cfg.lambda_l1,
                lambda_sobel=self.train_cfg.lambda_sobel,
                lambda_amp=self.train_cfg.lambda_amp,
                amp_weight_slope=self.train_cfg.amp_weight_slope,
                ndvi_mean=self.train_cfg.ndvi_mean,
                ndvi_alpha=self.train_cfg.ndvi_alpha,
            )

    # ── training ────────────────────────────────────────────────────────────────

    def fit(
        self,
        ckpt_path: str | None = None,
        init_weights: str | None = None,
        compile_model: bool = False,
    ) -> None:
        """Run training.

        Args:
            ckpt_path: Resume a full Lightning checkpoint (model + optimiser state).
            init_weights: Load model weights only, starting with a fresh optimiser.
            compile_model: Apply torch.compile (mode='reduce-overhead') for ~20-40%
                speedup on H100. Requires Triton — skip if Triton is broken on cluster.
        """
        # cuDNN autotuner: fixed 256×256 inputs → finds fastest conv kernel after warmup
        torch.backends.cudnn.benchmark = True
        # TF32 for matmul (on by default on Ampere+ but be explicit)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        dl_train, dl_val = self.build_dataloaders()
        module = self.build_module()

        if init_weights:
            raw = torch.load(init_weights, map_location="cpu")
            sd  = raw.get("state_dict", raw)
            missing, unexpected = module.load_state_dict(sd, strict=False)
            if missing:
                print(f"  Missing keys (will be randomly initialised): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected}")
            print(f"Loaded model weights from {init_weights} (fresh optimizer, strict=False)")

        if compile_model:
            module.vae = torch.compile(module.vae, mode="reduce-overhead")
            print("torch.compile enabled (mode=reduce-overhead)")

        ckpt_prefix = {"v1": "vae", "v2": "vae2", "v3": "vae3"}[self.model_version]
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
            strategy=DDPStrategy(find_unused_parameters=False),
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
                    patience=self.train_cfg.early_stopping_patience,
                    mode="min",
                    min_delta=1e-4,
                    verbose=True,
                    strict=False,      # don't crash if metric missing on resume
                ),
            ],
        )
        trainer.fit(module, dl_train, dl_val, ckpt_path=ckpt_path)
