"""NdviVAEv2: β-VAE with reflect padding + SSIM+L1 reconstruction loss.

Fixes vs v1 (NdviVAE):
  - Reflect padding in ALL convolutions → eliminates boundary ring artifacts
  - SSIM + L1 loss alongside MSE → sharper, less blurry reconstructions
  - Same 256×256×2 → 32×32×4 compression → drop-in replacement for diffusion

State dict key/shape compatibility with NdviVAE
------------------------------------------------
All internal parameter tensors have identical names and shapes as v1.
To use v2 weights in EDMDiffusion, pass vae_version='v2' (one-line change).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ──────────────────────────── loss helpers ────────────────────────────────────


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """1 − mean SSIM (lower = better). Operates on a single-channel tensor.

    Cast to float32: variance is computed as E[X²]−E[X]², which catastrophically
    cancels in bf16 on near-flat patches, producing negative variance → NaN grad.
    """
    pred, target = pred.float(), target.float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p = window // 2
    mu1   = F.avg_pool2d(pred,          window, stride=1, padding=p)
    mu2   = F.avg_pool2d(target,        window, stride=1, padding=p)
    mu1sq = mu1.pow(2)
    mu2sq = mu2.pow(2)
    mu12  = mu1 * mu2
    s1sq  = (F.avg_pool2d(pred.pow(2),   window, stride=1, padding=p) - mu1sq).clamp(min=0)
    s2sq  = (F.avg_pool2d(target.pow(2), window, stride=1, padding=p) - mu2sq).clamp(min=0)
    s12   = F.avg_pool2d(pred * target,  window, stride=1, padding=p) - mu12
    num   = (2 * mu12 + C1) * (2 * s12 + C2)
    den   = (mu1sq + mu2sq + C1) * (s1sq + s2sq + C2)
    return 1.0 - (num / den).mean()


# ──────────────────────────── building blocks ─────────────────────────────────
# Attribute names mirror NdviVAE v1 so state dict keys are identical.


class _ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode="reflect")
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, padding_mode="reflect")
        self.skip  = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Downsample(nn.Module):
    # stride-2: padding_mode='reflect' not supported for stride>1 in PyTorch,
    # so we manually reflect-pad before the strided conv.
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (1, 1, 1, 1), mode="reflect"))


class _Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ──────────────────────────── VAE ─────────────────────────────────────────────


class NdviVAEv2(nn.Module):
    """8× spatial compression: 256×256×2 ↔ 32×32×4.

    Identical API and state-dict structure to NdviVAE (v1).
    Drop-in replacement once EDMDiffusion is updated to vae_version='v2'.
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 4,
        channel_widths: tuple = (32, 64, 128, 256),
    ):
        super().__init__()
        cw = channel_widths

        # encoder: 256 → 128 → 64 → 32
        enc: list[nn.Module] = [
            nn.Conv2d(in_channels, cw[0], 3, padding=1, padding_mode="reflect")
        ]
        for i in range(len(cw) - 1):
            enc += [
                _ResBlock(cw[i], cw[i]),
                _Downsample(cw[i]),
                nn.Conv2d(cw[i], cw[i + 1], 1),
            ]
        enc += [_ResBlock(cw[-1], cw[-1]), _ResBlock(cw[-1], cw[-1])]
        self.encoder     = nn.Sequential(*enc)
        self.mu_head     = nn.Conv2d(cw[-1], latent_dim, 1)
        self.logvar_head = nn.Conv2d(cw[-1], latent_dim, 1)

        # decoder: 32 → 64 → 128 → 256
        dec: list[nn.Module] = [
            nn.Conv2d(latent_dim, cw[-1], 3, padding=1, padding_mode="reflect"),
            _ResBlock(cw[-1], cw[-1]),
            _ResBlock(cw[-1], cw[-1]),
        ]
        for i in range(len(cw) - 1, 0, -1):
            dec += [
                _Upsample(cw[i]),
                nn.Conv2d(cw[i], cw[i - 1], 1),
                _ResBlock(cw[i - 1], cw[i - 1]),
            ]
        dec += [
            nn.GroupNorm(min(8, cw[0]), cw[0]),
            nn.SiLU(),
            nn.Conv2d(cw[0], in_channels, 3, padding=1, padding_mode="reflect"),
        ]
        self.decoder = nn.Sequential(*dec)

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h      = self.encoder(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-30, 20)
        eps    = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        z      = mu + eps * (0.5 * logvar).exp()
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


# ──────────────────────────── Lightning module ────────────────────────────────


class NdviVAEv2Module(pl.LightningModule):
    """Lightning wrapper with SSIM + L1 + MSE reconstruction loss + β-KL annealing.

    Reconstruction loss (applied to NDVI channel, i.e. channel 0):
        recon = MSE(both) + λ_ssim · SSIM_loss(ch0) + λ_l1 · L1(ch0)

    Defaults give roughly equal weighting between MSE (absolute accuracy),
    SSIM (structural sharpness), and L1 (edge fidelity).
    """

    def __init__(
        self,
        vae: NdviVAEv2 | None = None,
        lr: float = 1e-3,
        beta_final: float = 1e-3,
        beta_anneal_steps: int = 5000,
        lambda_ssim: float = 0.15,
        lambda_l1: float = 0.10,
    ):
        super().__init__()
        self.vae               = vae or NdviVAEv2()
        self.lr                = lr
        self.beta_final        = beta_final
        self.beta_anneal_steps = beta_anneal_steps
        self.lambda_ssim       = lambda_ssim
        self.lambda_l1         = lambda_l1

    def _beta(self) -> float:
        t = min(self.global_step / max(self.beta_anneal_steps, 1), 1.0)
        return float(t * self.beta_final)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        target = torch.cat([batch["target_ndvi"], batch["loss_weight"]], dim=1)
        recon, mu, logvar = self.vae(target)

        mse  = F.mse_loss(recon, target)
        ssim = _ssim_loss(recon[:, :1], target[:, :1])   # NDVI channel only
        l1   = F.l1_loss(recon[:, :1], target[:, :1])    # NDVI channel only

        loss_recon = mse + self.lambda_ssim * ssim + self.lambda_l1 * l1
        kl         = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss       = loss_recon + self._beta() * kl

        psnr = -10.0 * torch.log10(mse.detach().clamp(min=1e-10))
        sync = (stage == "val")
        self.log(f"{stage}/recon", loss_recon, prog_bar=True,             sync_dist=sync)
        self.log(f"{stage}/mse",   mse,                                   sync_dist=sync)
        self.log(f"{stage}/ssim",  ssim,                                  sync_dist=sync)
        self.log(f"{stage}/kl",    kl,                                    sync_dist=sync)
        self.log(f"{stage}/loss",  loss,        prog_bar=True,             sync_dist=sync)
        self.log(f"{stage}/psnr",  psnr,        prog_bar=(stage == "val"), sync_dist=sync)
        if stage == "train":
            self.log("train/beta", self._beta())
        return loss

    def training_step(self, batch: dict, _) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.vae.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100_000, eta_min=1e-5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
