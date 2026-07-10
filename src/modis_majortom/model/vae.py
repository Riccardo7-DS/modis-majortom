"""NdviVAE: β-VAE compressing 256×256×2 (NDVI, soft_score) → 32×32×4 latent."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class _ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip  = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class NdviVAE(nn.Module):
    """8× spatial compression: 256×256×2 ↔ 32×32×4.

    Channels [NDVI, soft_score] → latent dim 4 (like SD-VAE convention).
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 4,
        channel_widths: tuple = (32, 64, 128, 256),
    ):
        super().__init__()
        cw = channel_widths

        # encoder: 256 → 128 → 64 → 32 (4 downsamples)
        enc: list[nn.Module] = [nn.Conv2d(in_channels, cw[0], 3, padding=1)]
        for i in range(len(cw) - 1):
            enc += [_ResBlock(cw[i], cw[i]), _Downsample(cw[i]),
                    nn.Conv2d(cw[i], cw[i + 1], 1)]
        enc += [_ResBlock(cw[-1], cw[-1]), _ResBlock(cw[-1], cw[-1])]
        self.encoder    = nn.Sequential(*enc)
        self.mu_head    = nn.Conv2d(cw[-1], latent_dim, 1)
        self.logvar_head = nn.Conv2d(cw[-1], latent_dim, 1)

        # decoder: 32 → 64 → 128 → 256
        dec: list[nn.Module] = [
            nn.Conv2d(latent_dim, cw[-1], 3, padding=1),
            _ResBlock(cw[-1], cw[-1]),
            _ResBlock(cw[-1], cw[-1]),
        ]
        for i in range(len(cw) - 1, 0, -1):
            dec += [_Upsample(cw[i]), nn.Conv2d(cw[i], cw[i - 1], 1),
                    _ResBlock(cw[i - 1], cw[i - 1])]
        dec += [
            nn.GroupNorm(min(8, cw[0]), cw[0]),
            nn.SiLU(),
            nn.Conv2d(cw[0], in_channels, 3, padding=1),
        ]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z, mu, logvar). Reparametrises during training, uses mu at inference."""
        h = self.encoder(x)
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


class NdviVAEModule(pl.LightningModule):
    """Lightning wrapper for NdviVAE with β-KL annealing."""

    def __init__(
        self,
        vae: NdviVAE | None = None,
        lr: float = 1e-3,
        beta_final: float = 1e-3,
        beta_anneal_steps: int = 5000,
    ):
        super().__init__()
        self.vae               = vae or NdviVAE()
        self.lr                = lr
        self.beta_final        = beta_final
        self.beta_anneal_steps = beta_anneal_steps

    def _beta(self) -> float:
        t = min(self.global_step / max(self.beta_anneal_steps, 1), 1.0)
        return float(t * self.beta_final)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        target = torch.cat([batch["target_ndvi"], batch["loss_weight"]], dim=1)
        recon, mu, logvar = self.vae(target)
        loss_recon = F.mse_loss(recon, target)
        kl   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = loss_recon + self._beta() * kl
        psnr = -10.0 * torch.log10(loss_recon.detach().clamp(min=1e-10))
        sync = (stage == "val")
        self.log(f"{stage}/recon", loss_recon, prog_bar=True,            sync_dist=sync)
        self.log(f"{stage}/kl",    kl,                                   sync_dist=sync)
        self.log(f"{stage}/loss",  loss,        prog_bar=True,            sync_dist=sync)
        self.log(f"{stage}/psnr",  psnr,        prog_bar=(stage == "val"), sync_dist=sync)
        if stage == "train":
            self.log("train/beta", self._beta())
        return loss

    def training_step(self, batch: dict, _) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _) -> None:
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=50_000, eta_min=1e-5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
