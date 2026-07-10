"""EDMDiffusion: Lightning module wrapping VAE + conditioners + denoising UNet.

Noise schedule: Karras et al. 2022 EDM (https://arxiv.org/abs/2206.00364).
Sampler:        DPMPP-2M (20 NFE) for inference.
CFG:            Conditioning signals dropped independently with probability cfg_p
                during training; guided at inference with scale w.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml

from .vae    import NdviVAE
from .vae_v2 import NdviVAEv2
from .conditioning import ERA5TokenEncoder, MTGFCIEncoder, LandCoverEmbedder
from .unet import DenoisingUNet2D


# ──────────────────────────── EDM schedule ───────────────────────────────────


class EDMSchedule:
    """Karras 2022 EDM preconditioners and loss weighting."""

    def __init__(self, sigma_data: float = 1.0):
        self.sigma_data = sigma_data

    def preconditioners(self, sigma: torch.Tensor) -> tuple:
        """sigma shape: (B,) or (B,1,1,1)."""
        s2      = self.sigma_data ** 2
        c_skip  = s2 / (sigma ** 2 + s2)
        c_out   = sigma * self.sigma_data / (sigma ** 2 + s2).sqrt()
        c_in    = 1.0 / (sigma ** 2 + s2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        s2 = self.sigma_data ** 2
        return (sigma ** 2 + s2) / (sigma * self.sigma_data) ** 2

    def sample_sigma(
        self, n: int, device: torch.device,
        P_mean: float = -1.2, P_std: float = 1.2,
    ) -> torch.Tensor:
        return (torch.randn(n, device=device) * P_std + P_mean).exp()


# ──────────────────────────── main module ────────────────────────────────────


class EDMDiffusion(pl.LightningModule):
    """Full latent diffusion model: VAE (frozen) + conditioners + UNet denoiser.

    Batch keys consumed
    -------------------
    ``X``            (B, 6, 3, 256, 256)   MTG FCI conditioning
    ``era5``         (B, 12, 15, 256, 256) ERA5-Land temporal features
    ``target_ndvi``  (B, 1, 256, 256)      NDVI target
    ``loss_weight``  (B, 1, 256, 256)      soft_score (used as uncertainty channel)

    Conditioning architecture
    -------------------------
    Both ERA5 and MTG produce (B, N, D_ctx) token sequences that are concatenated
    and fed into the UNet's cross-attention layers:
      - MTGFCIEncoder:   (B, 6, 3, 256, 256) → (B, 256, D_ctx)  spatial tokens
      - ERA5TokenEncoder:(B, 12,15, 256, 256) → (B,  15, D_ctx)  temporal tokens
      - combined context: (B, 271, D_ctx)
    UNet input channels: latent_dim (+ lc_out_channels if land cover enabled).

    Parameters
    ----------
    vae_ckpt :
        Path to a NdviVAEModule checkpoint produced by ``train_vae.py``.
        VAE weights are loaded from the ``vae.*`` sub-keys and frozen.
    era5_vars, era5_days :
        ERA5Source configuration (must match the dataset).
    era5_channels :
        Kept for config backward-compatibility; no longer controls UNet in_channels.
    mtg_times, mtg_bands :
        MTG FCI tensor dimensions.
    context_dim :
        Cross-attention context dim (shared between encoders & UNet).
    latent_dim :
        Must match the NdviVAE latent_dim (default 4).
    sigma_data :
        RMS of the training latents; 1.0 by default (re-estimate after VAE training).
    cfg_p :
        Probability of dropping each conditioning signal for classifier-free guidance.
    lr :
        AdamW learning rate.
    """

    def __init__(
        self,
        vae_ckpt: str | None = None,
        vae_version: str = "v1",
        era5_vars: int = 12,
        era5_days: int = 15,
        era5_channels: int = 64,
        mtg_times: int = 6,
        mtg_bands: int = 3,
        context_dim: int = 512,
        latent_dim: int = 4,
        sigma_data: float = 1.0,
        latent_channel_scale: list[float] | None = None,
        cfg_p: float = 0.1,
        lr: float = 1e-4,
        lc_n_classes: int = 18,
        lc_embed_dim: int = 32,
        lc_out_channels: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vae_ckpt"])
        self.cfg_p = cfg_p
        self.lr    = lr

        # ── VAE (frozen) ─────────────────────────────────────────────────
        self.vae = NdviVAEv2(latent_dim=latent_dim) if vae_version == "v2" \
                   else NdviVAE(latent_dim=latent_dim)
        if vae_ckpt:
            raw = torch.load(vae_ckpt, map_location="cpu")
            sd  = raw.get("state_dict", raw)
            vae_sd = {k.removeprefix("vae."): v
                      for k, v in sd.items() if k.startswith("vae.")}
            if vae_sd:
                self.vae.load_state_dict(vae_sd, strict=True)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        # Per-channel latent normalisation: divide mu by per-channel RMS so
        # each channel has unit variance entering the UNet. Measured from the
        # training data after VAE training; see scripts/diagnose_pipeline.py.
        # Default [1,1,1,1] = no normalisation (backward-compatible).
        scale = torch.tensor(
            latent_channel_scale if latent_channel_scale else [1.0] * latent_dim,
            dtype=torch.float32,
        )
        self.register_buffer("latent_scale", scale)

        # ── conditioners ─────────────────────────────────────────────────
        # ERA5: (B,V,T,H,W) → (B, n_days, context_dim) temporal tokens
        self.era5_enc = ERA5TokenEncoder(era5_vars, era5_days, context_dim)
        # MTG: (B,T,C,H,W) → (B, 256, context_dim) spatial tokens
        self.mtg_enc  = MTGFCIEncoder(mtg_times, mtg_bands, context_dim)
        self.lc_emb   = LandCoverEmbedder(n_classes=lc_n_classes,
                                           embed_dim=lc_embed_dim,
                                           out_channels=lc_out_channels)

        # UNet input: noisy latent + optional LC feature map (ERA5 now via cross-attn)
        in_ch = latent_dim + self.lc_emb.out_channels
        self.unet     = DenoisingUNet2D(in_ch, latent_dim, context_dim=context_dim)
        self.schedule = EDMSchedule(sigma_data)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _norm_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.latent_scale.view(1, -1, 1, 1)

    def _unnorm_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_scale.view(1, -1, 1, 1)

    @torch.no_grad()
    def _encode_target(self, batch: dict) -> torch.Tensor:
        # Use constant soft_score=1 instead of batch["loss_weight"]:
        # the binary cloud mask has a systematic rectangular border (alpha floor)
        # that gets encoded into the latent and the diffusion model learns to reproduce
        # it as a square artifact.  Constant 1 gives clean, spatially unbiased latents.
        ones   = torch.ones_like(batch["target_ndvi"])
        target = torch.cat([batch["target_ndvi"], ones], dim=1)
        _, mu, _ = self.vae.encode(target)
        return self._norm_latent(mu)

    def _condition(
        self, batch: dict,
        drop_mtg: bool = False,
        drop_era5: bool = False,
    ) -> torch.Tensor:
        """Encode all conditioning signals and return a combined context token sequence.

        Returns
        -------
        ctx : (B, 271, D_ctx)
            Concatenation of MTG spatial tokens (256) and ERA5 temporal tokens (15).
        """
        era5 = torch.nan_to_num(batch["era5"], nan=0.0, posinf=0.0, neginf=0.0)
        X    = torch.nan_to_num(batch["X"],    nan=0.0, posinf=0.0, neginf=0.0)
        era5_tokens = self.era5_enc(era5)   # (B, 15, D_ctx)
        mtg_tokens  = self.mtg_enc(X)       # (B, 256, D_ctx)
        if drop_era5:
            era5_tokens = torch.zeros_like(era5_tokens)
        if drop_mtg:
            mtg_tokens = torch.zeros_like(mtg_tokens)
        return torch.cat([mtg_tokens, era5_tokens], dim=1)  # (B, 271, D_ctx)

    def _denoiser(
        self,
        z_noisy: torch.Tensor,                  # (B, latent_dim, 32, 32)
        sigma: torch.Tensor,                    # (B,)
        ctx: torch.Tensor,                      # (B, N_ctx, D_ctx)
        lc_tensor: torch.Tensor | None = None,  # (B, 1, 256, 256)
    ) -> torch.Tensor:
        sig4d = sigma.view(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.schedule.preconditioners(sig4d)

        inp = c_in * z_noisy
        lc  = self.lc_emb(lc_tensor)
        if lc is not None:
            inp = torch.cat([inp, lc], dim=1)

        F_theta = self.unet(inp, c_noise.view(-1), ctx)
        return c_skip * z_noisy + c_out * F_theta

    # ── Lightning API ─────────────────────────────────────────────────────────

    @staticmethod
    def _soft_mask(batch: dict, latent: torch.Tensor) -> torch.Tensor:
        """Downsample soft_score to latent space (32²) with alpha floor removed.

        batch["loss_weight"] = soft + alpha*(1-soft) ∈ [0.2, 1.0] (alpha=0.2).
        Remapping to [0, 1] gives zero weight to pixels outside the MODIS footprint
        so the model gets no gradient signal there and predicts freely from conditioning.
        """
        soft = batch["loss_weight"].to(latent.device)                          # (B, 1, 256, 256)
        soft_lat = torch.nn.functional.avg_pool2d(soft, kernel_size=8, stride=8)  # (B, 1, 32, 32)
        return (soft_lat - 0.2).clamp(min=0) / 0.8                            # remap [0.2,1]→[0,1]

    def training_step(self, batch: dict, _) -> torch.Tensor:
        z     = self._encode_target(batch)
        sigma = self.schedule.sample_sigma(z.shape[0], z.device)
        noise = torch.randn_like(z)
        z_noisy = z + sigma[:, None, None, None] * noise

        # Per-batch CFG dropout (independent for each conditioning branch)
        drop_mtg  = bool(torch.rand(()) < self.cfg_p)
        drop_era5 = bool(torch.rand(()) < self.cfg_p)
        ctx       = self._condition(batch, drop_mtg, drop_era5)
        lc_tensor = batch.get("land_cover")

        D_theta   = self._denoiser(z_noisy, sigma, ctx, lc_tensor)
        w         = self.schedule.loss_weight(sigma)[:, None, None, None].clamp(max=1e4)
        soft_mask = self._soft_mask(batch, z)
        loss      = (w * soft_mask * (D_theta - z).pow(2)).mean()
        loss      = torch.nan_to_num(loss, nan=0.0, posinf=0.0)

        self.log("train/loss",       loss,          prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/sigma_mean", sigma.mean(),  on_step=True,  on_epoch=False)
        return loss

    def validation_step(self, batch: dict, _) -> None:
        z       = self._encode_target(batch)
        sigma   = self.schedule.sample_sigma(z.shape[0], z.device)
        z_noisy = z + sigma[:, None, None, None] * torch.randn_like(z)
        ctx       = self._condition(batch)
        lc_tensor = batch.get("land_cover")
        D_theta   = self._denoiser(z_noisy, sigma, ctx, lc_tensor)
        w         = self.schedule.loss_weight(sigma)[:, None, None, None].clamp(max=1e4)
        soft_mask = self._soft_mask(batch, z)
        loss      = (w * soft_mask * (D_theta - z).pow(2)).mean()
        loss      = torch.nan_to_num(loss, nan=0.0, posinf=0.0)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def sample(
        self,
        batch: dict,
        steps: int = 20,
        guidance: float = 3.0,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """DPMPP-2M sampler (Karras 2022). Returns decoded (B, 2, 256, 256)."""
        ctx_c     = self._condition(batch)                 # (B, 271, D_ctx)
        ctx_u     = torch.zeros_like(ctx_c)               # null context for CFG
        lc_tensor = batch.get("land_cover")               # static — same for both CFG branches

        B   = batch["X"].shape[0]
        dev = batch["X"].device
        z   = torch.randn(B, self.hparams.latent_dim, 32, 32, device=dev) * sigma_max

        # Karras sigma schedule: geometric ramp in ρ-space
        steps_t = torch.arange(steps + 1, device=dev, dtype=torch.float32)
        sigmas  = (
            sigma_max ** (1 / rho)
            + steps_t / steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        sigmas[-1] = 0.0

        old_denoised: torch.Tensor | None = None
        for i in range(steps):
            sig     = sigmas[i].expand(B)
            D_c     = self._denoiser(z, sig, ctx_c, lc_tensor)
            D_u     = self._denoiser(z, sig, ctx_u, lc_tensor)
            D_theta = D_u + guidance * (D_c - D_u)   # CFG blend

            t, t_next = sigmas[i], sigmas[i + 1]

            if t_next == 0:
                z = D_theta
            else:
                h = (t_next / t).log()
                if old_denoised is None:
                    # First step: Euler (DPMPP-2M reduces to Euler on step 0)
                    z = (t_next / t) * z - (-h).expm1() * D_theta
                else:
                    # Second-order correction
                    h_last = (t / sigmas[i - 1]).log()
                    r      = h_last / h
                    d      = (1 + 1 / (2 * r)) * D_theta - (1 / (2 * r)) * old_denoised
                    z      = (t_next / t) * z - (-h).expm1() * d

            old_denoised = D_theta

        return self.vae.decode(self._unnorm_latent(z))   # (B, 2, 256, 256)

    def configure_optimizers(self):
        params = (
            list(self.era5_enc.parameters())
            + list(self.mtg_enc.parameters())
            + list(self.lc_emb.parameters())
            + list(self.unet.parameters())
        )
        opt   = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=200_000, eta_min=1e-6)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "EDMDiffusion":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg.get("model", {}))
