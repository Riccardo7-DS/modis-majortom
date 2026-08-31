"""EDMDiffusion: Lightning module wrapping VAE + conditioners + denoising UNet.

Noise schedule: Karras et al. 2022 EDM (https://arxiv.org/abs/2206.00364).
Sampler:        DPMPP-2M (20 NFE) for inference.
CFG:            Conditioning signals dropped independently with probability cfg_p
                during training; guided at inference with scale w.
"""
from __future__ import annotations

from pathlib import Path

import torch
import pytorch_lightning as pl
import yaml

from .vae    import NdviVAE
from .vae_v2 import NdviVAEv2, NdviVAEv3
from .conditioning import (
    ERA5TokenEncoder, ERA5TemporalEncoder, MTGFCIEncoder, MTGSpatialEncoder,
    LandCoverEmbedder, NdviBackgroundEncoder,
)
from .unet import DenoisingUNet2D
from .lai_head import LAIHead


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
    ``X``            (B, 18, 256, 256)     MTG FCI conditioning (15 raw + 3 composite channels)
    ``era5``         (B, 12, 15, 256, 256) ERA5-Land temporal features
    ``target_ndvi``  (B, 1, 256, 256)      NDVI target
    ``loss_weight``  (B, 1, 256, 256)      soft_score (used as uncertainty channel)
    ``ndvi_monthly`` (B, 1, 128, 128)      MOD13A3 prev-month NDVI — only when ndvi_bg_channels>0
    ``target_lai``   (B, 1, 256, 256)      LAI target — only consumed when add_lai=True
    ``lai_mask``     (B, 1, 256, 256)      0 where no real MCD15A3H composite (or QC-bad
                                            pixel) exists — only consumed when add_lai=True
    ``target_fpar``  (B, 1, 256, 256)      FAPAR target — only consumed when add_lai=True
    ``fpar_mask``    (B, 1, 256, 256)      Same semantics as ``lai_mask``, independently
                                            masked — only consumed when add_lai=True

    Optional LAI/FAPAR head (add_lai=True)
    ---------------------------------------
    A deterministic (non-diffusion) ``LAIHead`` reads the same cross-attention
    context (``ctx``) used by the NDVI denoiser and jointly regresses LAI and
    FAPAR directly in pixel space, as ``(B, 2, 256, 256)``: channel 0 is LAI
    (Softplus'd, >=0), channel 1 is FAPAR (Sigmoid'd, in [0,1]). The two
    variables share the head's hidden trunk — only the final projection is
    variable-specific — since MCD15A3H retrieves both jointly from the same
    inputs. Sparse-estimation design, not forecasting: the head predicts
    ``LAI_t``/``FAPAR_t`` from the same conditioning window used for
    ``NDVI_t``, supervised only on real MCD15A3H composite dates via masked
    losses (``lai_mask``/``fpar_mask``), never densified/interpolated. Set
    add_lai=False (the default) to reproduce the pre-LAI model exactly — no
    new parameters are created and no LAI/FAPAR keys are read from the batch.

    Conditioning architecture
    -------------------------
    Both ERA5 and MTG produce (B, N, D_ctx) token sequences that are concatenated
    and fed into the UNet's cross-attention layers:
      - MTGFCIEncoder:   (B, 18, 256, 256) → (B, 256, D_ctx)  spatial tokens
      - ERA5TokenEncoder:(B, 12,15, 256, 256) → (B,  15, D_ctx)  temporal tokens
      - combined context: (B, 271, D_ctx)
    UNet input channels: latent_dim (+ lc_out_channels if land cover enabled)
                         (+ ndvi_bg_channels if monthly NDVI background enabled).

    Optional MTG spatial branch (mtg_spatial_channels > 0): a second, independent
    MTG path — MTGSpatialEncoder produces a 32×32 feature map that FiLM/AdaGN-
    modulates the UNet's hidden state right after conv_in (see unet.SpatialFiLM),
    rather than entering via cross-attention or channel-concat. Does not change
    UNet in_channels or the cross-attention context size; disabled by default
    (0), so checkpoints trained before this option keep loading unchanged.

    Parameters
    ----------
    vae_ckpt :
        Path to a NdviVAEModule checkpoint produced by ``train_vae.py``.
        VAE weights are loaded from the ``vae.*`` sub-keys and frozen.
    era5_vars, era5_days :
        ERA5Source configuration (must match the dataset).
    era5_channels :
        Kept for config backward-compatibility; no longer controls UNet in_channels.
    mtg_raw_channels, mtg_composite_channels :
        MTG FCI tensor channel split: 5 timestamps x 3 raw bands (15) plus 3
        temporal-reliability composites (NDVI75, NDVI_std, CloudScore).
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
    add_lai :
        Enable the joint LAI+FAPAR head (default False = identical to the
        pre-LAI model; no extra parameters, no LAI/FAPAR batch keys read).
    lai_head_channels, lai_head_blocks, lai_query_size :
        ``LAIHead`` architecture knobs. Only used when add_lai=True.
    lai_loss_weight :
        Multiplier on the combined (LAI + fapar_loss_weight * FAPAR) masked
        MSE term before adding it to the total training loss ("lambda_aux").
        Only used when add_lai=True.
    fapar_loss_weight :
        Multiplier on the FAPAR term relative to LAI within the combined
        auxiliary loss ("lambda_fapar"); tune if one task dominates training.
        Only used when add_lai=True.
    lai_norm_scale :
        Fixed physical normalisation divisor applied to LAI (prediction and
        target) before the squared-error term is computed, so its loss
        magnitude is on the same order as FAPAR's (already native [0,1])
        instead of ~15-20x larger — MCD15A3H ``Lai_500m``'s valid range is
        0-10 m^2/m^2 per the product spec, so dividing by 10 puts both
        variables on a comparable ~[0,1] scale. This removes scale as a
        confound from ``fapar_loss_weight``, which then only expresses
        genuine task-priority preference. Only used when add_lai=True.
    """

    def __init__(
        self,
        vae_ckpt: str | None = None,
        vae_version: str = "v1",
        era5_vars: int = 12,
        era5_days: int = 15,
        era5_channels: int = 64,
        mtg_raw_channels: int = 15,
        mtg_composite_channels: int = 3,
        context_dim: int = 512,
        latent_dim: int = 4,
        sigma_data: float = 1.0,
        latent_channel_scale: list[float] | None = None,
        cfg_p: float = 0.1,
        lr: float = 1e-4,
        lc_n_classes: int = 18,
        lc_embed_dim: int = 32,
        lc_out_channels: int = 32,
        era5_encoder: str = "spatial",
        vae_channel_widths: list[int] | None = None,
        mtg_spatial_channels: int = 0,
        ndvi_bg_channels: int = 0,
        add_lai: bool = False,
        lai_head_channels: int = 128,
        lai_head_blocks: int = 2,
        lai_query_size: int = 32,
        lai_loss_weight: float = 1.0,
        fapar_loss_weight: float = 1.0,
        lai_norm_scale: float = 10.0,
        use_soft_mask: bool = True,
        vae_use_output_norm: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vae_ckpt"])
        self.cfg_p = cfg_p
        self.lr    = lr

        # ── VAE (frozen) ─────────────────────────────────────────────────
        # _latent_size: derive from channel_widths when provided (n_downs = len-1),
        # otherwise fall back to the vae_version heuristic for old configs.
        if vae_channel_widths:
            self._latent_size = 256 // (2 ** (len(vae_channel_widths) - 1))
        else:
            self._latent_size = 64 if vae_version == "v3" else 32
        if vae_version == "v2":
            vae_kwargs = {"latent_dim": latent_dim}
            if vae_channel_widths:
                vae_kwargs["channel_widths"] = tuple(vae_channel_widths)
            self.vae = NdviVAEv2(**vae_kwargs)
        elif vae_version == "v3":
            vae_kwargs = {"latent_dim": latent_dim, "use_output_norm": vae_use_output_norm}
            if vae_channel_widths:
                vae_kwargs["channel_widths"] = tuple(vae_channel_widths)
            self.vae = NdviVAEv3(**vae_kwargs)
        else:
            self.vae = NdviVAE(latent_dim=latent_dim)
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
        self._era5_encoder = era5_encoder
        if era5_encoder == "token":
            # (B,V,T,H,W) → (B, n_days, context_dim) — cross-attention tokens
            self.era5_enc = ERA5TokenEncoder(era5_vars, era5_days, context_dim)
            era5_in_ch = 0
        else:
            # legacy: (B,V,T,H,W) → (B, era5_channels, 32, 32) — channel concat
            self.era5_enc = ERA5TemporalEncoder(era5_vars, era5_days, era5_channels)
            era5_in_ch = era5_channels

        self.mtg_enc  = MTGFCIEncoder(mtg_raw_channels, mtg_composite_channels, context_dim)
        self.mtg_spatial_enc = (
            MTGSpatialEncoder(mtg_raw_channels, mtg_composite_channels, mtg_spatial_channels)
            if mtg_spatial_channels > 0 else None
        )
        self.lc_emb      = LandCoverEmbedder(n_classes=lc_n_classes,
                                              embed_dim=lc_embed_dim,
                                              out_channels=lc_out_channels)
        self.ndvi_bg_enc = NdviBackgroundEncoder(out_channels=ndvi_bg_channels)

        in_ch = latent_dim + era5_in_ch + self.lc_emb.out_channels + self.ndvi_bg_enc.out_channels
        self.unet     = DenoisingUNet2D(in_ch, latent_dim, context_dim=context_dim,
                                         mtg_channels=mtg_spatial_channels)
        self.schedule = EDMSchedule(sigma_data)

        # ── optional joint LAI+FAPAR head (sparse estimation, see project memory) ──
        self.add_lai = add_lai
        self.lai_loss_weight = lai_loss_weight
        self.fapar_loss_weight = fapar_loss_weight
        self.lai_norm_scale = lai_norm_scale
        self.use_soft_mask = use_soft_mask
        self.lai_head = (
            LAIHead(context_dim=context_dim, channels=lai_head_channels,
                    query_size=lai_query_size, n_blocks=lai_head_blocks,
                    out_channels=2)
            if add_lai else None
        )

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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Encode conditioning signals.

        Returns
        -------
        era5_feat   : (B, era5_channels, 32, 32) if era5_encoder=="spatial", else None
        mtg_spatial : (B, mtg_spatial_channels, 32, 32) if mtg_spatial_channels>0, else None
        ctx         : (B, N_ctx, D_ctx) — MTG tokens [+ ERA5 tokens if era5_encoder=="token"]
        """
        era5 = torch.nan_to_num(batch["era5"], nan=0.0, posinf=0.0, neginf=0.0)
        X    = torch.nan_to_num(batch["X"],    nan=0.0, posinf=0.0, neginf=0.0)
        mtg_tokens = self.mtg_enc(X)  # (B, 256, D_ctx)

        mtg_spatial = self.mtg_spatial_enc(X) if self.mtg_spatial_enc is not None else None
        if drop_mtg and mtg_spatial is not None:
            mtg_spatial = torch.zeros_like(mtg_spatial)

        if self._era5_encoder == "token":
            era5_tokens = self.era5_enc(era5)   # (B, 15, D_ctx)
            if drop_era5:
                era5_tokens = torch.zeros_like(era5_tokens)
            if drop_mtg:
                mtg_tokens = torch.zeros_like(mtg_tokens)
            return None, mtg_spatial, torch.cat([mtg_tokens, era5_tokens], dim=1)
        else:
            era5_feat = self.era5_enc(era5)     # (B, era5_channels, 32, 32)
            if drop_era5:
                era5_feat = torch.zeros_like(era5_feat)
            if drop_mtg:
                mtg_tokens = torch.zeros_like(mtg_tokens)
            return era5_feat, mtg_spatial, mtg_tokens

    def _denoiser(
        self,
        z_noisy: torch.Tensor,                     # (B, latent_dim, 32, 32)
        sigma: torch.Tensor,                       # (B,)
        ctx: torch.Tensor,                         # (B, N_ctx, D_ctx)
        lc_tensor: torch.Tensor | None = None,     # (B, 1, 256, 256)
        era5_feat: torch.Tensor | None = None,     # (B, era5_ch, 32, 32) legacy only
        mtg_spatial: torch.Tensor | None = None,   # (B, mtg_spatial_channels, 32, 32)
        ndvi_bg: torch.Tensor | None = None,       # (B, 1, 128, 128)
    ) -> torch.Tensor:
        sig4d = sigma.view(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.schedule.preconditioners(sig4d)

        inp = c_in * z_noisy
        if era5_feat is not None:
            inp = torch.cat([inp, era5_feat], dim=1)
        lc = self.lc_emb(lc_tensor)
        if lc is not None:
            inp = torch.cat([inp, lc], dim=1)
        bg = self.ndvi_bg_enc(ndvi_bg)
        if bg is not None:
            inp = torch.cat([inp, bg], dim=1)

        F_theta = self.unet(inp, c_noise.view(-1), ctx, mtg_feat=mtg_spatial)
        return c_skip * z_noisy + c_out * F_theta

    # ── Lightning API ─────────────────────────────────────────────────────────

    def _soft_mask(self, batch: dict, latent: torch.Tensor) -> torch.Tensor:
        """Downsample soft_score to latent space with alpha floor removed.

        batch["loss_weight"] = soft + alpha*(1-soft) ∈ [0.2, 1.0] (alpha=0.2).
        Remapping to [0, 1] gives zero weight to pixels outside the MODIS footprint.
        When use_soft_mask=False, returns uniform ones (unweighted EDM loss).
        """
        if not self.use_soft_mask:
            return torch.ones(latent.shape[0], 1, latent.shape[2], latent.shape[3],
                              device=latent.device, dtype=latent.dtype)
        soft = batch["loss_weight"].to(latent.device)                          # (B, 1, 256, 256)
        pool_k = 256 // self._latent_size
        soft_lat = torch.nn.functional.avg_pool2d(soft, kernel_size=pool_k, stride=pool_k)
        return (soft_lat - 0.2).clamp(min=0) / 0.8                            # remap [0.2,1]→[0,1]

    def _gap_mask(self, batch: dict, latent: torch.Tensor) -> torch.Tensor:
        """Downsample the ±7d Whittaker cloud mask to latent space.

        Returns a (B, 1, H_lat, W_lat) float tensor: 1 = reliable pixel
        (clear observation exists within ±7 days), 0 = both gaps exceed 7 days.
        Falls back to all-ones (no masking) when the key is absent, preserving
        backward compatibility with datasets built without use_gap_mask=True.
        """
        if "whittaker_cloud_mask" not in batch:
            return torch.ones(latent.shape[0], 1, latent.shape[2], latent.shape[3],
                              device=latent.device, dtype=latent.dtype)
        mask   = batch["whittaker_cloud_mask"].to(latent.device)   # (B, 1, 256, 256)
        pool_k = mask.shape[-1] // latent.shape[-1]
        if pool_k > 1:
            mask = torch.nn.functional.avg_pool2d(mask, kernel_size=pool_k, stride=pool_k)
        return mask

    def _lai_loss(self, batch: dict, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked MSE for the joint LAI+FAPAR head, computed independently per variable.

        ``lai_mask``/``fpar_mask`` are 0 for pixels with no real MCD15A3H
        observation — either the whole sample falls on a non-composite date,
        or the pixel failed QC within a valid composite (see ``LAISource``).
        LAI and FAPAR are masked independently (their NaN patterns can differ
        slightly even though both derive from the same ``FparLai_QC`` field),
        and each is normalised by its own ``mask.sum()`` rather than the full
        batch size, keeping loss magnitude comparable across batches with
        different numbers of composite-date samples (~1-in-4 on average).

        LAI (prediction and target) is additionally divided by
        ``lai_norm_scale`` before squaring, so ``loss_lai`` and ``loss_fapar``
        land on comparable magnitudes by construction — raw LAI has ~15-20x
        the variance of FAPAR (physical units 0-10 vs a native [0,1]
        fraction), so without this the combined aux loss is dominated by LAI
        regardless of ``fapar_loss_weight``.

        Reuses ``ctx`` from the caller's ``_condition()`` call, including
        whatever CFG dropout was applied for the NDVI branch — a mild, cheap
        regulariser for the LAI/FAPAR head too, since it never uses CFG at
        inference (see ``predict_lai``).

        Returns (loss_lai, loss_fapar) — callers combine as
        ``loss_lai + fapar_loss_weight * loss_fapar``.
        """
        required = ("target_lai", "lai_mask", "target_fpar", "fpar_mask")
        missing = [k for k in required if k not in batch]
        if missing:
            raise KeyError(
                f"add_lai=True but batch is missing {missing} — the joint "
                "LAI+FAPAR head needs LAISource(..., fpar_band='Fpar_500m') "
                "passed to AlignedPatchDataset."
            )
        pred = self.lai_head(ctx, batch["target_lai"].shape[0])  # (B, 2, H, W)

        target_lai = batch["target_lai"].to(ctx.device) / self.lai_norm_scale
        pred_lai   = pred[:, 0:1] / self.lai_norm_scale
        mask_lai   = batch["lai_mask"].to(ctx.device)
        se_lai     = mask_lai * (pred_lai - target_lai).pow(2)
        loss_lai   = se_lai.sum() / mask_lai.sum().clamp(min=1.0)

        target_fapar = batch["target_fpar"].to(ctx.device)
        mask_fapar   = batch["fpar_mask"].to(ctx.device)
        se_fapar     = mask_fapar * (pred[:, 1:2] - target_fapar).pow(2)
        loss_fapar   = se_fapar.sum() / mask_fapar.sum().clamp(min=1.0)

        return loss_lai, loss_fapar

    def training_step(self, batch: dict, _) -> torch.Tensor:
        z     = self._encode_target(batch)
        sigma = self.schedule.sample_sigma(z.shape[0], z.device)
        noise = torch.randn_like(z)
        z_noisy = z + sigma[:, None, None, None] * noise

        # Per-batch CFG dropout (independent for each conditioning branch)
        drop_mtg  = bool(torch.rand(()) < self.cfg_p)
        drop_era5 = bool(torch.rand(()) < self.cfg_p)
        era5_feat, mtg_spatial, ctx = self._condition(batch, drop_mtg, drop_era5)
        lc_tensor = batch.get("land_cover")
        ndvi_bg   = batch.get("ndvi_monthly")

        D_theta   = self._denoiser(z_noisy, sigma, ctx, lc_tensor, era5_feat, mtg_spatial, ndvi_bg)
        w         = self.schedule.loss_weight(sigma)[:, None, None, None].clamp(max=1e4)
        soft_mask = self._soft_mask(batch, z)
        gap_mask  = self._gap_mask(batch, z)
        ndvi_loss = (w * soft_mask * gap_mask * (D_theta - z).pow(2)).mean()
        ndvi_loss = torch.nan_to_num(ndvi_loss, nan=0.0, posinf=0.0)
        loss      = ndvi_loss

        self.log("train/ndvi_loss",  ndvi_loss,     on_step=True,  on_epoch=True)
        self.log("train/sigma_mean", sigma.mean(),  on_step=True,  on_epoch=False)

        if self.lai_head is not None:
            loss_lai, loss_fapar = self._lai_loss(batch, ctx)
            aux_loss = loss_lai + self.fapar_loss_weight * loss_fapar
            loss     = loss + self.lai_loss_weight * aux_loss
            self.log("train/lai_loss",   loss_lai,   on_step=True, on_epoch=True)
            self.log("train/fapar_loss", loss_fapar, on_step=True, on_epoch=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, _) -> None:
        z       = self._encode_target(batch)
        sigma   = self.schedule.sample_sigma(z.shape[0], z.device)
        z_noisy = z + sigma[:, None, None, None] * torch.randn_like(z)
        era5_feat, mtg_spatial, ctx = self._condition(batch)
        lc_tensor = batch.get("land_cover")
        ndvi_bg   = batch.get("ndvi_monthly")
        D_theta   = self._denoiser(z_noisy, sigma, ctx, lc_tensor, era5_feat, mtg_spatial, ndvi_bg)
        w         = self.schedule.loss_weight(sigma)[:, None, None, None].clamp(max=1e4)
        soft_mask = self._soft_mask(batch, z)
        gap_mask  = self._gap_mask(batch, z)
        ndvi_loss = (w * soft_mask * gap_mask * (D_theta - z).pow(2)).mean()
        ndvi_loss = torch.nan_to_num(ndvi_loss, nan=0.0, posinf=0.0)
        loss      = ndvi_loss

        if self.lai_head is not None:
            loss_lai, loss_fapar = self._lai_loss(batch, ctx)
            aux_loss = loss_lai + self.fapar_loss_weight * loss_fapar
            loss     = loss + self.lai_loss_weight * aux_loss
            self.log("val/lai_loss",   loss_lai,   on_epoch=True, sync_dist=True)
            self.log("val/fapar_loss", loss_fapar, on_epoch=True, sync_dist=True)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

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
        era5_feat_c, mtg_spatial_c, ctx_c = self._condition(batch)
        era5_feat_u = torch.zeros_like(era5_feat_c) if era5_feat_c is not None else None
        mtg_spatial_u = torch.zeros_like(mtg_spatial_c) if mtg_spatial_c is not None else None
        ctx_u       = torch.zeros_like(ctx_c)
        lc_tensor   = batch.get("land_cover")
        ndvi_bg     = batch.get("ndvi_monthly")

        B   = batch["X"].shape[0]
        dev = batch["X"].device
        z   = torch.randn(B, self.hparams.latent_dim, self._latent_size, self._latent_size, device=dev) * sigma_max

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
            D_c     = self._denoiser(z, sig, ctx_c, lc_tensor, era5_feat_c, mtg_spatial_c, ndvi_bg)
            D_u     = self._denoiser(z, sig, ctx_u, lc_tensor, era5_feat_u, mtg_spatial_u, ndvi_bg)
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

    @torch.no_grad()
    def predict_lai(self, batch: dict) -> torch.Tensor | None:
        """Deterministic joint LAI+FAPAR prediction — no diffusion sampling loop needed.

        Returns (B, 2, 256, 256): channel 0 = LAI (Softplus'd, >=0),
        channel 1 = FAPAR (Sigmoid'd, in [0,1]).

        Unlike ``sample()``, this needs no NDVI-conditioning-only batch dict
        beyond ``X``/``era5`` (whatever ``InferencePatchDataset`` already
        provides) — the LAI head reads the same conditioning tokens the NDVI
        denoiser cross-attends into. Uses the full (undropped) context, unlike
        training where CFG dropout regularises the shared trunk.

        Returns None if add_lai=False.
        """
        if self.lai_head is None:
            return None
        _, _, ctx = self._condition(batch)
        B = batch["X"].shape[0]
        return self.lai_head(ctx, B)

    def configure_optimizers(self):
        params = (
            list(self.era5_enc.parameters())
            + list(self.mtg_enc.parameters())
            + list(self.lc_emb.parameters())
            + list(self.unet.parameters())
        )
        if self.mtg_spatial_enc is not None:
            params += list(self.mtg_spatial_enc.parameters())
        if self.lai_head is not None:
            params += list(self.lai_head.parameters())
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
