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


def _sobel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between Sobel gradient magnitudes. Input: single-channel float."""
    pred, target = pred.float(), target.float()
    device = pred.device
    Kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=device).view(1, 1, 3, 3)
    Ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=device).view(1, 1, 3, 3)

    def grad_mag(x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, Kx, padding=1)
        gy = F.conv2d(x, Ky, padding=1)
        return (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()

    return F.l1_loss(grad_mag(pred), grad_mag(target))


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


class NdviVAEv3(nn.Module):
    """4× spatial compression: 256×256×2 ↔ 64×64×16 with U-Net skip connections.

    Two improvements over NdviVAEv2:
    - 2 downsampling stages (vs 3) → latent is 64×64, not 32×32.  Features as
      small as ~4 px at 256×256 survive the bottleneck.
    - U-Net skip connections carry encoder feature maps to matching decoder scales,
      bypassing KL regularisation for fine spatial detail (water bodies, field
      edges).  Skip dropout (p=0.5 during training) ensures the latent remains
      self-sufficient so the decoder works at diffusion-inference time when no
      encoder image is available.

    API identical to NdviVAEv2: encode() → (z, mu, logvar), decode(z) → recon,
    forward() → (recon, mu, logvar).  Skips are stored transiently on the instance
    between encode() and decode() calls within the same forward pass.
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 16,
        channel_widths: tuple = (64, 128, 256),
        skip_drop_prob: float = 0.5,
        use_output_norm: bool = True,
    ):
        super().__init__()
        cw = list(channel_widths)
        n_down = len(cw) - 1  # number of downsampling stages (= 2)
        self.skip_drop_prob = skip_drop_prob
        self._enc_skips: list | None = None

        # ── Encoder ───────────────────────────────────────────────────────────
        # Each stage: Downsample+1×1 first, then ResBlock → save skip here.
        # Skips are therefore at (cw[i+1], H/2^(i+1), W/2^(i+1)) — avoids the
        # huge 256×256 tensor that a pre-downsample skip would produce.
        self.enc_in = nn.Conv2d(in_channels, cw[0], 3, padding=1, padding_mode="reflect")
        self.enc_downs  = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        for i in range(n_down):
            self.enc_downs.append(nn.Sequential(
                _Downsample(cw[i]),
                nn.Conv2d(cw[i], cw[i + 1], 1),
            ))
            self.enc_blocks.append(_ResBlock(cw[i + 1], cw[i + 1]))

        self.enc_mid = nn.Sequential(
            _ResBlock(cw[-1], cw[-1]),
            _ResBlock(cw[-1], cw[-1]),
        )
        self.mu_head     = nn.Conv2d(cw[-1], latent_dim, 1)
        self.logvar_head = nn.Conv2d(cw[-1], latent_dim, 1)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_in = nn.Sequential(
            nn.Conv2d(latent_dim, cw[-1], 3, padding=1, padding_mode="reflect"),
            _ResBlock(cw[-1], cw[-1]),
        )
        # Stages in reverse order: merge skip → ResBlock → Upsample+1×1
        # dec_merges[0] handles the deepest skip (same spatial as z),
        # dec_merges[1] handles the shallowest skip (2× spatial).
        self.dec_merges = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_ups    = nn.ModuleList()
        for i in range(n_down - 1, -1, -1):
            self.dec_merges.append(nn.Conv2d(cw[i + 1] * 2, cw[i + 1], 1))
            self.dec_blocks.append(_ResBlock(cw[i + 1], cw[i + 1]))
            self.dec_ups.append(nn.Sequential(
                _Upsample(cw[i + 1]),
                nn.Conv2d(cw[i + 1], cw[i], 1),
            ))

        # GroupNorm here zeros the feature-map mean, discarding amplitude info.
        # use_output_norm=False removes it so the decoder can preserve NDVI scale.
        out_layers: list[nn.Module] = []
        if use_output_norm:
            out_layers.append(nn.GroupNorm(min(8, cw[0]), cw[0]))
        out_layers += [nn.SiLU(), nn.Conv2d(cw[0], in_channels, 3, padding=1, padding_mode="reflect")]
        self.dec_out = nn.Sequential(*out_layers)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_zero_skips(self, z: torch.Tensor) -> list:
        """Zero-filled skip tensors for inference (no encoder image available).

        Shapes are derived from the merge-conv in_channels so the formula is
        correct regardless of channel_widths or n_down.
        """
        B, _, H, W = z.shape
        n = len(self.dec_merges)
        # enc skips[j] (j=0..n-1) is processed by dec_merges[n-1-j] via reversed().
        # skip_ch  = dec_merges[n-1-j].in_channels // 2
        # spatial  = (H * 2^(n-1-j), W * 2^(n-1-j))
        return [
            torch.zeros(
                B,
                self.dec_merges[n - 1 - j].in_channels // 2,
                H * (2 ** (n - 1 - j)),
                W * (2 ** (n - 1 - j)),
                device=z.device,
                dtype=z.dtype,
            )
            for j in range(n)
        ]

    # ── forward ───────────────────────────────────────────────────────────────

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc_in(x)
        skips: list[torch.Tensor] = []
        for down, block in zip(self.enc_downs, self.enc_blocks):
            h = down(h)
            h = block(h)
            skips.append(h)

        h = self.enc_mid(h)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-30, 20)
        eps    = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        z      = mu + eps * (0.5 * logvar).exp()

        # Skip dropout: zero out all skips with probability skip_drop_prob so
        # the decoder and latent learn to work without them (inference-safe).
        if self.training and torch.rand(1).item() < self.skip_drop_prob:
            self._enc_skips = [torch.zeros_like(s) for s in skips]
        else:
            self._enc_skips = skips

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)

        if self._enc_skips is not None:
            skips = self._enc_skips
            self._enc_skips = None
        else:
            skips = self._make_zero_skips(z)

        for merge, block, up, skip in zip(
            self.dec_merges, self.dec_blocks, self.dec_ups, reversed(skips)
        ):
            h = merge(torch.cat([h, skip], dim=1))
            h = block(h)
            h = up(h)

        return self.dec_out(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


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
        lambda_sobel: float = 0.0,
        lambda_amp: float = 0.0,
        amp_weight_slope: float = 2.0,
        ndvi_mean: float = 0.3,
        ndvi_alpha: float = 0.0,
    ):
        super().__init__()
        self.vae               = vae or NdviVAEv2()
        self.lr                = lr
        self.beta_final        = beta_final
        self.beta_anneal_steps = beta_anneal_steps
        self.lambda_ssim       = lambda_ssim
        self.lambda_l1         = lambda_l1
        self.lambda_sobel      = lambda_sobel
        self.lambda_amp        = lambda_amp
        self.amp_weight_slope  = amp_weight_slope
        self.ndvi_mean         = ndvi_mean
        self.ndvi_alpha        = ndvi_alpha
        self._val_gt_means:   list[torch.Tensor] = []
        self._val_pred_means: list[torch.Tensor] = []

    def _beta(self) -> float:
        t = min(self.global_step / max(self.beta_anneal_steps, 1), 1.0)
        return float(t * self.beta_final)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        target = torch.cat([batch["target_ndvi"], batch["loss_weight"]], dim=1)
        recon, mu, logvar = self.vae(target)

        ndvi_pred   = recon[:, :1]
        ndvi_target = target[:, :1]

        # pixel-wise weight: higher at NDVI extremes, minimum at dataset mean
        if self.ndvi_alpha > 0.0:
            w = (1.0 + self.ndvi_alpha * (ndvi_target - self.ndvi_mean).abs()).float()
            wmse = (w * (ndvi_pred - ndvi_target).float().pow(2)).mean()
            wl1  = (w * (ndvi_pred - ndvi_target).float().abs()).mean()
        else:
            wmse = F.mse_loss(ndvi_pred, ndvi_target)
            wl1  = F.l1_loss(ndvi_pred, ndvi_target)

        wmse_total = wmse

        ssim  = _ssim_loss(ndvi_pred, ndvi_target)
        sobel = _sobel_loss(ndvi_pred, ndvi_target) if self.lambda_sobel > 0.0 else ndvi_pred.new_tensor(0.0)

        # patch-level amplitude loss: penalises mean(recon) != mean(GT) per sample
        # Weighted by GT NDVI level: amp_weight=1 at low NDVI, up to 3 at very_high (>0.5).
        # Directly targets the high-NDVI amplitude collapse without changing low-NDVI gradients.
        amp_gt   = ndvi_target.mean(dim=[-2, -1])   # (B,) — GT patch mean NDVI
        amp_pred = ndvi_pred.mean(dim=[-2, -1])     # (B,) — predicted patch mean NDVI
        if self.lambda_amp > 0.0:
            amp_weight = (1.0 + self.amp_weight_slope * torch.relu(amp_gt - 0.5))
            amp = (amp_weight * (amp_pred - amp_gt).pow(2)).mean()
        else:
            amp = ndvi_pred.new_tensor(0.0)

        if stage == "val":
            self._val_gt_means.append(amp_gt.detach())
            self._val_pred_means.append(amp_pred.detach())

        loss_recon = (wmse_total
                      + self.lambda_ssim  * ssim
                      + self.lambda_l1    * wl1
                      + self.lambda_sobel * sobel
                      + self.lambda_amp   * amp)
        kl   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = loss_recon + self._beta() * kl

        # plain MSE on NDVI channel for PSNR tracking (comparable across runs)
        mse_plain = F.mse_loss(ndvi_pred.detach(), ndvi_target)
        psnr = -10.0 * torch.log10(mse_plain.clamp(min=1e-10))

        sync = (stage == "val")
        self.log(f"{stage}/recon",  loss_recon, prog_bar=True,             sync_dist=sync)
        self.log(f"{stage}/mse",    mse_plain,                             sync_dist=sync)
        self.log(f"{stage}/ssim",   ssim,                                  sync_dist=sync)
        self.log(f"{stage}/sobel",  sobel,                                 sync_dist=sync)
        self.log(f"{stage}/amp",    amp,                                   sync_dist=sync)
        self.log(f"{stage}/kl",     kl,                                    sync_dist=sync)
        self.log(f"{stage}/loss",   loss,        prog_bar=True,             sync_dist=sync)
        self.log(f"{stage}/psnr",   psnr,        prog_bar=(stage == "val"), sync_dist=sync)
        if stage == "train":
            self.log("train/beta", self._beta())
        return loss

    def training_step(self, batch: dict, _) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _) -> None:
        self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        if not self._val_gt_means:
            return

        gt   = self.all_gather(torch.cat(self._val_gt_means)).flatten()
        pred = self.all_gather(torch.cat(self._val_pred_means)).flatten()
        self._val_gt_means.clear()
        self._val_pred_means.clear()

        _BINS = [(-0.2, 0.2, "low"), (0.2, 0.5, "mid"), (0.5, 0.7, "high"), (0.7, 1.1, "very_high")]

        # per-bin bias scalars — logged on all ranks (same values after all_gather)
        for lo, hi, label in _BINS:
            mask = (gt >= lo) & (gt < hi)
            if mask.sum() > 0:
                self.log(f"val/bias_{label}", (pred[mask] - gt[mask]).mean(), sync_dist=False)

        if not self.trainer.is_global_zero:
            return

        # histograms: amp_gt vs amp_pred distributions (TensorBoard only)
        if self.logger and hasattr(self.logger, "experiment"):
            exp = self.logger.experiment
            if hasattr(exp, "add_histogram"):
                step = self.global_step
                exp.add_histogram("val/amp_gt",   gt.cpu(),   step)
                exp.add_histogram("val/amp_pred", pred.cpu(), step)

        header = f"{'bin':<10} {'n':>5} {'GT mean':>9} {'pred mean':>10} {'bias':>8}"
        rows = [header, "─" * len(header)]
        for lo, hi, label in _BINS:
            mask = (gt >= lo) & (gt < hi)
            if mask.sum() == 0:
                continue
            gt_m   = gt[mask].mean().item()
            pred_m = pred[mask].mean().item()
            rows.append(f"{label:<10} {int(mask.sum()):>5} {gt_m:>9.4f} {pred_m:>10.4f} {pred_m - gt_m:>+8.4f}")
        print(f"\n[amp bias @ step {self.global_step}]\n" + "\n".join(rows) + "\n")

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.vae.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100_000, eta_min=1e-5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
