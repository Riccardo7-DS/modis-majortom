"""LAIHead: deterministic (non-diffusion) regression head for sparse LAI/FAPAR supervision.

Reads off the same MTG+ERA5 cross-attention context used by the NDVI denoiser
(``EDMDiffusion._condition``) but predicts LAI/FAPAR directly in pixel space
via a small grid of learned spatial queries that cross-attend into the
context — no VAE, no diffusion noise schedule.

This matches the sparse-estimation design decided for the LAI/FAPAR extension:
MCD15A3H composites cover ~1-in-4 days, far too little data to justify a
second full latent-diffusion pipeline (which would need its own VAE trained
to autoencode LAI reconstructions), and there's no need for a stochastic,
multi-modal output for LAI here — it's estimated at the same conditioning
window used for NDVI, only ever supervised on real composite dates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import SpatialTransformer, _UpConv


class LAIHead(nn.Module):
    """Token-to-raster decoder: learned query grid → cross-attention → upsample.

    The hidden trunk (query grid, attention blocks, upsampling stack) is
    shared between LAI and FAPAR — only the final 1x1-equivalent projection
    is variable-specific. With ``out_channels=2`` (the default), channel 0 is
    LAI and channel 1 is FAPAR, both derived from the same MCD15A3H composite
    and sharing the same ``FparLai_QC`` field, which is why a single shared
    head makes sense here rather than two independent heads.

    Because the two variables have different physical ranges, each output
    channel gets its own activation rather than a shared linear readout:
    LAI (channel 0) passes through ``Softplus`` (LAI >= 0, unbounded above),
    FAPAR (channel 1) passes through ``Sigmoid`` (FAPAR in [0, 1]).

    Parameters
    ----------
    context_dim :
        Must match the cross-attention context dim produced by
        ``EDMDiffusion._condition`` (MTG [+ERA5] tokens).
    channels :
        Feature width of the query grid and upsampling stack.
    query_size :
        Spatial resolution of the learned query grid before upsampling.
    n_blocks :
        Number of self-attn + cross-attn ``SpatialTransformer`` blocks applied
        to the query grid.
    out_channels :
        Output channels. 2 (default) = joint LAI+FAPAR with the activation
        split described above. 1 = legacy LAI-only mode, no activation
        (kept for backward compatibility with pre-FAPAR checkpoints).
    target_size :
        Output spatial resolution. Must be reachable from ``query_size`` via
        power-of-2 nearest-neighbour upsampling (matches ``_UpConv``).
    """

    def __init__(
        self,
        context_dim: int = 512,
        channels: int = 128,
        query_size: int = 32,
        n_blocks: int = 2,
        out_channels: int = 2,
        target_size: int = 256,
    ):
        super().__init__()
        self.query_size = query_size
        self.out_channels = out_channels
        self.query = nn.Parameter(torch.randn(1, channels, query_size, query_size) * 0.02)
        self.blocks = nn.ModuleList([
            SpatialTransformer(channels, context_dim) for _ in range(n_blocks)
        ])

        ups: list[nn.Module] = []
        size = query_size
        while size < target_size:
            ups.append(_UpConv(channels))
            size *= 2
        if size != target_size:
            raise ValueError(
                f"target_size={target_size} not reachable from query_size={query_size} "
                "via power-of-2 upsampling"
            )
        self.up = nn.Sequential(*ups)

        self.out_norm = nn.GroupNorm(min(8, channels), channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1, padding_mode="reflect")

    def forward(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        """context: (B, N_ctx, D_ctx). Returns (B, out_channels, target_size, target_size).

        With out_channels=2: channel 0 = LAI (Softplus'd), channel 1 = FAPAR
        (Sigmoid'd). With out_channels=1 (legacy): raw linear output, no
        activation — unchanged from the pre-FAPAR head.
        """
        h = self.query.expand(batch_size, -1, -1, -1)
        for blk in self.blocks:
            h = blk(h, context)
        h = self.up(h)
        h = self.conv_out(F.silu(self.out_norm(h)))
        if self.out_channels == 2:
            lai   = F.softplus(h[:, 0:1])
            fapar = torch.sigmoid(h[:, 1:2])
            return torch.cat([lai, fapar], dim=1)
        return h
