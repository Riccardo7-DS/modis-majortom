"""DenoisingUNet2D: compact 32×32 latent-space denoiser with cross-attention.

Architecture (default channel_mult=(256,384,512,512)):

  Encoder:
    32×32  C=256   ResBlock×2
    16×16  C=384   ResBlock×2 + SpatialTransformer (cross-attn with MTG tokens)
     8×8   C=512   ResBlock×2 + SpatialTransformer
     4×4   C=512   ResBlock×2

  Bottleneck 4×4:
    ResBlock + SpatialTransformer + ResBlock

  Decoder (mirror, with skip connections):
     4×4 → 8×8 → 16×16 → 32×32

Timestep conditioning via AdaGroupNorm (scale + shift from sinusoidal embedding).
Cross-attention context comes from MTGFCIEncoder (B, 256 tokens, D_ctx).

Optional MTG spatial branch (mtg_channels > 0): a 32×32 feature map from
conditioning.MTGSpatialEncoder FiLM/AdaGN-modulates h once, right after
conv_in, via SpatialFiLM (spatially-varying GroupNorm scale+shift, zero-init).
Independent of the MTGFCIEncoder token branch above — no shared weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── time embedding ────────────────────────────────


def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args  = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)  # (B, dim)


class TimestepEmbedder(nn.Module):
    """sinusoidal → 2-layer MLP → emb_dim embedding used by every ResBlock."""

    def __init__(self, model_channels: int = 256, emb_dim: int = 1024):
        super().__init__()
        self.model_channels = model_channels
        self.net = nn.Sequential(
            nn.Linear(model_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(_sinusoidal_embedding(t, self.model_channels))


# ──────────────────────────── building blocks ────────────────────────────────


class AdaGroupNorm(nn.Module):
    """GroupNorm with scale+shift predicted from the timestep embedding."""

    def __init__(self, c: int, emb_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, c, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * c)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, emb_dim) → scale/shift: (B, c)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class SpatialFiLM(nn.Module):
    """GroupNorm modulated by per-pixel scale+shift predicted from a spatial
    conditioning map. Like AdaGroupNorm but spatially-varying (SPADE-style)
    instead of a single vector broadcast uniformly over the feature map —
    used for the optional MTG spatial branch, which carries per-location
    information (see conditioning.MTGSpatialEncoder).

    proj is zero-initialised so the conditioning content contributes nothing
    at construction time (ControlNet-style "zero conv"): scale=shift=0, so
    training starts as if the branch weren't there and its influence grows
    in gradually. Note this still applies GroupNorm to x unconditionally, so
    it is not a strict identity on x the way a residual add would be.
    """

    def __init__(self, c: int, cond_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, c, affine=False)
        self.proj = nn.Conv2d(cond_channels, 2 * c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=1)  # (B, c, H, W) each
        # tanh bounds scale to (-1, 1) → multiplicative factor in (0, 2), preventing
        # unbounded growth of proj weights from destabilising the hidden state.
        return self.norm(x) * (1 + torch.tanh(scale)) + shift


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, emb_dim: int):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_c, emb_dim)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode="reflect")
        self.norm2 = AdaGroupNorm(out_c, emb_dim)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, padding_mode="reflect")
        self.skip  = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, emb)))
        h = self.conv2(F.silu(self.norm2(h, emb)))
        return h + self.skip(x)


class _DownConv(nn.Module):
    """Stride-2 downsampling with reflect padding (padding_mode unsupported for stride>1)."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (1, 1, 1, 1), mode="reflect"))


class _UpConv(nn.Module):
    """Nearest-neighbour upsample + reflect-padded 3×3 conv."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class CrossAttention(nn.Module):
    """Multi-head attention: queries from spatial features, keys/values from context."""

    def __init__(self, dim: int, context_dim: int, n_heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner        = n_heads * head_dim
        self.scale   = head_dim ** -0.5
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.to_q   = nn.Linear(dim, inner, bias=False)
        self.to_k   = nn.Linear(context_dim, inner, bias=False)
        self.to_v   = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.LayerNorm(dim))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h       = self.n_heads
        q = self.to_q(x).reshape(B, N, h, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, h, self.head_dim).transpose(1, 2)
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out  = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out  = out.transpose(1, 2).reshape(B, N, h * self.head_dim)
        return x + self.to_out(out)


class SpatialTransformer(nn.Module):
    """Self-attention + cross-attention applied to 2-D feature maps.

    Flattens spatial dims → token sequence → attention → reshape back.
    """

    def __init__(self, c: int, context_dim: int = 512,
                 n_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.norm_in   = nn.GroupNorm(8, c, affine=False)
        self.proj_in   = nn.Conv2d(c, c, 1)
        self.self_attn  = CrossAttention(c, c, n_heads, head_dim)
        self.cross_attn = CrossAttention(c, context_dim, n_heads, head_dim)
        self.ff         = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c),
        )
        self.proj_out   = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h      = self.proj_in(self.norm_in(x))
        tokens = h.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        tokens = self.self_attn(tokens, tokens)
        tokens = self.cross_attn(tokens, context)
        tokens = tokens + self.ff(tokens)
        h      = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(h)


# ──────────────────────────── UNet ──────────────────────────────────────────


class DenoisingUNet2D(nn.Module):
    """32×32 latent-space denoiser.

    Parameters
    ----------
    in_channels :
        latent_dim (+ optional lc_channels). Default 4.
        ERA5 conditioning is now injected via cross-attention tokens, not channels.
    latent_dim :
        UNet output channels (= VAE latent dim). Default 4.
    channel_mult :
        Feature channels at each resolution level (32, 16, 8, 4 pixels).
    emb_dim :
        Dimension of the timestep embedding MLP output.
    context_dim :
        Cross-attention context dimension (from MTGFCIEncoder).
    mtg_channels :
        Channel width of the optional MTG spatial conditioning map (from
        conditioning.MTGSpatialEncoder). 0 (default) disables the branch —
        no SpatialFiLM submodule is created, and forward() ignores mtg_feat.
        When > 0, the map FiLM/AdaGN-modulates the hidden state once, right
        after conv_in, at the model's native 32×32 resolution.
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 4,
        channel_mult: tuple = (256, 384, 512, 512),
        emb_dim: int = 1024,
        context_dim: int = 512,
        mtg_channels: int = 0,
    ):
        super().__init__()
        cw = channel_mult

        self.time_emb = TimestepEmbedder(emb_dim=emb_dim)
        self.conv_in  = nn.Conv2d(in_channels, cw[0], 3, padding=1, padding_mode="reflect")
        self.mtg_film = SpatialFiLM(cw[0], mtg_channels) if mtg_channels > 0 else None

        # ── encoder ──────────────────────────────────────────────────────
        # Level 0: 32×32, C=cw[0]
        self.down0 = nn.ModuleList([
            ResBlock(cw[0], cw[0], emb_dim),
            ResBlock(cw[0], cw[0], emb_dim),
        ])
        self.down0_ds = _DownConv(cw[0], cw[0])

        # Level 1: 16×16, C=cw[1], + cross-attention
        self.down1 = nn.ModuleList([
            ResBlock(cw[0], cw[1], emb_dim),
            ResBlock(cw[1], cw[1], emb_dim),
        ])
        self.down1_attn = SpatialTransformer(cw[1], context_dim)
        self.down1_ds   = _DownConv(cw[1], cw[1])

        # Level 2: 8×8, C=cw[2], + cross-attention
        self.down2 = nn.ModuleList([
            ResBlock(cw[1], cw[2], emb_dim),
            ResBlock(cw[2], cw[2], emb_dim),
        ])
        self.down2_attn = SpatialTransformer(cw[2], context_dim)
        self.down2_ds   = _DownConv(cw[2], cw[2])

        # Level 3: 4×4, C=cw[3]
        self.down3 = nn.ModuleList([
            ResBlock(cw[2], cw[3], emb_dim),
            ResBlock(cw[3], cw[3], emb_dim),
        ])

        # ── bottleneck ────────────────────────────────────────────────────
        self.mid0     = ResBlock(cw[3], cw[3], emb_dim)
        self.mid_attn = SpatialTransformer(cw[3], context_dim)
        self.mid1     = ResBlock(cw[3], cw[3], emb_dim)

        # ── decoder ───────────────────────────────────────────────────────
        # Level 3 → 2 (4→8)
        self.up3 = nn.ModuleList([
            ResBlock(cw[3] * 2, cw[3], emb_dim),
            ResBlock(cw[3], cw[2], emb_dim),
        ])
        self.up3_us = _UpConv(cw[2])

        # Level 2 → 1 (8→16) + cross-attention
        self.up2 = nn.ModuleList([
            ResBlock(cw[2] * 2, cw[2], emb_dim),
            ResBlock(cw[2], cw[1], emb_dim),
        ])
        self.up2_attn = SpatialTransformer(cw[1], context_dim)
        self.up2_us   = _UpConv(cw[1])

        # Level 1 → 0 (16→32) + cross-attention
        self.up1 = nn.ModuleList([
            ResBlock(cw[1] * 2, cw[1], emb_dim),
            ResBlock(cw[1], cw[0], emb_dim),
        ])
        self.up1_attn = SpatialTransformer(cw[0], context_dim)
        self.up1_us   = _UpConv(cw[0])

        # Level 0 (32×32)
        self.up0 = nn.ModuleList([
            ResBlock(cw[0] * 2, cw[0], emb_dim),
            ResBlock(cw[0], cw[0], emb_dim),
        ])

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, cw[0]),
            nn.SiLU(),
            nn.Conv2d(cw[0], latent_dim, 3, padding=1, padding_mode="reflect"),
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, in_channels, 32, 32)
        t: torch.Tensor,       # (B,) EDM c_noise = sigma.log()/4
        context: torch.Tensor, # (B, N_ctx, D_ctx) from MTGFCIEncoder
        mtg_feat: torch.Tensor | None = None,  # (B, mtg_channels, 32, 32), optional
    ) -> torch.Tensor:
        emb = self.time_emb(t)   # (B, emb_dim)
        h   = self.conv_in(x)    # (B, cw[0], 32, 32)
        if mtg_feat is not None:
            h = self.mtg_film(h, mtg_feat)

        # Encoder — save skips after attention (not after downsampling)
        for blk in self.down0:
            h = blk(h, emb)
        skip0 = h                                    # (B, cw[0], 32, 32)
        h     = self.down0_ds(h)

        for blk in self.down1:
            h = blk(h, emb)
        h     = self.down1_attn(h, context)
        skip1 = h                                    # (B, cw[1], 16, 16)
        h     = self.down1_ds(h)

        for blk in self.down2:
            h = blk(h, emb)
        h     = self.down2_attn(h, context)
        skip2 = h                                    # (B, cw[2], 8, 8)
        h     = self.down2_ds(h)

        for blk in self.down3:
            h = blk(h, emb)
        skip3 = h                                    # (B, cw[3], 4, 4)

        # Bottleneck
        h = self.mid0(h, emb)
        h = self.mid_attn(h, context)
        h = self.mid1(h, emb)

        # Decoder
        h = torch.cat([h, skip3], dim=1)
        for blk in self.up3:
            h = blk(h, emb)
        h = self.up3_us(h)

        h = torch.cat([h, skip2], dim=1)
        for blk in self.up2:
            h = blk(h, emb)
        h = self.up2_attn(h, context)
        h = self.up2_us(h)

        h = torch.cat([h, skip1], dim=1)
        for blk in self.up1:
            h = blk(h, emb)
        h = self.up1_attn(h, context)
        h = self.up1_us(h)

        h = torch.cat([h, skip0], dim=1)
        for blk in self.up0:
            h = blk(h, emb)

        return self.conv_out(h)  # (B, latent_dim, 32, 32)
