"""Conditioning modules for the NDVI diffusion model.

ERA5TokenEncoder     – DenseNet-style Conv3d + global pool, (B,12,15,256,256) → (B,15,512) tokens
ERA5TemporalEncoder  – DenseNet-style Conv3d + spatial down,(B,12,15,256,256) → (B,64,32,32) map
MTGFCIEncoder        – dual-branch CNN encoder (raw + composite, fused),   (B,18,256,256) → (B,256,512) tokens
MTGSpatialEncoder     – small dual-branch CNN, independent of MTGFCIEncoder, (B,18,256,256) → (B,out_ch,32,32) map
LandCoverEmbedder    – MCD12Q1 IGBP class embedder,        (B,1,256,256)     → (B,out_ch,32,32)

ERA5TokenEncoder is the current default: it produces one token per day so the UNet can
attend selectively to individual days.  ERA5TemporalEncoder is kept for loading older
checkpoints (trained before the v7 refactor) that channel-concatenated ERA5 features.
Select via EDMDiffusion(era5_encoder="token"|"spatial").

MTGSpatialEncoder is an optional second MTG path (EDMDiffusion(mtg_spatial_channels=N)):
its 32×32 feature map FiLM/AdaGN-modulates the UNet's hidden state right after conv_in
(see unet.SpatialFiLM), independent of MTGFCIEncoder's 256-token cross-attention branch.

NdviBackgroundEncoder encodes the MOD13A3 monthly NDVI composite (prev-month, always
available at inference time) as a 32×32 feature map for channel-concat into UNet conv_in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        g = min(8, c)
        self.net = nn.Sequential(
            nn.GroupNorm(g, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(g, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# ERA5TemporalEncoder  (legacy — kept for loading pre-v7 checkpoints)
# ---------------------------------------------------------------------------

class ERA5TemporalEncoder(nn.Module):
    """DenseNet-style Conv3d that collapses time into a spatial feature map.

    Output: (B, out_channels, 32, 32) — channel-concatenated with noisy latent.
    Kept for backward-compatible loading of checkpoints trained before v7.
    New runs should use ERA5TokenEncoder instead.
    """

    def __init__(self, n_vars: int = 12, n_days: int = 15, out_channels: int = 64):
        super().__init__()
        self.n_days       = n_days
        self.out_channels = out_channels

        c0 = n_vars
        c1, c2, c3 = 16, 32, 64

        self.conv_in = nn.Conv3d(c0, c1, (1, 1, 1))
        self.conv1   = nn.Conv3d(c0 + c1, c2, (1, 1, 1))
        self.conv2   = nn.Conv3d(c0 + c1 + c2, c3, (1, 1, 1))
        self.conv_t  = nn.Conv3d(c0 + c1 + c2 + c3, out_channels, (n_days, 1, 1))
        self.spatial_down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 4, stride=4),
            _ResBlock2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 2, stride=2),
        )
        self.out_norm = nn.GroupNorm(8, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = x
        h1 = F.silu(self.conv_in(h0))
        h2 = F.silu(self.conv1(torch.cat([h0, h1], dim=1)))
        h3 = F.silu(self.conv2(torch.cat([h0, h1, h2], dim=1)))
        ht = F.silu(self.conv_t(torch.cat([h0, h1, h2, h3], dim=1)))
        return F.silu(self.out_norm(self.spatial_down(ht.squeeze(2))))


# ---------------------------------------------------------------------------
# ERA5TokenEncoder
# ---------------------------------------------------------------------------

class ERA5TokenEncoder(nn.Module):
    """ERA5-Land temporal encoder that preserves the day dimension as tokens.

    Produces one token per day so the UNet can attend selectively to individual
    days in the 15-day climate window, rather than collapsing time into a single
    spatial feature map.

    Architecture
    ~~~~~~~~~~~~
    Step 1 – pointwise Conv3d DenseNet (kernel 1×1×1) aggregates the V variable
    channels at each (t, h, w) independently:
        input   (B, V, T, H, W)  V=12 vars, T=15 days
        conv_in  V      → 16  channels; concat with input → 28
        conv1    28     → 32  channels; concat all       → 60
        conv2    60     → 64  channels; concat all       → 124

    Step 2 – spatial global-average-pool over (H, W) → (B, 124, T)
    Step 3 – transpose → (B, T, 124); linear projection → (B, T, D_ctx); LayerNorm
    Output: (B, n_days=15, D_ctx=512) token sequence for cross-attention.
    """

    def __init__(self, n_vars: int = 12, n_days: int = 15, d_ctx: int = 512):
        super().__init__()
        self.n_days = n_days

        c0 = n_vars   # 12
        c1 = 16
        c2 = 32
        c3 = 64
        c_dense = c0 + c1 + c2 + c3  # 124 — output of the DenseNet stack

        self.conv_in = nn.Conv3d(c0, c1, (1, 1, 1))
        self.conv1   = nn.Conv3d(c0 + c1, c2, (1, 1, 1))
        self.conv2   = nn.Conv3d(c0 + c1 + c2, c3, (1, 1, 1))

        self.proj = nn.Linear(c_dense, d_ctx)
        self.norm = nn.LayerNorm(d_ctx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, T, H, W)
        h0 = x
        h1 = F.silu(self.conv_in(h0))
        h2 = F.silu(self.conv1(torch.cat([h0, h1], dim=1)))
        h3 = F.silu(self.conv2(torch.cat([h0, h1, h2], dim=1)))
        h  = torch.cat([h0, h1, h2, h3], dim=1)  # (B, c_dense, T, H, W)

        # global spatial average → (B, c_dense, T)
        h = h.mean(dim=[-2, -1])
        # → (B, T, c_dense) → (B, T, D_ctx)
        return self.norm(self.proj(h.permute(0, 2, 1)))


# ---------------------------------------------------------------------------
# MTGFCIEncoder
# ---------------------------------------------------------------------------

class MTGFCIEncoder(nn.Module):
    """Encodes MTG FCI multi-temporal imagery to cross-attention context tokens.

    Input : (B, n_raw_channels + n_composite_channels = 18, H=256, W=256)
        Channels 0:n_raw_channels       — 5 timestamps x [vis06, vis08, cos_sza]
        Channels n_raw_channels:18      — [NDVI75, NDVI_std, CloudScore]
    Output: (B, N_ctx=256, D_ctx)  — sequence of spatial tokens for cross-attention

    The raw sequence and the composite reliability channels are encoded by
    separate stem branches, then fused (concat + 1x1 conv) before the shared
    strided CNN trunk that downsamples to 16×16 spatial resolution.  The 256
    spatial positions become the token sequence.
    """

    def __init__(
        self,
        n_raw_channels: int = 15,
        n_composite_channels: int = 3,
        d_ctx: int = 512,
        raw_stem_width: int = 64,
        composite_stem_width: int = 16,
    ):
        super().__init__()
        self.n_raw_channels       = n_raw_channels
        self.n_composite_channels = n_composite_channels

        self.raw_stem = nn.Sequential(
            nn.Conv2d(n_raw_channels, raw_stem_width, 3, padding=1), nn.SiLU(),
        )
        self.composite_stem = nn.Sequential(
            nn.Conv2d(n_composite_channels, composite_stem_width, 3, padding=1), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(raw_stem_width + composite_stem_width, raw_stem_width, 1), nn.SiLU(),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(raw_stem_width, 128, 3, stride=2, padding=1), nn.SiLU(),   # 256→128
            _ResBlock2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),  # 128→64
            _ResBlock2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.SiLU(),  # 64→32
            _ResBlock2d(512),
            nn.Conv2d(512, d_ctx, 3, stride=2, padding=1), nn.SiLU(),# 32→16
            _ResBlock2d(d_ctx),
        )
        self.proj = nn.Linear(d_ctx, d_ctx)
        self.norm = nn.LayerNorm(d_ctx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw       = x[:, :self.n_raw_channels]
        composite = x[:, self.n_raw_channels:self.n_raw_channels + self.n_composite_channels]
        raw_feat  = self.raw_stem(raw)
        comp_feat = self.composite_stem(composite)
        h      = self.fusion(torch.cat([raw_feat, comp_feat], dim=1))
        feat   = self.encoder(h)                           # (B, D_ctx, 16, 16)
        tokens = feat.flatten(2).permute(0, 2, 1)         # (B, 256, D_ctx)
        return self.norm(self.proj(tokens))


# ---------------------------------------------------------------------------
# MTGSpatialEncoder
# ---------------------------------------------------------------------------

class MTGSpatialEncoder(nn.Module):
    """Small CNN over MTG FCI producing a 32×32 feature map for FiLM/AdaGN
    modulation of the UNet (see unet.SpatialFiLM). Independent of
    MTGFCIEncoder's 256-token cross-attention branch — no shared weights.
    Follows the same stride-4 + ResBlock + stride-2 downsampling tail used
    by ERA5TemporalEncoder / LandCoverEmbedder.

    Input : (B, n_raw_channels + n_composite_channels = 18, H=256, W=256)
        Channels 0:n_raw_channels       — 5 timestamps x [vis06, vis08, cos_sza]
        Channels n_raw_channels:18      — [NDVI75, NDVI_std, CloudScore]
    Output: (B, out_channels, 32, 32)

    Like MTGFCIEncoder, the raw sequence and composite reliability channels
    are encoded by separate stem branches, fused (concat + 1x1 conv), then
    passed through the shared spatial downsampling tail.
    """

    def __init__(
        self,
        n_raw_channels: int = 15,
        n_composite_channels: int = 3,
        out_channels: int = 32,
        raw_stem_width: int = 32,
        composite_stem_width: int = 8,
    ):
        super().__init__()
        self.n_raw_channels       = n_raw_channels
        self.n_composite_channels = n_composite_channels

        self.raw_stem = nn.Sequential(
            nn.Conv2d(n_raw_channels, raw_stem_width, 3, padding=1), nn.SiLU(),
        )
        self.composite_stem = nn.Sequential(
            nn.Conv2d(n_composite_channels, composite_stem_width, 3, padding=1), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(raw_stem_width + composite_stem_width, raw_stem_width, 1), nn.SiLU(),
        )
        self.spatial_down = nn.Sequential(
            nn.Conv2d(raw_stem_width, out_channels, 4, stride=4),   # 256→64
            _ResBlock2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 2, stride=2),  # 64→32
        )
        self.out_norm = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw       = x[:, :self.n_raw_channels]
        composite = x[:, self.n_raw_channels:self.n_raw_channels + self.n_composite_channels]
        raw_feat  = self.raw_stem(raw)
        comp_feat = self.composite_stem(composite)
        h = self.fusion(torch.cat([raw_feat, comp_feat], dim=1))
        return F.silu(self.out_norm(self.spatial_down(h)))


# ---------------------------------------------------------------------------
# LandCoverEmbedder
# ---------------------------------------------------------------------------

class LandCoverEmbedder(nn.Module):
    """MCD12Q1 IGBP land cover class embedder.

    Input : (B, 1, 256, 256) float32 — integer class indices 0-17 stored as float
    Output: (B, out_channels, 32, 32) float32, or None when out_channels=0

    Each class index is looked up in a learned embedding table, then spatially
    downsampled 256→32 (8×) with the same stride-4 + ResBlock + stride-2 pattern
    used by ERA5TemporalEncoder.
    """

    def __init__(self, n_classes: int = 18, embed_dim: int = 32, out_channels: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.n_classes    = n_classes
        if out_channels > 0:
            self.embedding = nn.Embedding(n_classes, embed_dim)
            self.spatial_down = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, 4, stride=4),  # 256→64
                _ResBlock2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 2, stride=2),  # 64→32
            )
            self.out_norm = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, lc_tensor: torch.Tensor | None = None) -> torch.Tensor | None:
        if self.out_channels == 0 or lc_tensor is None:
            return None
        # lc_tensor: (B, 1, H, W) float32 with values 0-(n_classes-1)
        idx = lc_tensor[:, 0].long().clamp(0, self.n_classes - 1)  # (B, H, W)
        emb = self.embedding(idx)                                    # (B, H, W, embed_dim)
        emb = emb.permute(0, 3, 1, 2).contiguous()                  # (B, embed_dim, H, W)
        return F.silu(self.out_norm(self.spatial_down(emb)))         # (B, out_channels, 32, 32)


# ---------------------------------------------------------------------------
# NdviBackgroundEncoder
# ---------------------------------------------------------------------------

class NdviBackgroundEncoder(nn.Module):
    """MOD13A3 monthly NDVI background encoder.

    Input : (B, 1, 128, 128) float32 — prev-month composite at 1 km resolution
    Output: (B, out_channels, 32, 32) — channel-concat into UNet conv_in

    The 128-px patch covers the same 128 km footprint as the 256-px MOD09GA
    patches, so 4× downsampling (128→32) matches the 8× VAE downsampling of
    the 256-px inputs.  At inference time this is always the previous month's
    composite — no look-ahead required.
    """

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.out_channels = out_channels
        if out_channels > 0:
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(),
                _ResBlock2d(32),
                nn.Conv2d(32, out_channels, 4, stride=4),   # 128→32
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if self.out_channels == 0 or x is None:
            return None
        return self.net(x)
