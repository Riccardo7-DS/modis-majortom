"""Conditioning modules for the NDVI diffusion model.

ERA5TokenEncoder     – DenseNet-style Conv3d + global pool, (B,12,15,256,256) → (B,15,512) tokens
MTGFCIEncoder        – CNN image encoder,                   (B,6,3,256,256)   → (B,256,512) tokens
LandCoverEmbedder    – MCD12Q1 IGBP class embedder,        (B,1,256,256)     → (B,out_ch,32,32)

Both ERA5TokenEncoder and MTGFCIEncoder produce (B, N, D_ctx) token sequences that are
concatenated and fed into the UNet's cross-attention layers.  ERA5 contributes one token
per day (15 tokens), preserving temporal structure so the UNet can attend selectively to
individual days in the climate window.
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

    Input : (B, n_times=6, n_bands=3, H=256, W=256)
    Output: (B, N_ctx=256, D_ctx)  — sequence of spatial tokens for cross-attention

    All 6 × 3 = 18 channels are concatenated along the channel axis then
    passed through a strided CNN to 16×16 spatial resolution.  The 256
    spatial positions become the token sequence.
    """

    def __init__(
        self,
        n_times: int = 6,
        n_bands: int = 3,
        d_ctx: int = 512,
    ):
        super().__init__()
        in_c   = n_times * n_bands  # 18

        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),   # 256→128
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
        B, T, C, H, W = x.shape
        h      = x.reshape(B, T * C, H, W)
        feat   = self.encoder(h)                           # (B, D_ctx, 16, 16)
        tokens = feat.flatten(2).permute(0, 2, 1)         # (B, 256, D_ctx)
        return self.norm(self.proj(tokens))


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
