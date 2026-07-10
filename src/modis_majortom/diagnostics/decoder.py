"""Decoder diagnostics: spatial bias maps and v2-decoder-swapped inference."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.vae import NdviVAE, NdviVAEModule
from ..model.vae_v2 import NdviVAEv2, NdviVAEv2Module
from ..model.diffusion import EDMDiffusion
from ..transform.dataset import ERA5Source
from ..inference.dataset import InferencePatchDataset, build_inference_index
from ..inference.runner import InferenceRunner, _NDVI_KW

log = logging.getLogger(__name__)

_DIFF_KW = dict(cmap="RdBu_r", vmin=-0.15, vmax=0.15)
_BIAS_KW = dict(cmap="RdBu_r", vmin=-0.05, vmax=0.05)


def _load_vae1(ckpt_path: str, device: torch.device) -> NdviVAE:
    state = torch.load(ckpt_path, map_location="cpu")
    hp = state.get("hyper_parameters", {})
    mod = NdviVAEModule(vae=NdviVAE(
        **{k: v for k, v in hp.items() if k in ("in_channels", "latent_dim", "channel_widths")}
    ))
    mod.load_state_dict(state["state_dict"], strict=True)
    return mod.vae.to(device).eval()


def _load_vae2(ckpt_path: str, device: torch.device) -> NdviVAEv2:
    state = torch.load(ckpt_path, map_location="cpu")
    hp = state.get("hyper_parameters", {})
    mod = NdviVAEv2Module(vae=NdviVAEv2(
        **{k: v for k, v in hp.items() if k in ("in_channels", "latent_dim", "channel_widths")}
    ))
    mod.load_state_dict(state["state_dict"], strict=True)
    return mod.vae.to(device).eval()


@torch.no_grad()
def _spatial_bias_map(
    vae: torch.nn.Module,
    device: torch.device,
    n_samples: int = 64,
    latent_shape: tuple[int, ...] = (4, 32, 32),
) -> np.ndarray:
    """Mean decoded output over N random N(0,1) latents. Flat ≈ no bias; ring ≈ decoder bias."""
    acc: np.ndarray | None = None
    for _ in range(n_samples):
        z = torch.randn(1, *latent_shape, device=device)
        out = vae.decode(z).float().cpu().numpy()[0, 0]
        acc = out if acc is None else acc + out
    return acc / n_samples  # (H, W)


# ── DecoderBiasDiagnostics ───────────────────────────────────────────────────


class DecoderBiasDiagnostics:
    """Compare VAE v1 vs v2 decoders to locate the origin of spatial artifacts.

    Three sections:
      A. Spatial bias maps — average output over N random N(0,1) latents per decoder.
      B. Real patch encode-decode — encode with v1, decode with both.
      C. Same Gaussian z decoded by both (seed=42).

    Args:
        v1_ckpt: Path to NdviVAEModule (v1) checkpoint.
        v2_ckpt: Path to NdviVAEv2Module (v2) checkpoint.
        out_fig: Output figure path.
        zarr_path: Optional MODIS zarr for real patches (sections B is skipped if absent).
        device: Torch device string or "auto".
        n_real: Number of real patches for section B.
        n_bias: Number of random latents for section A bias maps.
    """

    def __init__(
        self,
        v1_ckpt: str,
        v2_ckpt: str,
        out_fig: str = "outputs/decoder_bias_test.png",
        zarr_path: str | None = None,
        device: str = "auto",
        n_real: int = 4,
        n_bias: int = 64,
    ) -> None:
        self.v1_ckpt = v1_ckpt
        self.v2_ckpt = v2_ckpt
        self.out_fig = Path(out_fig)
        self.zarr_path = Path(zarr_path) if zarr_path else None
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else torch.device(device))
        self.n_real = n_real
        self.n_bias = n_bias

    def _load_real_patches(self) -> np.ndarray | None:
        if not self.zarr_path or not self.zarr_path.exists():
            return None
        try:
            store = zarr.open(str(self.zarr_path), mode="r")
            patches: list[np.ndarray] = []
            for date in store["patches"]["ndvi_envelope"].keys():
                for gid in store["patches"]["ndvi_envelope"][date].keys():
                    try:
                        ndvi = store["patches"]["ndvi_envelope"][date][gid][:]
                        sw = store["patches"]["soft_score"][date][gid][:]
                        if np.isnan(ndvi).mean() < 0.1:
                            patches.append(np.stack([ndvi, sw], axis=0).astype(np.float32))
                    except Exception:
                        continue
                    if len(patches) >= self.n_real:
                        break
                if len(patches) >= self.n_real:
                    break
            return np.stack(patches[: self.n_real], axis=0) if patches else None
        except Exception as e:
            log.warning("Could not load real patches (%s), skipping section B.", e)
            return None

    @staticmethod
    def _cb(ax, im, fmt: str = "%.2f") -> None:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format=fmt)
        cb.ax.tick_params(labelsize=6)

    def run(self) -> None:
        log.info("Device: %s", self.device)
        vae1 = _load_vae1(self.v1_ckpt, self.device)
        vae2 = _load_vae2(self.v2_ckpt, self.device)
        log.info("Both VAEs loaded.")

        # A: spatial bias maps
        log.info("Computing spatial bias maps (%d random latents)…", self.n_bias)
        bias1 = _spatial_bias_map(vae1, self.device, self.n_bias)
        bias2 = _spatial_bias_map(vae2, self.device, self.n_bias)
        bias_diff = bias1 - bias2

        # B: real patch encode-decode
        real_patches = self._load_real_patches()
        real_out1 = real_out2 = None
        if real_patches is not None:
            x_t = torch.from_numpy(real_patches).to(self.device)
            with torch.no_grad():
                _, mu, _ = vae1.encode(x_t)
                real_out1 = vae1.decode(mu).float().cpu().numpy()[:, 0]
                real_out2 = vae2.decode(mu).float().cpu().numpy()[:, 0]

        # C: same Gaussian z
        torch.manual_seed(42)
        noise_z = torch.randn(4, 4, 32, 32, device=self.device)
        with torch.no_grad():
            noise_out1 = vae1.decode(noise_z).float().cpu().numpy()[:, 0]
            noise_out2 = vae2.decode(noise_z).float().cpu().numpy()[:, 0]

        # Figure
        n_rows_real = self.n_real if real_patches is not None else 0
        total_rows = 1 + n_rows_real + 4

        fig = plt.figure(figsize=(4 * 3.5, total_rows * 3.5 + 1.0))
        fig.suptitle("VAE decoder bias test  —  same latent → VAE v1 vs VAE v2",
                     fontsize=12, y=0.999)
        gs = GridSpec(total_rows, 4, figure=fig, hspace=0.08, wspace=0.05,
                      top=0.97, bottom=0.01, left=0.06, right=0.97)
        col_titles = ["VAE v1 (zero pad)", "VAE v2 (reflect pad)", "Diff (v1 − v2)", "Info"]
        row = 0

        # Row A: bias maps
        axs = [fig.add_subplot(gs[row, c]) for c in range(4)]
        im0 = axs[0].imshow(bias1,    **_BIAS_KW); self._cb(axs[0], im0, "%.3f")
        im1 = axs[1].imshow(bias2,    **_BIAS_KW); self._cb(axs[1], im1, "%.3f")
        im2 = axs[2].imshow(bias_diff, **_BIAS_KW); self._cb(axs[2], im2, "%.3f")
        axs[3].axis("off")
        axs[3].text(0.1, 0.5,
                    f"Spatial bias map\n({self.n_bias}× random N(0,1) latents)\n\n"
                    f"v1 range: [{bias1.min():.3f}, {bias1.max():.3f}]\n"
                    f"v2 range: [{bias2.min():.3f}, {bias2.max():.3f}]",
                    transform=axs[3].transAxes, va="center", fontsize=8)
        for c, (ax, t) in enumerate(zip(axs, col_titles)):
            ax.set_title(t, fontsize=8, pad=3); ax.set_xticks([]); ax.set_yticks([])
        axs[0].set_ylabel("A: decoder bias\n(random z)", fontsize=7)
        row += 1

        # Row B: real patches
        if real_patches is not None and real_out1 is not None:
            real_orig = real_patches[:, 0]
            for i in range(self.n_real):
                axs = [fig.add_subplot(gs[row, c]) for c in range(4)]
                diff = real_out1[i] - real_out2[i]
                im0 = axs[0].imshow(real_out1[i], **_NDVI_KW); self._cb(axs[0], im0)
                im1 = axs[1].imshow(real_out2[i], **_NDVI_KW); self._cb(axs[1], im1)
                im2 = axs[2].imshow(diff,          **_DIFF_KW); self._cb(axs[2], im2)
                im3 = axs[3].imshow(real_orig[i],  **_NDVI_KW); self._cb(axs[3], im3)
                for ax in axs:
                    ax.set_xticks([]); ax.set_yticks([])
                axs[0].set_ylabel(f"B: real patch {i+1}\n(v1 encoded)", fontsize=7)
                if i == 0:
                    axs[3].set_title("Original NDVI", fontsize=8, pad=3)
                row += 1

        # Row C: Gaussian z
        for i in range(4):
            axs = [fig.add_subplot(gs[row, c]) for c in range(4)]
            diff = noise_out1[i] - noise_out2[i]
            im0 = axs[0].imshow(noise_out1[i], **_NDVI_KW); self._cb(axs[0], im0)
            im1 = axs[1].imshow(noise_out2[i], **_NDVI_KW); self._cb(axs[1], im1)
            im2 = axs[2].imshow(diff,           **_DIFF_KW); self._cb(axs[2], im2)
            axs[3].axis("off")
            axs[3].text(0.1, 0.5,
                        f"seed 42, sample {i+1}\n(no encoder used)\n\n"
                        f"v1 mean: {noise_out1[i].mean():.3f}\n"
                        f"v2 mean: {noise_out2[i].mean():.3f}",
                        transform=axs[3].transAxes, va="center", fontsize=8)
            for ax in axs[:3]:
                ax.set_xticks([]); ax.set_yticks([])
            axs[0].set_ylabel(f"C: noise z #{i+1}", fontsize=7)
            row += 1

        self.out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(self.out_fig), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved → %s", self.out_fig)

        # Summary
        print("\n=== Spatial bias summary ===")
        border = lambda b: np.concatenate([b[:10].ravel(), b[-10:].ravel(),
                                            b[:, :10].ravel(), b[:, -10:].ravel()]).mean()
        for name, b in [("v1", bias1), ("v2", bias2)]:
            delta = b[100:156, 100:156].mean() - border(b)
            print(f"{name} bias  mean={b.mean():.4f}  std={b.std():.4f}  center−border={delta:.4f}")
        print("If |center-border| >> 0 for v1 → decoder introduces the square.")


# ── V2DecoderInference ────────────────────────────────────────────────────────


class V2DecoderInference:
    """Runs diffusion sampling with the v1 UNet but a swapped-in NdviVAEv2 decoder.

    Used to isolate whether the square artifact is introduced during decoding
    (not during the denoising pass). Renders a visual comparison — no zarr output.

    Args:
        diff_ckpt: Path to the EDMDiffusion Lightning checkpoint.
        v2_ckpt: Path to the NdviVAEv2Module checkpoint.
        fci_zarr: Path to the MTG FCI zarr store.
        era5_path: Path to the ERA5 NetCDF file.
        out_fig: Output figure path.
        n_viz / k_draws / steps / guidance / max_samples / seed: inference knobs.
        device: Torch device string or "auto".
        era5_days: History window for ERA5 conditioning.
    """

    def __init__(
        self,
        diff_ckpt: str,
        v2_ckpt: str,
        fci_zarr: str,
        era5_path: str,
        out_fig: str = "outputs/diagnosis_calibrated.png",
        n_viz: int = 6,
        k_draws: int = 3,
        steps: int = 20,
        guidance: float = 3.0,
        max_samples: int | None = 12,
        seed: int = 42,
        device: str = "auto",
        era5_days: int = 15,
    ) -> None:
        self.diff_ckpt = diff_ckpt
        self.v2_ckpt = v2_ckpt
        self.fci_zarr = fci_zarr
        self.era5_path = era5_path
        self.out_fig = Path(out_fig)
        self.n_viz = n_viz
        self.k_draws = k_draws
        self.steps = steps
        self.guidance = guidance
        self.max_samples = max_samples
        self.seed = seed
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else torch.device(device))
        self.era5_days = era5_days

    def _load_model(self) -> EDMDiffusion:
        """Load diffusion checkpoint and swap v1 VAE for NdviVAEv2 decoder."""
        log.info("Loading diffusion checkpoint from %s …", self.diff_ckpt)
        state = torch.load(self.diff_ckpt, map_location="cpu")
        hp = state.get("hyper_parameters", {})
        model = EDMDiffusion(**{k: v for k, v in hp.items() if k != "vae_ckpt"})
        model.load_state_dict(state["state_dict"], strict=True)

        log.info("Swapping in VAE v2 decoder from %s …", self.v2_ckpt)
        vae2 = _load_vae2(self.v2_ckpt, torch.device("cpu"))
        model.vae = vae2
        for p in model.vae.parameters():
            p.requires_grad_(False)

        model = model.to(self.device).eval()
        log.info("  %.1fM parameters", sum(p.numel() for p in model.parameters()) / 1e6)
        return model

    @torch.no_grad()
    def _run_loop(
        self,
        model: EDMDiffusion,
        dl: DataLoader,
        viz_indices: set[int],
    ) -> list[dict]:
        viz_samples: list[dict] = []
        global_idx = 0
        for batch in tqdm(dl, desc="Inference (v2 decoder)", unit="batch"):
            B = batch["X"].shape[0]
            batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            draws = [
                model.sample(batch_dev, steps=self.steps, guidance=self.guidance)
                .cpu().float().numpy()
                for _ in range(self.k_draws)
            ]
            stack = np.stack(draws, axis=0)
            mean_ndvi = stack[:, :, 0].mean(axis=0)
            std_ndvi = stack[:, :, 0].std(axis=0)
            for i in range(B):
                if global_idx + i in viz_indices:
                    viz_samples.append({
                        "X":         batch["X"][i].numpy(),
                        "mean_ndvi": mean_ndvi[i],
                        "std_ndvi":  std_ndvi[i],
                        "grid_id":   batch["grid_id"][i],
                        "date":      batch["date"][i],
                        "cell_lat":  float(batch["cell_lat"][i]),
                        "cell_lon":  float(batch["cell_lon"][i]),
                    })
            global_idx += B
        return viz_samples

    def run(self) -> None:
        """Execute v2-decoder inference and save the preview figure."""
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

        model = self._load_model()
        era5_src = ERA5Source(nc_path=self.era5_path, n_days=self.era5_days)
        samples = build_inference_index(self.fci_zarr, era5_src)
        log.info("Total samples: %d", len(samples))
        if self.max_samples:
            samples = samples[: self.max_samples]

        viz_idx = set(random.sample(range(len(samples)), min(self.n_viz, len(samples))))
        dataset = InferencePatchDataset(
            fci_store_path=self.fci_zarr, era5_source=era5_src, samples=samples
        )
        dl = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=4, pin_memory=(self.device.type == "cuda"),
                        persistent_workers=True)

        log.info("Running inference (k=%d, steps=%d, guidance=%.1f)…",
                 self.k_draws, self.steps, self.guidance)
        viz_samples = self._run_loop(model, dl, viz_idx)

        if viz_samples:
            self._plot_preview(viz_samples)
        else:
            log.warning("No viz samples collected.")

    def _plot_preview(self, viz_samples: list[dict]) -> None:
        n = len(viz_samples)
        fig = plt.figure(figsize=(4 * 3.0, n * 3.0 + 0.5))
        fig.suptitle("EDMDiffusion — cloud-free NDVI (VAE v2 decoder)",
                     fontsize=11, y=0.998)
        gs = GridSpec(n, 4, figure=fig, hspace=0.05, wspace=0.04,
                      top=0.97, bottom=0.01, left=0.04, right=0.97)
        titles = ["MTG FCI (false colour)", "MTG FCI NDVI\n(75-pctile)", "Predicted NDVI", "Sample std"]
        cb = lambda ax, im, fmt="%.2f": (
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format=fmt)
            .ax.tick_params(labelsize=6)
        )
        for row, s in enumerate(viz_samples):
            X, pr, std = s["X"], s["mean_ndvi"], s["std_ndvi"]
            axs = [fig.add_subplot(gs[row, c]) for c in range(4)]
            axs[0].imshow(InferenceRunner._fci_rgb(X))
            axs[0].set_ylabel(f"{s['grid_id']}\n{s['date']}", fontsize=6.5)
            cb(axs[1], axs[1].imshow(X[5, 0], **_NDVI_KW))
            cb(axs[2], axs[2].imshow(pr, **_NDVI_KW))
            cb(axs[3], axs[3].imshow(std, cmap="plasma", vmin=0,
                                      vmax=max(float(std.max()), 0.01)), "%.3f")
            for c, ax in enumerate(axs):
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[c], fontsize=8, pad=3)
        self.out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(self.out_fig), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved → %s", self.out_fig)
