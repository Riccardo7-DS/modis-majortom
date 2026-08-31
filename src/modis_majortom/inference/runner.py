"""InferenceRunner — ensemble inference loop, zarr writer, and preview visualisation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
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

from ..model.diffusion import EDMDiffusion
from ..transform.dataset import ERA5Source, MOD13A3Source
from ..transform.land_cover import LandCoverSource
from .dataset import InferencePatchDataset, build_inference_index

log = logging.getLogger(__name__)

_NDVI_KW = dict(cmap="RdYlGn", vmin=-0.2, vmax=0.8)


@dataclass
class InferenceConfig:
    """All knobs for a production inference run."""
    diff_ckpt: str
    fci_zarr: str
    era5_path: str
    out_zarr: str = "outputs/ndvi_predictions.zarr"
    out_fig: str = "outputs/inference_preview.png"
    era5_days: int = 15
    era5_vars: int | None = None
    n_viz: int = 6
    k_draws: int = 5
    batch_size: int = 8
    guidance: float = 3.0
    steps: int = 20
    seed: int = 42
    device: str = "auto"
    skip_existing: bool = False
    max_samples: int | None = None
    dates: list[str] | None = None
    grid_ids: list[str] | None = None
    modis_proc_zarr: str | None = None
    lc_zarr: str | None = None
    mod13a3_zarr: str | None = None


class InferenceRunner:
    """Loads a trained EDMDiffusion, runs ensemble inference, writes zarr, saves a figure.

    Args:
        cfg: All inference parameters.
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg

    # ── helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @staticmethod
    def _existing_pairs(zarr_path: Path) -> set[tuple[str, str]]:
        if not zarr_path.exists():
            return set()
        root = zarr.open(str(zarr_path), mode="r")
        try:
            pred = root["patches"]["ndvi_pred"]
        except KeyError:
            return set()
        return {(gid, date) for date in pred.keys() for gid in pred[date].keys()}

    @staticmethod
    def _write_prediction(
        root: zarr.Group,
        grid_id: str,
        date: str,
        ndvi_pred: np.ndarray,
        uncertainty: np.ndarray,
    ) -> None:
        for var, arr in [("ndvi_pred", ndvi_pred), ("uncertainty", uncertainty)]:
            grp = root["patches"].require_group(var).require_group(date)
            grp.create_array(grid_id, data=arr.astype(np.float32), overwrite=True)

    # ── model loading ────────────────────────────────────────────────────────────

    def load_model(self, device: torch.device) -> EDMDiffusion:
        """Load EDMDiffusion from checkpoint, stripping the vae_ckpt hyper-param."""
        log.info("Loading model from %s …", self.cfg.diff_ckpt)
        state = torch.load(self.cfg.diff_ckpt, map_location="cpu")
        hp = dict(state.get("hyper_parameters", {}))
        sd = state["state_dict"]
        # Checkpoints trained without era5_encoder hparam always used ERA5TokenEncoder.
        if "era5_encoder" not in hp:
            hp["era5_encoder"] = "token" if "era5_enc.proj.weight" in sd else "spatial"
        model = EDMDiffusion(**{k: v for k, v in hp.items() if k != "vae_ckpt"})
        model.load_state_dict(sd, strict=True)
        if "ema_shadow" in state:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in state["ema_shadow"]:
                        param.data.copy_(state["ema_shadow"][name])
            log.info("  Applied EMA shadow weights.")
        model = model.to(device).eval()
        log.info("  %.1fM parameters", sum(p.numel() for p in model.parameters()) / 1e6)
        return model

    # ── inference loop ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_loop(
        self,
        model: EDMDiffusion,
        dl: DataLoader,
        device: torch.device,
        viz_indices: set[int],
    ) -> list[dict]:
        out_zarr = Path(self.cfg.out_zarr)
        root = zarr.open(str(out_zarr), mode="a")
        root.require_group("patches")

        viz_samples: list[dict] = []
        global_idx = 0

        proc_store = None
        if self.cfg.modis_proc_zarr:
            proc_store = zarr.open(self.cfg.modis_proc_zarr, mode="r", use_consolidated=False)

        for batch in tqdm(dl, desc="Inference", unit="batch"):
            B = batch["X"].shape[0]
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            draws = [
                model.sample(batch_dev, steps=self.cfg.steps, guidance=self.cfg.guidance)
                .cpu().float().numpy()
                for _ in range(self.cfg.k_draws)
            ]
            stack = np.stack(draws, axis=0)       # (K, B, 2, H, W)
            mean_ndvi = stack[:, :, 0].mean(axis=0)
            std_ndvi = stack[:, :, 0].std(axis=0)

            for i in range(B):
                self._write_prediction(
                    root, batch["grid_id"][i], batch["date"][i],
                    np.clip(mean_ndvi[i], -1.0, 1.0),
                    np.clip(std_ndvi[i], 0.0, 2.0),
                )
                if global_idx + i in viz_indices:
                    ndvi_gt = None
                    if proc_store is not None:
                        try:
                            ndvi_gt = proc_store["patches"]["ndvi_envelope"][
                                batch["date"][i]][batch["grid_id"][i]][:]
                        except KeyError:
                            pass
                    lc_arr = None
                    if "land_cover" in batch:
                        lc_arr = batch["land_cover"][i, 0].numpy()  # (H, W)
                    viz_samples.append({
                        "X":         batch["X"][i].numpy(),
                        "mean_ndvi": mean_ndvi[i],
                        "std_ndvi":  std_ndvi[i],
                        "grid_id":   batch["grid_id"][i],
                        "date":      batch["date"][i],
                        "cell_lat":  float(batch["cell_lat"][i]),
                        "cell_lon":  float(batch["cell_lon"][i]),
                        "ndvi_gt":   ndvi_gt,
                        "land_cover": lc_arr,
                    })
            global_idx += B

        return viz_samples

    # ── visualisation ────────────────────────────────────────────────────────────

    @staticmethod
    def _fci_rgb(X: np.ndarray) -> np.ndarray:
        """(18, H, W) → (H, W, 3) uint8 false-colour from the best FCI frame."""
        for i in range(4, -1, -1):
            fr = X[i * 3:i * 3 + 3]
            if fr.max() > 0:
                break
        r = np.clip(fr[0] / 100.0, 0, 1)
        g = np.clip(fr[1] / 100.0, 0, 1)
        b = np.clip(((fr[1] - fr[0]) / np.maximum(fr[1] + fr[0], 1e-6) + 1) / 2, 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    @staticmethod
    def _cb(ax, im, fmt: str = "%.2f") -> None:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format=fmt)
        cb.ax.tick_params(labelsize=6)

    # IGBP LC_Type1 class names (index 0 = unclassified)
    _LC_NAMES = [
        "Unclassified", "Evergreen Needleleaf", "Evergreen Broadleaf",
        "Deciduous Needleleaf", "Deciduous Broadleaf", "Mixed Forest",
        "Closed Shrublands", "Open Shrublands", "Woody Savannas",
        "Savannas", "Grasslands", "Permanent Wetlands",
        "Croplands", "Urban", "Cropland/Natural Veg.", "Snow/Ice",
        "Barren", "Water",
    ]
    _LC_COLORS = [
        "#000000", "#05450a", "#086a10", "#54a708", "#78d203", "#009900",
        "#c6b044", "#dcd159", "#dade48", "#fbff13", "#b6ff05", "#27ff87",
        "#c24f44", "#a5a5a5", "#ff6d4c", "#69fff8", "#f9ffa4", "#1c0dff",
    ]

    def plot_preview(self, viz_samples: list[dict]) -> None:
        """Render and save the 7-panel preview figure per sample row.

        Columns: FCI RGB | FCI NDVI | Pred NDVI | Std | MODIS GT | |Error| | Land Cover
        """
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        n = len(viz_samples)
        has_gt  = any(s.get("ndvi_gt")    is not None for s in viz_samples)
        has_lc  = any(s.get("land_cover") is not None for s in viz_samples)

        # Always use 7 columns when LC available; fall back gracefully
        col_names = ["MTG FCI\n(false colour)", "FCI NDVI\n(75-pctile)",
                     "Predicted NDVI", "Uncertainty\n(std)"]
        if has_gt:
            col_names += ["MODIS GT\n(ndvi_envelope)", "|GT − Pred|"]
        if has_lc:
            col_names += ["Land Cover\n(LC_Type1)"]
        n_cols = len(col_names)

        out_path = Path(self.cfg.out_fig)
        fig = plt.figure(figsize=(n_cols * 3.0, n * 3.2 + 0.6))
        fig.suptitle(
            "EDMDiffusion + Land Cover — cloud-free NDVI  (MTG FCI + ERA5 + LC)",
            fontsize=11, y=0.999,
        )
        gs = GridSpec(n, n_cols, figure=fig, hspace=0.06, wspace=0.04,
                      top=0.975, bottom=0.01, left=0.05, right=0.97)

        lc_cmap  = mcolors.ListedColormap(self._LC_COLORS)
        lc_norm  = mcolors.BoundaryNorm(np.arange(-0.5, 18.5), lc_cmap.N)

        for row, s in enumerate(viz_samples):
            X, pr, std = s["X"], s["mean_ndvi"], s["std_ndvi"]
            fci_ndvi = X[15]
            axs = [fig.add_subplot(gs[row, c]) for c in range(n_cols)]
            col = 0

            # 0 — FCI RGB
            axs[col].imshow(self._fci_rgb(X))
            axs[col].set_ylabel(f"{s['grid_id']}\n{s['date']}", fontsize=6)
            col += 1

            # 1 — FCI NDVI
            im = axs[col].imshow(fci_ndvi, **_NDVI_KW); self._cb(axs[col], im); col += 1

            # 2 — Predicted NDVI
            im = axs[col].imshow(pr, **_NDVI_KW); self._cb(axs[col], im); col += 1

            # 3 — Uncertainty
            im = axs[col].imshow(std, cmap="plasma", vmin=0,
                                  vmax=max(float(std.max()), 0.01))
            self._cb(axs[col], im, fmt="%.3f"); col += 1

            if has_gt:
                gt = s.get("ndvi_gt")
                # 4 — MODIS GT
                if gt is not None:
                    im = axs[col].imshow(gt, **_NDVI_KW); self._cb(axs[col], im)
                else:
                    axs[col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                  transform=axs[col].transAxes, fontsize=9, color="grey")
                col += 1

                # 5 — |Error|
                if gt is not None:
                    err = np.abs(gt - pr)
                    im = axs[col].imshow(err, cmap="Reds", vmin=0,
                                         vmax=max(float(err.max()), 0.01))
                    self._cb(axs[col], im, fmt="%.2f")
                else:
                    axs[col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                  transform=axs[col].transAxes, fontsize=9, color="grey")
                col += 1

            if has_lc:
                lc = s.get("land_cover")
                # 6 — Land Cover
                if lc is not None:
                    im = axs[col].imshow(lc, cmap=lc_cmap, norm=lc_norm,
                                         interpolation="nearest")
                    present = sorted(set(lc.ravel().astype(int)) & set(range(18)))
                    patches = [
                        mpatches.Patch(color=self._LC_COLORS[i],
                                       label=f"{i} {self._LC_NAMES[i][:12]}")
                        for i in present
                    ]
                    axs[col].legend(handles=patches, loc="lower left",
                                    fontsize=4, framealpha=0.6,
                                    ncol=max(1, len(patches) // 6))
                else:
                    axs[col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                  transform=axs[col].transAxes, fontsize=9, color="grey")
                col += 1

            for c, ax in enumerate(axs):
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_names[c], fontsize=7.5, pad=3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved figure → %s", out_path)

    # ── entry point ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full inference pipeline."""
        import random as _random  # local to avoid shadowing the module
        cfg = self.cfg
        _random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        device = self._resolve_device(cfg.device)
        log.info("Device: %s", device)

        model = self.load_model(device)

        era5_src = ERA5Source(nc_path=cfg.era5_path, n_days=cfg.era5_days)
        if cfg.era5_vars is not None:
            era5_src.variables = era5_src.variables[:cfg.era5_vars]
        skip_set = self._existing_pairs(Path(cfg.out_zarr)) if cfg.skip_existing else None
        if skip_set:
            log.info("Skipping %d already-written samples", len(skip_set))

        samples = build_inference_index(cfg.fci_zarr, era5_src, skip_set)
        if cfg.dates:
            allowed = set(cfg.dates)
            samples = [(g, d) for g, d in samples if d in allowed]
        if cfg.grid_ids:
            allowed_gids = set(cfg.grid_ids)
            samples = [(g, d) for g, d in samples if g in allowed_gids]
        log.info("Total inference samples: %d", len(samples))

        if not samples:
            log.warning("Nothing to process.")
            return

        # When GT is available, prefer viz samples that have MODIS GT (scan before max_samples cap)
        if cfg.modis_proc_zarr:
            proc = zarr.open(cfg.modis_proc_zarr, mode="r", use_consolidated=False)
            try:
                env = proc["patches"]["ndvi_envelope"]
                gt_pairs = {
                    (gid, date)
                    for date in env.keys()
                    for gid in env[date].keys()
                }
            except KeyError:
                gt_pairs = set()
            gt_indices = [i for i, (g, d) in enumerate(samples) if (g, d) in gt_pairs]
            if len(gt_indices) >= cfg.n_viz:
                viz_idx = set(_random.sample(gt_indices, cfg.n_viz))
                log.info("viz samples chosen from %d GT-available indices", len(gt_indices))
            else:
                viz_idx = set(_random.sample(range(len(samples)), min(cfg.n_viz, len(samples))))
        else:
            viz_idx = set(_random.sample(range(len(samples)), min(cfg.n_viz, len(samples))))

        if cfg.max_samples:
            # Ensure viz indices are within the capped range; if not, include them anyway
            capped = set(range(min(cfg.max_samples, len(samples))))
            missing = viz_idx - capped
            if missing:
                samples_extra = [samples[i] for i in sorted(missing)]
                samples = samples[:cfg.max_samples] + samples_extra
                viz_idx = {
                    i if i < cfg.max_samples else cfg.max_samples + sorted(missing).index(i)
                    for i in viz_idx
                }
            else:
                samples = samples[:cfg.max_samples]
            log.info("Capped to %d samples (max_samples=%d)", len(samples), cfg.max_samples)
        lc_source = None
        if cfg.lc_zarr:
            lc_source = LandCoverSource(zarr_path=cfg.lc_zarr, variables=["LC_Type1"])
            log.info("Land cover source: %s", cfg.lc_zarr)

        mod13a3_source = None
        if cfg.mod13a3_zarr:
            mod13a3_source = MOD13A3Source(zarr_path=cfg.mod13a3_zarr)
            log.info("MOD13A3 source: %s", cfg.mod13a3_zarr)

        dataset = InferencePatchDataset(
            fci_store_path=cfg.fci_zarr,
            era5_source=era5_src,
            samples=samples,
            lc_source=lc_source,
            mod13a3_source=mod13a3_source,
        )
        dl = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )

        log.info("Running inference (k=%d draws, steps=%d, guidance=%.1f) …",
                 cfg.k_draws, cfg.steps, cfg.guidance)
        viz_samples = self._run_loop(model, dl, device, viz_idx)
        log.info("Written to zarr: %s", cfg.out_zarr)

        if viz_samples:
            log.info("Rendering preview figure (%d samples) …", len(viz_samples))
            self.plot_preview(viz_samples)

        log.info("Done.")
