"""PipelineDiagnostics — VAE latent stats, roundtrip quality, and prediction distribution."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..model.vae import NdviVAE

log = logging.getLogger(__name__)


class PipelineDiagnostics:
    """Runs a structured quality audit of the VAE + diffusion pipeline.

    Steps:
        1+2. VAE latent statistics (mu-RMS, std, per-channel) and roundtrip MSE/PSNR.
        3.   Predicted NDVI distribution from a saved zarr (mean, std, range).
        4.   sigma_data calibration check.

    Args:
        vae_ckpt: Path to the NdviVAEModule checkpoint.
        modis_zarr: Path to the raw MOD09GA zarr (used to load real NDVI patches).
        pred_zarr: Path to the inference output zarr with ``ndvi_pred`` arrays.
        device: Torch device string for VAE computation.
    """

    def __init__(
        self,
        vae_ckpt: str,
        modis_zarr: str,
        pred_zarr: str,
        device: str = "cpu",
    ) -> None:
        self.vae_ckpt = vae_ckpt
        self.modis_zarr = modis_zarr
        self.pred_zarr = pred_zarr
        self.device = torch.device(device)

    # ── loaders ─────────────────────────────────────────────────────────────────

    def _load_vae(self) -> NdviVAE:
        raw = torch.load(self.vae_ckpt, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        vae_sd = {k.removeprefix("vae."): v for k, v in sd.items() if k.startswith("vae.")}
        vae = NdviVAE()
        vae.load_state_dict(vae_sd, strict=True)
        return vae.eval()

    def _load_real_ndvi(self, n: int = 64) -> torch.Tensor:
        """Load n real MODIS NDVI patches from zarr. Returns (N, 2, 256, 256)."""
        z = zarr.open(self.modis_zarr)
        patches: list[np.ndarray] = []
        for date in sorted(z["patches/sur_refl_b01"].keys()):
            b01_grp = z[f"patches/sur_refl_b01/{date}"]
            b02_grp = z[f"patches/sur_refl_b02/{date}"]
            for gid in b01_grp.keys():
                if gid not in b02_grp:
                    continue
                b01 = np.array(b01_grp[gid]).astype(np.float32)
                b02 = np.array(b02_grp[gid]).astype(np.float32)
                denom = b01 + b02
                valid = denom > 0.001
                ndvi = np.where(valid, (b02 - b01) / denom, 0.0).clip(-1, 1)
                patches.append(np.stack([ndvi, valid.astype(np.float32)], axis=0))
                if len(patches) >= n:
                    break
            if len(patches) >= n:
                break
        log.info("Loaded %d real MODIS NDVI patches from %s", len(patches), self.modis_zarr)
        return torch.from_numpy(np.stack(patches, axis=0))

    # ── steps ────────────────────────────────────────────────────────────────────

    def diagnose_vae(self) -> dict:
        """Compute VAE latent statistics and roundtrip reconstruction quality."""
        log.info("Step 1+2: VAE latent statistics & roundtrip")
        vae = self._load_vae().to(self.device)
        x = self._load_real_ndvi(n=64).to(self.device)

        with torch.no_grad():
            z, mu, logvar = vae.encode(x)
            x_recon = vae.decode(mu)

        mu_np = mu.cpu().numpy()
        mu_rms = float(np.sqrt(np.mean(mu_np ** 2)))
        mu_std = float(np.std(mu_np))
        mu_mean = float(np.mean(mu_np))
        per_channel_rms = [float(np.sqrt(np.mean(mu_np[:, c] ** 2)))
                           for c in range(mu_np.shape[1])]
        mse = float(torch.mean((x_recon - x) ** 2).cpu())
        psnr = -10.0 * np.log10(max(mse, 1e-10))

        log.info("  Latent μ-RMS  : %.4f  ← should equal sigma_data in config", mu_rms)
        log.info("  Latent μ-mean : %.4f", mu_mean)
        log.info("  Latent μ-std  : %.4f", mu_std)
        log.info("  Per-channel RMS: %s", [f"{v:.3f}" for v in per_channel_rms])
        log.info("  Roundtrip MSE : %.5f", mse)
        log.info("  Roundtrip PSNR: %.1f dB", psnr)

        return dict(
            mu_rms=mu_rms, mu_std=mu_std, mu_mean=mu_mean,
            per_channel_rms=per_channel_rms,
            mse=mse, psnr=psnr,
            x_sample=x[:4].cpu(),
            x_recon_sample=x_recon[:4].cpu(),
            mu_sample=mu[:4].cpu(),
        )

    def diagnose_predictions(self) -> dict:
        """Read prediction zarr and report NDVI distribution statistics."""
        log.info("Step 3: Predicted NDVI distribution")
        z = zarr.open(self.pred_zarr)
        preds, uncerts = [], []
        date = list(z["patches/ndvi_pred"].keys())[0]
        for gid in z[f"patches/ndvi_pred/{date}"].keys():
            arr = np.array(z[f"patches/ndvi_pred/{date}/{gid}"])
            mask = arr != 0
            if mask.sum() > 100:
                preds.append(arr[mask].ravel())
                unc = np.array(z[f"patches/uncertainty/{date}/{gid}"])
                uncerts.append(unc[mask].ravel())

        preds_all = np.concatenate(preds)
        uncerts_all = np.concatenate(uncerts)

        log.info("  N valid pixels : %d", len(preds_all))
        log.info("  NDVI pred mean : %.4f", preds_all.mean())
        log.info("  NDVI pred std  : %.4f  ← low std = degenerate output", preds_all.std())
        log.info("  NDVI pred range: [%.3f, %.3f]", preds_all.min(), preds_all.max())
        log.info("  Uncertainty mean: %.4f", uncerts_all.mean())

        return dict(
            preds=preds_all, uncerts=uncerts_all,
            pred_mean=float(preds_all.mean()),
            pred_std=float(preds_all.std()),
        )

    def check_sigma_data(self, mu_rms: float, expected: float = 1.0) -> None:
        """Warn if the latent RMS deviates >10% from the configured sigma_data."""
        log.info("Step 4: sigma_data calibration check")
        if abs(mu_rms / expected - 1.0) > 0.1:
            log.warning(
                "MISCALIBRATED: actual latent RMS=%.4f but sigma_data=%.1f. "
                "Update diffusion configs to set sigma_data=%.3f.",
                mu_rms, expected, mu_rms,
            )
        else:
            log.info("sigma_data=%.1f is approximately correct (latent RMS=%.4f).", expected, mu_rms)

    # ── plotting ─────────────────────────────────────────────────────────────────

    def plot(self, vae_results: dict, pred_results: dict, out_path: str | Path) -> None:
        """Render a 4-row diagnosis figure and save to out_path."""
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("Pipeline Diagnosis", fontsize=14, fontweight="bold")
        n_samples = 4

        for i in range(n_samples):
            ax = fig.add_subplot(4, n_samples + 2, i + 1)
            ax.imshow(vae_results["x_sample"][i, 0].numpy(), cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title(f"Input NDVI #{i}", fontsize=7)
            ax.axis("off")

        for i in range(n_samples):
            ax = fig.add_subplot(4, n_samples + 2, n_samples + 2 + i + 1)
            ax.imshow(vae_results["x_recon_sample"][i, 0].numpy(), cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title(f"Recon #{i}", fontsize=7)
            ax.axis("off")

        mu = vae_results["mu_sample"].numpy()
        for c in range(4):
            ax = fig.add_subplot(4, 4, 8 + c + 1)
            vals = mu[:, c].ravel()
            ax.hist(vals, bins=50, color="steelblue", alpha=0.7)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"Latent ch{c}\nRMS={np.sqrt(np.mean(vals**2)):.3f}", fontsize=7)
            ax.set_xlabel("μ value", fontsize=6)

        ax1 = fig.add_subplot(4, 2, 7)
        ax1.hist(pred_results["preds"], bins=100, color="green", alpha=0.7)
        ax1.set_title(
            f"Predicted NDVI distribution\n"
            f"mean={pred_results['pred_mean']:.3f}  std={pred_results['pred_std']:.3f}",
            fontsize=8,
        )
        ax1.set_xlabel("NDVI")
        ax1.axvline(pred_results["pred_mean"], color="red", lw=1)

        ax2 = fig.add_subplot(4, 2, 8)
        ax2.hist(pred_results["uncerts"], bins=100, color="orange", alpha=0.7)
        ax2.set_title(
            f"Prediction uncertainty (std across draws)\n"
            f"mean={pred_results['uncerts'].mean():.3f}",
            fontsize=8,
        )
        ax2.set_xlabel("std")

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
        log.info("Saved → %s", out_path)

    # ── entry point ──────────────────────────────────────────────────────────────

    def run(self, out_fig: str | Path = "outputs/diagnosis.png") -> None:
        """Run all steps, print a summary, and save the figure."""
        vae_results = self.diagnose_vae()
        self.check_sigma_data(vae_results["mu_rms"])
        pred_results = self.diagnose_predictions()
        self.plot(vae_results, pred_results, out_fig)

        print("\n" + "=" * 55)
        print("DIAGNOSIS SUMMARY")
        print("=" * 55)
        print(f"  VAE latent RMS (actual sigma_data) : {vae_results['mu_rms']:.4f}")
        print(f"  Config sigma_data                  : 1.0")
        print(f"  VAE roundtrip PSNR                 : {vae_results['psnr']:.1f} dB")
        print(f"  Predicted NDVI mean ± std          : {pred_results['pred_mean']:.3f} ± {pred_results['pred_std']:.3f}")
        print(f"  Mean uncertainty                   : {pred_results['uncerts'].mean():.3f}")
        print()
        if abs(vae_results["mu_rms"] - 1.0) > 0.1:
            print(f"  ⚠ sigma_data MISCALIBRATED: set to {vae_results['mu_rms']:.3f} in both configs")
        if pred_results["pred_std"] < 0.05:
            print(f"  ⚠ DEGENERATE OUTPUT: predicted NDVI std={pred_results['pred_std']:.3f} (near-constant)")
        print("=" * 55)
