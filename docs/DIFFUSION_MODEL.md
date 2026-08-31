# NDVI Diffusion Model

A latent diffusion model that predicts cloud-free MODIS NDVI (+ uncertainty) at 500 m from
MTG FCI (geostationary) and ERA5 conditioning. A frozen VAE compresses NDVI patches into a
latent space; an EDM-style diffusion model (`EDMDiffusion`) is trained on those latents.

This document is a from-scratch reconstruction guide, written from the current state of
`src/modis_majortom/`. The actual training entry point (`scripts/train_diffusion.py`) and the
per-run config YAMLs are **not** committed to this repo (`scripts/` is gitignored — it also holds
HPC job scripts and ad-hoc analysis scripts that shouldn't be public). Everything below is
enough to recreate an equivalent script/config pair from scratch; ask the repo owner directly if
you'd rather just get the actual files.

## 1. Architecture

- **VAE**: `NdviVAEv3` (`src/modis_majortom/model/vae_v2.py`, `vae_version="v3"`) — compresses an
  NDVI patch into a latent grid. Trained separately and frozen during diffusion training.
- **Diffusion model**: `EDMDiffusion` (`src/modis_majortom/model/diffusion.py`), a
  `pytorch_lightning.LightningModule` implementing the EDM (Karras et al.) formulation, sampled
  with DPM++2M at inference.
- **Conditioning encoders** (`src/modis_majortom/model/conditioning.py`):
  | Encoder | Input | Purpose |
  |---|---|---|
  | `MTGSpatialEncoder` / `MTGFCIEncoder` | MTG FCI raw + composite channels | Geostationary imagery, spatial FiLM into the UNet |
  | `ERA5TokenEncoder` | ERA5-Land daily variables | Cross-attention tokens (one per day in the lookback window) |
  | `LandCoverEmbedder` | MCD12Q1 IGBP classes | Channel-concat land-cover embedding |
  | `NdviBackgroundEncoder` | MOD13A3 monthly composite | Channel-concat "what NDVI usually looks like this month" prior |
- **Optional joint head**: `LAIHead` (`src/modis_majortom/model/lai_head.py`) — a deterministic
  (non-diffusion) cross-attention decoder that regresses LAI/FAPAR from the same conditioning
  tokens, supervised only on real MCD15A3H composite dates (see `add_lai` below).

## 2. Model version lineage (diff10 → diff18)

Each `diffN` is a config/architecture snapshot, not a code branch — they all run through the same
`EDMDiffusion` class with different constructor flags. Versions relevant here:

| Version | Adds on top of the previous version |
|---|---|
| diff14 | Baseline: VAE `vae9d_cont`, `ERA5TokenEncoder`, `MTGSpatialEncoder` FiLM, `use_soft_mask=False` |
| **diff15** | + Land cover conditioning (`LandCoverEmbedder`, `lc_out_channels=32`) |
| diff16 | + MOD13A3 monthly NDVI background (`NdviBackgroundEncoder`, `ndvi_bg_channels=16`) |
| **diff17** | + Joint LAI/FAPAR head (`add_lai=True`, `LAIHead`, MCD15A3H supervision) |
| diff18 | Later iteration — check `scripts/configs/diffusion_v18_hpc.yaml` if/when it's shared |

So **diff17 = diff15 + MOD13A3 background conditioning + the LAI head**; diff16 is the
intermediate step without the LAI head.

## 3. Prerequisites

### 3.1 Frozen VAE checkpoint

Both diff15 and diff17 use the same frozen VAE: `NdviVAEv3`, trained as `vae9d`, warm-started
from an earlier `vae9b` checkpoint, then continued (`vae9d_cont`) to
`step=059444, recon=0.0003`. To reproduce:

```bash
# fresh vae9d run, warm-started from a vae9b checkpoint
PYTHONPATH=src python scripts/train_vae.py \
    --model v3 \
    --config <your_vae9d_config.yaml> \
    --init_weights <path/to/vae9b-stepNNNNNN-reconX.ckpt>

# continuation run, resuming from vae9d's own last.ckpt
PYTHONPATH=src python scripts/train_vae.py \
    --model v3 \
    --config <your_vae9d_config.yaml> \
    --ckpt <path/to/vae9d/last.ckpt>
```

Key `vae9d` config choices vs. earlier VAE versions: no GroupNorm in `dec_out`,
`vae_channel_widths: [64, 128, 256, 256]`, `vae_use_output_norm: false`.

### 3.2 Data (zarr stores / files referenced by `data:` in the diffusion config)

| Config key | Source | Built with |
|---|---|---|
| `modis_raw_zarr` | MOD09GA surface reflectance | `python -m modis_majortom.eo_data.pipeline_data --product reflectance_500m ...` (see main README) |
| `modis_proc_zarr` | Whittaker-smoothed NDVI target (used when `whittaker_target: true`) | `scripts/compute_whittaker_features.py`, using `NdviWhittakerSmoother` (`src/modis_majortom/transform/ndvi_whittaker.py`) |
| `lc_zarr` | MCD12Q1 land cover (IGBP) | `scripts/download_MCD12Q1_africa.py` (or the equivalent `*_latin_america.py` variant) |
| `mod13a3_zarr` | MOD13A3 monthly NDVI composite | `python -m modis_majortom.eo_data.pipeline_data --product NDVI_1km_monthly ...` |
| `lai_zarr` (diff17 only) | MCD15A3H LAI/FAPAR 4-day composite | `scripts/download_MCD15A3H.py`, consumed by `LAISource` (`src/modis_majortom/transform/dataset.py`) |
| `fci_zarr` | MTG FCI geostationary imagery | **External** — built by the sibling `eumetsearch` package (`pyproject.toml` pulls it as a local editable dependency, not part of this repo) |
| `era5_path` | ERA5-Land daily NetCDF | **External** — in the reference run this came from a sibling project's ERA5 pipeline, not from anything in `modis-majortom`. You need an ERA5-Land NetCDF with the same variable/day layout `ERA5TokenEncoder` expects (`era5_vars`, `era5_days`) — see `src/modis_majortom/model/conditioning.py`. |

The two external dependencies (`fci_zarr`, `era5_path`) are the main things a colleague needs
from *outside* this repo — everything else is reproducible with scripts already in `scripts/`.

## 4. Diffusion training config

`scripts/train_diffusion.py --config <yaml>` loads a YAML with `data:`, `model:`, and `train:`
sections, mapped onto `DiffusionDataConfig` / `EDMDiffusion` kwargs / `DiffusionTrainConfig`
respectively (`src/modis_majortom/training/trainer_diffusion.py`).

### diff15 config

```yaml
data:
  fci_zarr:        <path>/MTG_FCI_mtg_2025_06.zarr
  modis_raw_zarr:  <path>/MOD09GA_dataset.zarr
  modis_proc_zarr: <path>/MOD09GA_processed.zarr   # Whittaker target
  era5_path:       <path>/era5land_africa_2025.nc
  lc_zarr:         <path>/MCD12Q1_majortom_africa_eu.zarr
  max_fci_nan_fraction: 0.10
  whittaker_target: true
  max_lap_var: 0.05

model:
  vae_version: "v3"
  vae_channel_widths: [64, 128, 256, 256]
  vae_ckpt: <path>/vae9d_cont-step059444-recon0.0003.ckpt
  vae_use_output_norm: false
  era5_vars: 12
  era5_days: 15
  mtg_raw_channels: 15
  mtg_composite_channels: 2
  context_dim: 512
  latent_dim: 32
  sigma_data: 1.0
  latent_channel_scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  cfg_p: 0.1
  lr: 2.0e-4
  lc_n_classes: 18
  lc_embed_dim: 32
  lc_out_channels: 32          # land-cover conditioning — new in diff15
  era5_encoder: "token"
  mtg_spatial_channels: 32
  use_soft_mask: false

train:
  batch_size: 8
  num_workers: 16
  max_steps: 200000
  val_every: 5000
  save_every: 5000
  precision: "bf16-mixed"
  root_dir: <your_lightning_logs_dir>
```

### diff17 config — diff15 + MOD13A3 background + LAI head

Same as above, plus:

```yaml
data:
  # ...as diff15, plus:
  mod13a3_zarr:    <path>/MOD13A3_dataset.zarr
  lai_zarr:        <path>/MCD15A3H_dataset.zarr

model:
  # ...as diff15, plus:
  ndvi_bg_channels: 16         # MOD13A3 monthly background — new in diff16
  add_lai: true                # joint LAI/FAPAR head — new in diff17
  lai_head_channels: 128
  lai_head_blocks: 2
  lai_query_size: 32
  lai_loss_weight: 0.1
  fapar_loss_weight: 0.1
```

`add_lai=False` (the default) is byte-for-byte the pre-LAI model — no new params get allocated
and no LAI batch keys are read, so it's safe to leave `lai_zarr` unset unless `add_lai: true`.

For diff16 (the intermediate step, no LAI head), use the diff15 config plus only the
`mod13a3_zarr` / `ndvi_bg_channels` additions, and leave `add_lai` unset.

## 5. Running

```bash
PYTHONPATH=src python scripts/train_diffusion.py --config <your_config.yaml>

# resume from a checkpoint
PYTHONPATH=src python scripts/train_diffusion.py --config <your_config.yaml> --ckpt <path/to/last.ckpt>

# sample from a trained checkpoint (DPM++2M, 20 steps by default)
PYTHONPATH=src python scripts/train_diffusion.py --config <your_config.yaml> \
    --ckpt <path/to/checkpoint.ckpt> --sample_only --steps 20 --guidance 3.0
```

## 6. Checkpoint naming convention

Checkpoints are named `diff<version>[-cont[N]]-step<NNNNNN>[-val<loss>].ckpt`, e.g.
`diff17-step025000-val0.0840.ckpt`. "Latest" = highest step number; "best" = lowest val loss —
the two don't always coincide, so check both when picking a checkpoint for inference.
