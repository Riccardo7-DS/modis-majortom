# modis-majortom

A Python package for downloading, processing, and archiving MODIS satellite data with cloud masking capabilities and MajorTom grid integration. Handles data acquisition from NASA Earthdata, applies advanced cloud detection algorithms, performs reprojection, and exports data in Zarr or GeoTIFF format.

## Features

- Download MODIS/VIIRS products via NASA Earthdata (`earthaccess`)
- Mosaic multi-tile HDF files using GDAL VRT pipelines
- Reproject to EPSG:6933, UTM, or any target CRS
- Product-specific DN → physical units conversions (LST, NDVI, reflectance, night lights)
- Multi-sensor cloud masking (MOD09 QA + MOD35 + spectral veto + morphological processing)
- Optional ML-based cloud confidence scoring using a DINOv2-backed adapter model
- Export to chunked Zarr stores or cloud-optimized GeoTIFFs
- Optional cloud storage via MinIO/S3
- Google Earth Engine integration for land fraction filtering and auxiliary products
- Incremental processing: skips already-processed dates automatically

## Supported Products

| Name | Product ID | Resolution | Description |
|---|---|---|---|
| `reflectance_250m` | MOD09GQ | 250 m | Surface reflectance (bands 1–2) |
| `reflectance_500m` | MOD09GA | 500 m | Surface reflectance + state QA flags |
| `LST` | MOD11A1 | 1 km | Land Surface Temperature |
| `NDVI_1km_monthly` | MOD13A3 | 1 km | Monthly NDVI/EVI |
| `VIIRS_500m_night_daily` | VNP46A2 | 500 m | Daily night lights |
| `VIIRS_500m_night_monthly` | VNP46A3 | 500 m | Monthly night lights |
| `modis_cloud_mask` | MOD35_L2 | 1 km | Cloud detection confidence |

## Installation

Requires Python >= 3.11. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

The `majortom` dependency is pulled directly from ESA PhiLab's GitHub repository:

```toml
majortom = { git = "https://github.com/ESA-PhiLab/Major-TOM.git" }
```

## Authentication

**NASA Earthdata** — required for downloading MODIS data. Set credentials as environment variables or in a `.env` file:

```
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password
```

**Google Earth Engine** — required for land fraction filtering. Authenticate once via:

```bash
earthengine authenticate
```

and set project-ID as environment variables or in a `.env` file:

```
EE_PROJECT=your_gee_project_id
```

**MinIO/S3** (optional) — required only when using `--store_cloud`:

```
MINIO_ENDPOINT=your_endpoint
MINIO_ACCESS_KEY=your_key
MINIO_SECRET_KEY=your_secret
MINIO_BUCKET=your_bucket
```

## Usage

### CLI

```bash
python -m modis_majortom.eo_data.pipeline_data \
  --product reflectance_250m \
  --start_date 2024-01-01 \
  --end_date 2024-03-31 \
  --lon_min -10 --lon_max 40 \
  --lat_min -35 --lat_max 38 \
  --n_lon 5 --n_lat 5 \
  --output_format zarr \
  --reproj_lib rioxarray \
  --reproj_method bilinear
```

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--product` | `reflectance_250m` | MODIS product to download |
| `--start_date` | — | Start date (`YYYY-MM-DD`) |
| `--end_date` | — | End date (`YYYY-MM-DD`) |
| `--lon_min/max` | — | Longitude bounds |
| `--lat_min/max` | — | Latitude bounds |
| `--n_lon` / `--n_lat` | — | Number of tiles along each axis |
| `--batch_days` | `30` | Days per download batch |
| `--output_format` | `zarr` | Output format: `zarr` or `tiff` |
| `--reproj_lib` | `rioxarray` | Reprojection library: `rioxarray` or `xesmf` |
| `--reproj_method` | `nearest` | Interpolation: `nearest` or `bilinear` |
| `--majortom_grid` | `False` | Align output to MajorTom grid spec |
| `--store_cloud` | `False` | Upload output to MinIO/S3 |
| `--add_variables` | `False` | Add variables to existing Zarr without re-downloading |
| `--variables_override` | — | Comma-separated list of specific bands |

### Python API

```python
from modis_majortom.eo_data.modis import EarthAccessDownloader

downloader = EarthAccessDownloader(
    short_name="MOD11A1",
    variable=["LST_Day_1km", "QC_Day"],
    bbox=[-10, -35, 40, 38],
    start_date="2024-01-01",
    end_date="2024-03-31",
    output_format="zarr",
)
downloader.run()
```

```python
from modis_majortom.transform.cloud_adapter import generate_cloud_mask

result = generate_cloud_mask(
    mod09qa_bits=state_1km_array,       # uint16 QA flags
    mod35_confidence=cloud_conf_array,  # uint8 confidence
    blue_band=band3_reflectance,        # float32
)

print(result.cleaned_mask)   # bool (H, W) — True = cloudy
print(result.soft_score)     # float32 (H, W) — clearness confidence
```

## Architecture

```
pipeline_data.py (CLI)
       │
       ▼
EarthAccessDownloader (modis.py)
  ├── Search & Download (earthaccess)
  │     └── Skip already-processed dates via Zarr index
  ├── Parse & Mosaic (GDAL)
  │     ├── Open HDF subdatasets via GDAL
  │     ├── Reproject tiles to target CRS
  │     └── Merge via VRT + GDAL Translate
  ├── Product Preprocessing
  │     └── DN → physical units (LST, NDVI, night lights)
  ├── Cloud Masking (optional)
  │     └── generate_cloud_mask() (cloud_adapter.py)
  └── Export
        ├── Zarr (chunked, time-indexed)
        └── GeoTIFF (rasterio)
              └── Optional MinIO/S3 upload
```

## Cloud Masking

`generate_cloud_mask()` fuses three sources of cloud information:

1. **MOD09 QA flags** — state bits 0–1 from `MOD09GA`
2. **MOD35 confidence** — 4-level cloud confidence from `MOD35_L2`
3. **Blue band reflectance** — spectral veto: pixels with reflectance > 0.15 flagged as cloudy

Processing steps:
1. Extract cloud flags from each source
2. Apply spectral veto for disagreeing pixels
3. Morphological dilation (5×5 kernel) to buffer cloud edges
4. Union of dilated masks + morphological closing to fill holes
5. Soft confidence scoring: weighted blend (50% MOD35 + 30% MOD09 + 20% blue penalty)

The `meta_mask_channel` output is designed as a conditioning input for cloud inpainting models.

An optional `CloudAdapterModel` (DINOv2 ViT-S/14 backbone + lightweight adapter + upsampling decoder) is available for learning-based cloud probability estimation.

## Output Structure

**Zarr (default)**:
```
data/modis/
└── MOD09GQ_061/
    └── <bbox>/
        ├── MOD09GQ_dataset.zarr/
        │   ├── patches/
        │   │   ├── sur_refl_b01/
        │   │   │   └── <date>/<grid_id>/...
        │   │   └── sur_refl_b02/
        │   └── .zmetadata
        └── MOD09GQ_dataset.zarr.index.json
```

**GeoTIFF**:
```
data/modis/
└── MOD09GQ_061/
    └── <bbox>/
        └── <date>/
            └── <grid_id>.tif
```

## Utilities

- `generate_bboxes_fixed()` — divide a region into n×m equal tiles
- `tile_has_min_land_fraction()` — filter tiles by land coverage (GEE-based)
- `CalculationsMajorTom` — lat/lon ↔ MODIS sinusoidal ↔ tile (h,v) ↔ pixel conversions
- `compute_ndvi()` — NDVI from red and NIR arrays
- `minio_client()` — initialise MinIO S3 client from environment

## License

See [LICENSE](LICENSE) for details.
