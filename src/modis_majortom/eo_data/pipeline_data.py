from .modis import EarthAccessDownloader
from ..utils import generate_bboxes_fixed, init_logging, tile_has_min_land_fraction, plot_boxes_on_map, generate_bboxes_from_resolution
# from osgeo import gdal
from ..definitions import ROOT_DIR, DATA_PATH
from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
import shutil
from ..utils import init_logging
import sys
from tqdm import tqdm
import numpy as np
from dask.diagnostics import ProgressBar
t = ProgressBar().register()

logger = init_logging(log_file="modis_downloader.log", verbose=False)

# Load environment variables
load_dotenv()

os.environ["PYDEVD_THREAD_DUMP_ON_WARN_EVALUATION_TIMEOUT"] = "true"
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT "] = "600"  # milliseconds
# # GDAL tuning
# gdal.SetConfigOption('GDAL_CACHEMAX', '512')    # MB, tune to memory
# gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
sys.dont_write_bytecode = True

# Argument parsing
parser = argparse.ArgumentParser(description="MODIS Downloader")
parser.add_argument('--product', type=str, default=os.getenv('product', 'reflectance_250m'), help='MODIS product to download')
parser.add_argument('--reproj_lib', choices=['rioxarray', 'xesmf'], default=os.getenv('REPROJ_LIB', 'rioxarray'), help='Reprojection library to use')
parser.add_argument('--reproj_method', choices=['nearest', 'bilinear'], default=os.getenv('REPROJ_METHOD', 'nearest'), help='Reprojection method to use')
parser.add_argument('-d', '--delete_temp', action='store_true', default=False, help='Delete temporary files after processing')
parser.add_argument('--lon_min', type=float, default=os.getenv('lon_min', -70))
parser.add_argument('--lat_min', type=float, default=os.getenv('lat_min', -70))
parser.add_argument('--lon_max', type=float, default=os.getenv('lon_max', 70))
parser.add_argument('--lat_max', type=float, default=os.getenv('lat_max', 70))
parser.add_argument('--n_lon', type=int, default=os.getenv('n_lon', 7))
parser.add_argument('--n_lat', type=int, default=os.getenv('n_lat', 7))
parser.add_argument('--start_date', type=str, default=os.getenv('start_date', "2025-07-01"))
parser.add_argument('--end_date', type=str, default=os.getenv('end_date', "2025-07-05"))
parser.add_argument('--batch_days', type=int, default=os.getenv("batch_days", 30), help='Number of days per download batch')
parser.add_argument("--store_cloud", action="store_true", help="Store data in cloud storage")
parser.add_argument("--majortom_grid", action="store_true", help="Reproject to MajorTom grid (only supported for Zarr output format)")
parser.add_argument("--output_format", type=str, choices=["tiff", "zarr"], default=os.getenv("output_format", "zarr"), help="Output format for downloaded data")
parser.add_argument("--add_variables", action="store_true", default=False, help="Add new variables to an existing zarr store without re-appending existing time steps")
parser.add_argument("--variables_override", type=str, default=None, help="Comma-separated list of variables to download instead of the product defaults (e.g. 'sur_refl_b03')")
args = parser.parse_args()

if args.majortom_grid and args.output_format != "zarr":
    raise ValueError("MajorTom grid is only supported with Zarr output format.")

products = {
        "LST": {
            "short_name": "MOD11A1",
            "variables": ["MODIS_Grid_Daily_1km_LST:LST_Day_1km"]
        },

        "modis_cloud_mask": {
            "short_name": "MOD35_L2",
            "variables": ["Cloud_Mask"],
            "raw_data_type" : "hdf",
            "resolution": 250,
            "patch_size": 256
        },

        "reflectance_250m": {
            "short_name": "MOD09GQ",
            "variables": ["sur_refl_b01",
                           "sur_refl_b02",
                           "QC_250m"
            ],
            "raw_data_type" : "hdf",
            "crs": "EPSG:6933",
            "resolution": 250,
            "patch_size": 256
        },
        "reflectance_500m": {
            "short_name": "MOD09GA",
            "variables": ["sur_refl_b01",
                          "sur_refl_b02",
                          "sur_refl_b03",
                          "state_1km"
            ],
            "raw_data_type" : "hdf",
            "crs": "EPSG:6933",
            "resolution": 500,
            "patch_size": 128
        },
        "NDVI_1km_monthly": {
            "short_name": "MOD13A3",
            "variables": ["NDVI",
                          "EVI",
                          "SummaryQA"
            ],
            "raw_data_type" : "hdf",
            "crs": "EPSG:6933"
        },
        "VIIRS_500m_night_monthly": {
            "short_name": "VNP46A3",
            "variables": ["NearNadir_Composite_Snow_Free",
                          "NearNadir_Composite_Snow_Free_Std",
                          "NearNadir_Composite_Snow_Free_Quality"
            ],
            "raw_data_type" : "h5"
        },

        "VIIRS_500m_night_daily": {
            "short_name": "VNP46A2",
            "variables": ["Gap_Filled_DNB_BRDF_Corrected_NTL",
                          "DNB_BRDF_Corrected_NTL",
                          "QF_Cloud_Mask"
            ],
            "raw_data_type" : "h5"
        }
    }

def mask_env(value, visible=4):
    if not value:
        return "<not set>"
    if len(value) <= visible:
        return "*" * len(value)
    return value[:visible] + "****"

if args.store_cloud:
    print("AWS_ACCESS_KEY_ID:", mask_env(os.getenv("AWS_ACCESS_KEY_ID")))
    print("AWS_SECRET_ACCESS_KEY:", mask_env(os.getenv("AWS_SECRET_ACCESS_KEY")))
    print("AWS_S3_ENDPOINT:", os.getenv("AWS_S3_ENDPOINT") or "<not set>")

if args.majortom_grid and args.output_format != "zarr":
    logger.error("MajorTom grid is only supported with Zarr output format.")
    raise ValueError("MajorTom grid is only supported with Zarr output format.")

if args.majortom_grid:
    logger.info("******* Using MajorTom grid for download. *******")

variables = [v.strip() for v in args.variables_override.split(",")] if args.variables_override else products[args.product]["variables"]
short_name = products[args.product]["short_name"]
raw_data_type = products[args.product]["raw_data_type"]
resolution = products[args.product].get("resolution", None)
patch_size = products[args.product].get("patch_size", 256)

logger.info(f"Starting download for product: {args.product} ({short_name}) with variables: {variables}, patch_size: {patch_size}")

start = args.start_date
end = args.end_date
batch_days = args.batch_days

bboxes = generate_bboxes_fixed(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
         lat_min=args.lat_min,
        lat_max=args.lat_max,
        n_lon=args.n_lon,
        n_lat=args.n_lat,
)

logger.info(f"Generated {len(bboxes)} total tiles for the area of interest.")

if os.path.exists(Path(DATA_PATH) / "water_min.npy"):
    logger.info("Loading precomputed filtered bounding boxes from water_min.npy")
    water_min = np.load(Path(DATA_PATH) / "water_min.npy", allow_pickle=True).tolist()

else:
    water_min = [
        tile
        for tile in tqdm(bboxes, desc="Filtering tiles")
        if tile_has_min_land_fraction(tile, scale=250, min_percentage=30)
    ]
    np.save("water_min.npy", np.array(water_min))

logger.info(f"Selected {len(water_min)} tiles out of {len(bboxes)} total tiles")
plot_boxes_on_map(water_min)

# Loop over each bounding box and download
for i, bbox in tqdm(enumerate(water_min), desc="Processing tiles for download", total=len(water_min)):
    downloader = None
    try:

        downloader = EarthAccessDownloader(
            args=args,
            data_dir="/vsis3/mtg-fci-data/modis" if args.store_cloud else Path(DATA_PATH) / "modis",
            short_name=short_name,
            resolution = resolution,
            bbox= bbox,
            variables=variables,
            date_range=(start, end),
            collection_name=f"{short_name}_061",
            output_format=args.output_format,
            raw_data_type=raw_data_type,
            add_new_variables=args.add_variables,
    )
        if args.delete_temp and downloader.temp_dir.exists():
            logger.warning("Deleting temporary directory as per user request.")
            shutil.rmtree( downloader.temp_dir)

        downloader.run(batch_days=batch_days, majortom_grid = args.majortom_grid, patch_size=patch_size, pixel_size=resolution)

    except Exception as e:
        downloader.cleanup()
        logger.error(e)
        raise e
