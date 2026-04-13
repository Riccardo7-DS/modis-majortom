import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
# from rasterio.features import rasterize
from tqdm.auto import tqdm
from typing import Union
import logging
import itertools
import geopandas as gpd
from pathlib import Path
import itertools
import math
import os 
from urllib.parse import urlparse
from minio import Minio

from ..definitions import DATA_PATH
logger = logging.getLogger(__name__)


def minio_client():
    endpoint = os.getenv("AWS_S3_ENDPOINT")
    if endpoint is None:
        raise ValueError("AWS_S3_ENDPOINT is not set")

    parsed = urlparse(endpoint)

    host = parsed.netloc if parsed.netloc else parsed.path

    return Minio(
        host,
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        secure=(parsed.scheme == "https"),
    )

def extract_object_from_minio(temp_path:Union[str, Path]):
    path = temp_path.lstrip("/").removeprefix("vsis3/")
    _, key = path.split("/", 1)
    key = key.rsplit("/", 1)[0]
    return key 

def setup_minio_config():
    """
    Setup MinIO/S3 configuration from environment variables.
    Expects: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    minio_endpoint = os.getenv('AWS_ENDPOINT_URL')
    minio_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    minio_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not minio_endpoint:
        raise ValueError(
            "MinIO endpoint not configured. Please set MINIO_ENDPOINT environment variable."
        )
    
    # Configure GDAL for S3/MinIO access
    os.environ['AWS_S3_ENDPOINT'] = minio_endpoint
    os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key or ''
    os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key or ''
    os.environ['AWS_VIRTUAL_HOSTING'] = 'FALSE'
    os.environ['AWS_NO_SIGN_REQUEST'] = 'NO'
    
    logger.info(f"MinIO correctly configured")

def meters_to_degrees_lat(meters):
    """Convert meters to degrees latitude."""
    return meters / 111_320.0


def meters_to_degrees_lon(meters, lat):
    """Convert meters to degrees longitude at a given latitude."""
    return meters / (111_320.0 * math.cos(math.radians(lat)))

def generate_bboxes_from_resolution(
    lat_min=-70,
    lat_max=70,
    lon_min=-70,
    lon_max=70,
    spatial_resolution_m=250,  # e.g. MODIS
    n_pixels=7
):
    """
    Generate bounding boxes of n_pixels × n_pixels given a spatial resolution.
    
    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds.
    lon_min, lon_max : float
        Longitude bounds.
    spatial_resolution_m : float
        Pixel size in meters (e.g. 250 for MODIS).
    n_pixels : int
        Number of pixels per side (n × n).
    
    Returns
    -------
    List of bounding boxes: [lon_min, lat_min, lon_max, lat_max]
    """

    # Tile size in meters
    tile_size_m = spatial_resolution_m * n_pixels

    bboxes = []

    lat = lat_min
    while lat < lat_max:
        # Latitude step is constant
        lat_step = meters_to_degrees_lat(tile_size_m)

        lon = lon_min
        while lon < lon_max:
            # Longitude step depends on latitude
            lon_step = meters_to_degrees_lon(tile_size_m, lat)

            bboxes.append([
                float(lon),
                float(lat),
                float(lon + lon_step),
                float(lat + lat_step)
            ])

            lon += lon_step

        lat += lat_step

    return bboxes

def generate_bboxes_fixed(lat_min=-70, lat_max=70, lon_min=-70, lon_max=70, n_lat=7, n_lon=7):
    """
    Divide a lat/lon area into n_lat × n_lon non-overlapping tiles.
    
    Parameters
    ----------
    lat_min, lat_max : float
        Minimum and maximum latitude.
    lon_min, lon_max : float
        Minimum and maximum longitude.
    n_lat : int
        Number of tiles along latitude.
    n_lon : int
        Number of tiles along longitude.
    
    Returns
    -------
    List of bounding boxes: [lon_min, lat_min, lon_max, lat_max]
    """
    # Compute tile size
    lat_step = (lat_max - lat_min) / n_lat
    lon_step = (lon_max - lon_min) / n_lon

    # Generate edges
    lat_edges = [lat_min + i * lat_step for i in range(n_lat)]
    lon_edges = [lon_min + i * lon_step for i in range(n_lon)]

    bboxes = []
    for lat, lon in itertools.product(lat_edges, lon_edges):
        bboxes.append([
            float(lon),
            float(lat),
            float(lon + lon_step),
            float(lat + lat_step)
        ])
    return bboxes


def clip_file(dataset:Union[xr.DataArray, xr.Dataset], 
              gdf=None, 
              invert=False):
    
    from shapely.geometry import mapping, box
    import geopandas as gpd

    epsg_coords = "EPSG:4326"
    dataset = prepare(dataset)
    
    if gdf is not None:
        clipped = dataset.rio.clip(gdf.geometry.apply(mapping), 
                             gdf.crs, 
                             drop=True,
                             invert=invert)
    elif gdf is None:
        logging.info(f"Using the defeault bbox for Latin America"
                     f" to clip the images")
        bbox = latin_box(invert=invert)
        geodf = gpd.GeoDataFrame(
                geometry=[box(bbox[0], bbox[1], bbox[2], bbox[3])],
                crs=epsg_coords)
        
        clipped = dataset.rio.clip(geodf.geometry.values, 
                                       geodf.crs,
                                       drop=True,
                                       invert=invert)
    return clipped

def subsetting_by_names(dataset:Union[xr.DataArray, xr.Dataset], 
                        countries:Union[list, None], 
                        column = "adm_0_name",
                        shapefile="GAUL_2024.zip",
                        invert=False):
    
    from ..definitions import DATA_PATH
    import geopandas as gpd
    import shapely
    if countries is None:
        raise ValueError("You must a list of countries")
    
    

    shapefile_path = Path(DATA_PATH / "shapefiles"/ shapefile)
    gdf = gpd.read_file(shapefile_path)
    if column not in gdf.columns:
        print(f"Available columns in the shapefile: {gdf.columns.tolist()}")
        raise ValueError(f"The shapefile must contain the column {column}")
    subset = gdf[gdf[column].isin(countries)]

    if invert==True:
        subset = subset['geometry'].map(
            lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon))
    return clip_file(dataset, subset)


def countries_to_bbox(country_list:list, geopandas_object:gpd.GeoDataFrame, col_name="adm_0_name"):
    # Load Natural Earth country boundaries (comes bundled with geopandas)
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # Filter for the selected countries
    selected = geopandas_object[geopandas_object[col_name].isin(country_list)]

    if selected.empty:
        raise ValueError("No matching country names found.")

    # Compute bounding box of the union
    minx, miny, maxx, maxy = selected.total_bounds  # (west, south, east, north)

    return [minx, miny, maxx, maxy], selected


def latin_box(invert:bool=False):
    if invert:
        return [-86.308594,-35.317366, -34.277344, 13.111580]
    else:
        return [-35.317366,-86.308594,13.111580,-34.277344]
    
def mtg_box(invert:bool=False):
    if invert:
        return [-70, 70, -70, 70]
    else:
        return [-70, -70, 70, 70]

african_countries = [ 
    'ANGOLA', 'BENIN', 'BURKINA FASO', 'CABO VERDE', 'CAMEROON', 
    'CENTRAL AFRICAN REPUBLIC', 'CHAD', "COTE D'IVOIRE", 'ERITREA', 
    'ETHIOPIA', 'GHANA', 'GUINEA', 'KENYA', 'MALI', 'MAURITANIA', 
    'MAURITIUS', 'MAYOTTE', 'NIGER', 'REUNION', 'SAO TOME AND PRINCIPE',
    'SENEGAL','SEYCHELLES', 'SUDAN', 'TOGO', 'UNITED REPUBLIC OF TANZANIA']

# Your list of countries/territories
other_places = [
    'AFGHANISTAN', 'AMERICAN SAMOA', 'ANGUILLA', 'ANTIGUA AND BARBUDA',
    'ARGENTINA', 'ARUBA', 'BAHAMAS', 'BARBADOS', 'BELIZE', 'BERMUDA',
    'BOLIVIA', 'BONAIRE, SAINT EUSTATIUS AND SABA', 'BRAZIL',
    'CAMBODIA', 'CAYMAN ISLANDS', 'CHILE', 'COLOMBIA', 'COOK ISLANDS',
    'COSTA RICA', 'CUBA', 'CURACAO', 'DOMINICA', 'DOMINICAN REPUBLIC',
    'ECUADOR', 'EL SALVADOR', 'FIJI', 'FRENCH GUIANA',
    'FRENCH POLYNESIA', 'GRENADA', 'GUADELOUPE', 'GUAM', 'GUATEMALA',
    'GUYANA', 'HAITI', 'HONDURAS', 'JAMAICA', 'JAPAN', 'KIRIBATI',
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'MALAYSIA', 'MARSHALL ISLANDS',
    'MARTINIQUE', 'MEXICO', 'MICRONESIA (FEDERATED STATES OF)',
    'MONTSERRAT', 'NAURU', 'NEPAL', 'NEW CALEDONIA', 'NICARAGUA',
    'NIUE', 'NORTHERN MARIANA ISLANDS', 'PAKISTAN', 'PALAU', 'PANAMA',
    'PAPUA NEW GUINEA', 'PARAGUAY', 'PERU', 'PHILIPPINES', 'PITCAIRN',
    'PUERTO RICO', 'SAINT BARTHELEMY', 'SAINT KITTS AND NEVIS',
    'SAINT LUCIA', 'SAINT MARTIN', 'SAINT VINCENT AND THE GRENADINES',
    'SAMOA', 'SAUDI ARABIA', 'SINGAPORE', 'SINT MAARTEN',
    'SOLOMON ISLANDS', 'SRI LANKA', 'SUDAN', 'SURINAME', 'TAIWAN',
    'TOKELAU', 'TONGA', 'TRINIDAD AND TOBAGO',
    'TURKS AND CAICOS ISLANDS', 'TUVALU', 'UNITED STATES OF AMERICA',
    'URUGUAY', 'VANUATU', 'VENEZUELA', 'VIET NAM',
    'VIRGIN ISLANDS (UK)', 'VIRGIN ISLANDS (US)', 'WALLIS AND FUTUNA',
    'YEMEN'
]


# -------------------------
# Function: Download all days for a collection
# -------------------------
def download_collection_tiffs(client, 
                              bucket: str, 
                              collection_path: str, 
                              local_dir: str):
    """
    Download all available TIFF files for a MinIO collection to a local directory.
    
    Args:
        client: MinIO client
        bucket (str): MinIO bucket name
        collection_path (str): path inside the bucket, e.g., "modis/MOD09GQ_061/"
        local_dir (str): local directory to save TIFFs (will be created if it doesn't exist)
    
    Returns:
        dict: summary of downloaded files organized by tile
    """
    import os
    from pathlib import Path
    from tqdm import tqdm
    from utils import get_tiles, get_days_for_tile
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Get all tiles
    tiles = get_tiles(client, bucket, collection_path)
    summary = {}
    
    for tile in tqdm(tiles, desc="Processing tiles"):
        tile_dir = local_path / tile
        tile_dir.mkdir(exist_ok=True)
        
        # Get all days for this tile
        days = get_days_for_tile(client, bucket, collection_path, tile)
        summary[tile] = []
        
        for day in tqdm(days, desc=f"Downloading {tile}", leave=False):
            object_name = f"{collection_path}{tile}/tiffs/{day}"
            local_file = tile_dir / day
            
            try:
                client.fget_object(
                    bucket_name=bucket, 
                    object_name=object_name, 
                    file_path=str(local_file)
                )
                summary[tile].append(day)
            except Exception as e:
                logger.warning(f"Failed to download {object_name}: {e}")
    
    logger.info(f"Downloaded TIFFs to {local_dir}")
    logger.info(f"Summary: {sum(len(days) for days in summary.values())} files across {len(summary)} tiles")
    
    return summary

def prepare(dataset:Union[xr.DataArray, xr.Dataset]):
    if "valid_time" in dataset.dims:
        dataset = dataset.rename({"valid_time":"time"})
    if "longitude" in dataset.dims:
        dataset = dataset.rename({"latitude":"lat", "longitude":"lon"})
    if "x" in dataset.dims:
        dataset = dataset.rename({"y":"lat", "x":"lon"})
    if "X" in dataset.dims:
        dataset = dataset.rename({"Y":"lat", "X":"lon"})
    dataset = dataset.rio.write_crs("EPSG:4326")
    dataset = dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    return dataset



def build_timeseries(da, n_weeks=12):

    """
    Convert xarray DataArray (time x region) into dataset [S, T, F].
    Drops windows where all values are NaN.
    """
    values = da.values  # shape (time, region)
    n_time, n_regions = values.shape
    
    sequences = []
    for r in tqdm(range(n_regions)):
        series = values[:, r]  # time series for region r
        for t in range(n_time - n_weeks + 1):
            window = series[t : t + n_weeks]
            if np.isnan(window).all():   # 👈 skip windows with only NaNs
                continue
            sequences.append(window)
    
    # Convert to numpy array: [S, T, F]
    sequences = np.array(sequences)
    sequences = sequences[..., np.newaxis]  # add feature dimension
    return sequences

def process_gdf(zip_path, countries:list=None):
    # Read the shapefile directly from the zip
    gdf = gpd.read_file(f"zip://{zip_path}")

    # Optional: standardize column names to match your table
    gdf = gdf.rename(columns={
        "gaul0_name": "adm_0_name",
        "gaul1_name": "adm_1_name",
        "gaul2_name": "adm_2_name",
        "GAUL_CODE": "GAUL_code_2"
    })

    gdf["region_id"] = gdf["adm_0_name"].str.title() + "_" + gdf["adm_2_name"].str.title()

    if countries is not None:
        gdf = gdf[gdf["adm_0_name"].str.title().isin(countries)]
        return gdf.reset_index(drop=True)
    else:
        return gdf

def convert_data_to_yearly_presence(data:pd.DataFrame, filter_year:int=2000, spatial_resolution:str="Admin2"):
    
    data["start_dt"] = pd.to_datetime(data["calendar_start_date"])
    data["year"] = data["start_dt"].dt.year

    data = data.loc[data["S_res"]==spatial_resolution]
    data = data.loc[data["year"] >= filter_year]  # <- filter years if needed

    # Build presence/absence table
    presence = (
        data.groupby(["adm_0_name", "year"])
        .size()                # count rows
        .reset_index(name="count")
    )

    presence["exists"] = (presence["count"] > 0).astype(int)

    # Get full year range
    all_years = pd.Series(range(data["year"].min(), data["year"].max() + 1))

    # Pivot
    heatmap_data = (
        presence.pivot(index="adm_0_name", columns="year", values="exists")
        .reindex(columns=all_years, fill_value=0)  # <- add missing years
        .fillna(0)
        .T
    )

    names = data["adm_0_name"].unique()
    
    return heatmap_data, names


def init_logging(log_file=None, verbose=False):
    import os
    # Determine the logging level
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Define the logging format
    formatter = "%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)s()] : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # Setup basic configuration for logging
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(log_file, "w"),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=formatter,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler()
            ]
        )

    logger = logging.getLogger()
    return logger
