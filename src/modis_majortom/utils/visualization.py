
# -------------------------
# Function: List tiles
# -------------------------

import math
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
from rasterio.enums import Resampling
import logging

logger = logging.getLogger(__name__)

def get_tiles(client, bucket, collection_path: str):
    """
    List all tiles in a MinIO collection.
    
    Args:
        collection_path (str): path inside the bucket, e.g., "modis/MOD09GQ_061/"
    
    Returns:
        List[str]: tile names
    """
    objects = client.list_objects(bucket_name=bucket, prefix=collection_path, recursive=False)
    tiles = [str(obj.object_name).split("/")[2] for obj in objects if obj.is_dir]
    return tiles


# -------------------------
# Function: List available days for a tile
# -------------------------
def get_days_for_tile(client, bucket, collection_path: str, tile: str):
    """
    List all available days (TIFF filenames) for a given tile.
    
    Args:
        collection_path (str): path inside the bucket, e.g., "modis/MOD09GQ_061/"
        tile (str): tile name
    
    Returns:
        List[str]: day filenames
    """
    prefix = f"{collection_path}{tile}/tiffs/"
    objects = client.list_objects(bucket_name=bucket, prefix=prefix, recursive=True)
    days = [str(obj.object_name).split("/")[4] for obj in objects]
    return days


# -------------------------
# Function: Download and plot a tile for a given day
# -------------------------
def plot_tile_day(client, bucket, collection_path: str, tile: str, day: str, cmap="viridis"):
    """
    Download a TIFF for a given tile/day and plot the first band.
    
    Args:
        collection_path (str): path inside the bucket
        tile (str): tile name
        day (str): TIFF filename
        cmap (str): colormap for plotting
    """
    object_name = f"{collection_path}{tile}/tiffs/{day}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, day)
        client.fget_object(bucket_name=bucket, object_name=object_name, file_path=local_path)
        
        with rasterio.open(local_path) as src:
            band1 = src.read(1)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(band1, cmap=cmap)
        plt.colorbar()
        plt.title(f"{tile} - {day}")
        plt.show()




def plot_day_across_tiles(
    client,
    bucket,
    collection_path: str,
    day: str,
    cmap="RdYlGn",
    coarsen_factor: int = 1,
):
    """
    Plot NDVI for the same day across all tiles in a MinIO collection.

    NDVI = (NIR - RED) / (NIR + RED)
    Assumes:
      - band 1 = RED
      - band 2 = NIR (MOD09GQ convention)
    """

    # --- Discover tiles ---
    objects = client.list_objects(
        bucket_name=bucket,
        prefix=collection_path,
        recursive=False,
    )
    tiles = [str(obj.object_name).split("/")[2] for obj in objects if obj.is_dir]

    n_tiles = len(tiles)
    if n_tiles == 0:
        print("No tiles found.")
        return

    # --- Layout ---
    n_cols = min(5, n_tiles)
    n_rows = math.ceil(n_tiles / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()

    # --- Loop with progress bar ---
    for i, tile in enumerate(tqdm(tiles, desc=f"NDVI {day}", unit="tile")):
        object_name = f"{collection_path}{tile}/tiffs/{day}"
        ax = axes[i]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, day)
                client.fget_object(bucket, object_name, local_path)

                with rasterio.open(local_path) as src:
                    if coarsen_factor > 1:
                        red = src.read(
                            1,
                            out_shape=(
                                src.height // coarsen_factor,
                                src.width // coarsen_factor,
                            ),
                            resampling=Resampling.average,
                        ).astype("float32")

                        nir = src.read(
                            2,
                            out_shape=(
                                src.height // coarsen_factor,
                                src.width // coarsen_factor,
                            ),
                            resampling=Resampling.average,
                        ).astype("float32")
                    else:
                        red = src.read(1).astype("float32")
                        nir = src.read(2).astype("float32")

                    bounds = src.bounds

                # --- NDVI computation (safe) ---
                np.seterr(divide="ignore", invalid="ignore")
                ndvi = (nir - red) / (nir + red)
                ndvi = np.clip(ndvi, -1, 1)

                im = ax.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)

                ax.set_title(
                    f"{tile}\n"
                    f"{bounds.left:.2f}, {bounds.bottom:.2f}, "
                    f"{bounds.right:.2f}, {bounds.top:.2f}",
                    fontsize=9,
                )
                ax.axis("off")

        except Exception as e:
            ax.set_title(f"{tile}\n❌ failed")
            ax.axis("off")
            print(f"[WARN] {tile}: {e}")

    # --- Hide unused axes ---
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # --- Shared NDVI colorbar ---
    cbar = fig.colorbar(
        im,
        ax=axes.tolist(),
        shrink=0.6,
        label="NDVI",
    )

    # --- Spacing ---
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.05,
        wspace=0.25,
        hspace=0.35,
    )

    fig.suptitle(
        f"{day} – NDVI (coarsen ×{coarsen_factor})",
        fontsize=16,
    )

    plt.show()


def inspect_raster_resolution(tif_path: str):
    """
    Inspect spatial resolution and size of a GeoTIFF.

    Args:
        tif_path (str): Path to local GeoTIFF

    Returns:
        dict with resolution and metadata
    """
    with rasterio.open(tif_path) as src:
        res_x, res_y = src.res          # pixel size in map units
        width, height = src.width, src.height
        crs = src.crs
        bounds = src.bounds
        transform = src.transform

    info = {
        "pixel_size_x": res_x,
        "pixel_size_y": res_y,
        "width_px": width,
        "height_px": height,
        "crs": crs.to_string() if crs else None,
        "bounds": bounds,
        "transform": transform,
    }

    return info


