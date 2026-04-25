# import ee
import logging
from pathlib import Path
from osgeo import gdal
from pyhdf.SD import SD, SDC
from ..definitions import DATA_PATH
import tempfile
from collections import defaultdict
import gc
import rasterio
import os
import re
import glob
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm
import earthaccess
from dotenv import load_dotenv
import numpy as np
import xarray as xr
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..utils import prepare, minio_client, setup_minio_config, extract_object_from_minio
from pyproj import Transformer
import ee, geemap
from datetime import datetime, timedelta
from pystac_client import Client
import pyproj
from majortom import Grid
import zarr
import json
import re
import time
import errno
from contextlib import contextmanager
from pyresample import geometry
from pyresample.kd_tree import resample_nearest
import requests


REFL_BANDS = {
    "sur_refl_b01", "sur_refl_b02", "sur_refl_b03",
    "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"}

QC_BANDS = {"qc_500m", "qc_250m"}

CLOUD_BANDS = {"state_1km", "cloud_mask"}

logger = logging.getLogger(__name__)

class EeModis():
    def __init__(self,
                 start_date:str,
                 end_date:str,
                 name:Literal["ref_250m_061","NDVI_1km_monthly_061", "LST_061"],
                 output_dir:str,
                 geometry=None,
                 output_resolution:int=1000,
                 crs='EPSG:4326',
                 format:Literal["GeoTIFF","COG"]="GeoTIFF"):


        ee.Authenticate()
        ee.Initialize(project=os.environ["EE_PROJECT"])

        valid_names = {"ref_250m_061","LST_061", "NDVI_1km_monthly_061"}
        assert name in valid_names, \
            "Invalid value for 'name'. It should be one of: 'ref_250m_061', 'LST_061', 'NDVI_1km_monthly_061'"

        ee.Initialize(project=os.environ.get("EE_PROJECT"))

        self.start_date = start_date
        self.end_date = end_date
        self.out_dir = output_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.polygon = ee.Geometry.Rectangle(geometry) if type(geometry) is list else geometry
        self.output_resolution = None if output_resolution is None else output_resolution
        self.product = self._get_product_name(name)
        self.name = name
        self._format = format
        self._crs = crs


    def run(self,
            download_collection:bool=False,
            compute_ndvi:bool=False):

        """
        download_collection: if True, download the entire image collection,
                                otherwise download each image separately.
        """
        import ee

        # Import images
        images = ee.ImageCollection(self.product)\
                    .filterDate(self.start_date, self.end_date)\
                    .filterBounds(self.polygon)

        self.bands = self._get_bands(self.product)
        img_bands = images.select(self.bands)

        img_bands = self._preprocess(img_bands, compute_ndvi=compute_ndvi)

        if download_collection:
            self._collection_prepr_download(img_bands)
        else:
            self._export_images_to_local(img_bands,
                self.out_dir,
                scale=self.output_resolution)

    def _preprocess(self, img_bands, compute_ndvi:bool=False):

        if self.name == "ref_250m_061" and compute_ndvi:
            img_bands = img_bands.map(lambda x: self._compute_ndvi(x, self.bands[1], self.bands[0],
                                                                   self.bands[2]))
        if self.name == "LST_061":
            img_bands = img_bands.map(self._scale_lst)

        images_preprocessed = img_bands.map(lambda image: image.clip(self.polygon))

        if self.output_resolution is not None:
            images_preprocessed = images_preprocessed.map(self._imreproj)
        #     images_preprocessed.aggregate_array("system:index").getInfo()

        return images_preprocessed



    # -------- LST helpers -------- #
    def _scale_lst(self, img):
        """Convert LST from MOD11A1 to Celsius and mask by QA."""
        lst_day = img.select("LST_Day_1km").multiply(0.02).subtract(273.15)
        lst_night = img.select("LST_Night_1km").multiply(0.02).subtract(273.15)
        qa_day = img.select("QC_Day")
        qa_night = img.select("QC_Night")

        # simple QA mask (only keep qa==0)
        mask_day = qa_day.eq(0)
        mask_night = qa_night.eq(0)

        lst_day = lst_day.updateMask(mask_day).rename("LST_Day_C")
        lst_night = lst_night.updateMask(mask_night).rename("LST_Night_C")

        return img.addBands([lst_day, lst_night])

    def _get_product_name(self, name):
        if name == "ref_061":
            return "MODIS/061/MOD09GQ"
        elif name == "LST_061":
            return "MODIS/061/MOD11A1"
        else:
            raise NotImplementedError(f"Product {name} not implemented")

    def _get_bands(self, product_name):
        if product_name == "MODIS/061/MOD09GQ":
            return ["sur_refl_b01", "sur_refl_b02","QC_250m"]
        elif product_name == "MODIS/061/MOD11A1":
            return ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night"]
        else:
            raise NotImplementedError(f"Product {product_name} not implemented")

    def _extract_qa(self, img, qaBand):
        qa_mask = img.select(qaBand)
        quality_mask = self._bitwise_extract(qa_mask, 0, 1).eq(0)
        return quality_mask

    def _bitwise_extract(self, input_image, from_bit, to_bit):
        import ee
        mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
        mask = ee.Number(1).leftShift(mask_size).subtract(1)
        return input_image.rightShift(from_bit).bitwiseAnd(mask)

    def _compute_ndvi(self, img, nirBand, redBand, qaBand=None):
        ndvi = img.normalizedDifference([nirBand, redBand])
        if qaBand is not None:
            qa = self._extract_qa(img, qaBand)
            return ndvi.updateMask(qa)
        else:
            return ndvi

    def _imreproj(self, image):
        # Example for 500m MODIS pixel
        # meters_per_degree = 111_319  # approximate at equator
        # scale_deg = resolution / meters_per_degree  # ~0.0045°
        return image.resample("bilinear")

    def ee_hoa_geometry(self):
        logger.info("Using default geometry for horn of Africa")

        polygon = ee.Geometry.Polygon(
            [[[30.288233396779802,-5.949173816626356],
            [51.9972177717798,-5.949173816626356],
            [51.9972177717798,15.808293611760663],
            [30.288233396779802,15.808293611760663]]]
        )
        return  ee.Geometry(polygon, None, False)

    def _collection_prepr_download(self, images):
        geemap.ee_export_image_collection(images, self.out_dir, self._format, crs=self._crs)


    """Export each MODIS image locally as a Cloud Optimized GeoTIFF."""
    def _export_images_to_local(self, images, output_dir, scale=None):

        image_list = images.toList(images.size())
        n = images.size().getInfo()

        for i in range(n):
            image = ee.Image(image_list.get(i))
            image_id = image.get("system:index").getInfo()
            filename = Path(output_dir) / f"{image_id}.tif"

            logger.debug(f"Exporting {image_id} → {filename}")
            geemap.ee_export_image(
                image,
                filename=str(filename),
                scale=scale,
                region=image.geometry(),
                file_per_band=False,
                format=self._format,
            )

    def _split_roi(self, roi, nx=2, ny=2):
        """Split an ee.Geometry.Rectangle into nx*ny tiles."""
        import ee
        coords = roi.coordinates().get(0).getInfo()
        xmin, ymin = coords[0][0], coords[0][1]
        xmax, ymax = coords[2][0], coords[2][1]
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        rois = []
        for i in range(nx):
            for j in range(ny):
                x0, x1 = xmin + i*dx, xmin + (i+1)*dx
                y0, y1 = ymin + j*dy, ymin + (j+1)*dy
                rois.append(ee.Geometry.Rectangle([x0, y0, x1, y1]))
        return rois

    def _image_prepr_download(self, images, nx=2, ny=2):
        """Export each image in the collection, automatically splitting large ROIs."""
        import ee
        import logging
        logger = logging.getLogger(__name__)

        # Split ROI
        tiles = self._split_roi(self.polygon, nx=nx, ny=ny)

        # Get image IDs
        image_ids = images.aggregate_array("system:index").getInfo()

        for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            img = ee.Image(image_id)

            for t_idx, tile in enumerate(tiles):
                img_tile = img.clip(tile)
                proj = img_tile.projection().getInfo()
                filename = f"{image_id.split('/')[-1]}_tile{t_idx}"

                task = ee.batch.Export.image.toDrive(
                    image=img_tile,
                    description=filename,
                    folder="MODIS_EXPORT",
                    region=tile.bounds(),
                    crs=proj["crs"],
                    crsTransform=proj["transform"],
                    maxPixels=1e13,
                    fileFormat="GeoTIFF"
                )
                task.start()
                logger.info(f"Exporting {filename}...")

    def _preprocess_file(self, ds:xr.Dataset):
        import pandas as pd
        time = ds.encoding['source'].split("/")[-1].replace("_","/")[:-4]
        date_xr = pd.to_datetime(time)
        ds = ds.assign_coords(time=date_xr)
        ds = ds.expand_dims(dim="time")
        return ds.isel(band=0)

    def xarray_preprocess(self):
        import xarray as xr
        import os
        files = [os.path.join(self.out_dir,f) for f in os.listdir(self.out_dir) if f.endswith(".tif")]
        dataset = xr.open_mfdataset(files,
                                    preprocess=self._preprocess_file,
                                    engine="rasterio")
        return dataset


class EarthAccessDownloader:
    """
    Generic downloader for NASA Earthdata (via earthaccess).
    Handles: authentication, search, download, mosaic, cleanup, and export to Zarr.
    """

    def __init__(
        self,
        args,
        short_name="MOD11A1",
        variables="MODIS_Grid_Daily_1km_LST:LST_Day_1km",
        resolution=None,
        date_range=None,
        bbox=None,
        data_dir=None,
        collection_name="lst_061",
        crs="EPSG:6933",
        output_format:Literal["tiff","zarr"]="zarr",
        raw_data_type=".hdf",
        add_new_variables: bool = False,
    ):

        if resolution is None:
            raise ValueError("resolution must be provided.")
        else:
            self._resolution = resolution

        self._add_new_variables = add_new_variables

        self.date_range = self._check_dates(date_range)

        self.short_name = short_name
        self.variable = variables
        self.bbox = bbox
        self.collection_name = collection_name

        self._check_cloud_or_local(data_dir, args.store_cloud)

        self._crs = crs
        self._reproj_lib = self._check_reproj_lib(args.reproj_lib)
        self._reproj_method = self._check_reproj_method(args.reproj_method)

        self.raw_data_type = raw_data_type
        self._output_format = output_format

        self.temp_dir = Path(tempfile.mkdtemp())

        # Initialize MinIO configuration if cloud storage is enabled
        if self._store_cloud and os.getenv('AWS_ACCESS_KEY_ID') is None:
            setup_minio_config()

        self._login_earthaccess()

        if "state_1km" in (self.variable or []):
            logger.info(
                f"[{self.short_name}] 'state_1km' will be stored as raw uint16 QA bits. "
                f"Decode at load time with MODISQCMask (e.g. state_1km_cloud_mask)."
            )

    def _check_cloud_or_local(self, data_dir, store_cloud, zarr_path=None):
        """
        Set up data directories for local or cloud storage.

        Parameters
        ----------
        data_dir : str | Path | None
            Local path or /vsis3/ path for cloud storage.s
        store_cloud : bool
            True for cloud storage, False for local filesystem.
        """
        self._store_cloud = store_cloud

        # Set data_dir
        if store_cloud:
            if not data_dir:
                raise ValueError("For cloud storage, data_dir must be provided as a /vsis3/ path.")
            self.data_dir = str(data_dir)  # keep as string for GDAL
        else:
            self.data_dir = Path(data_dir) if data_dir else Path(DATA_PATH) / "modis"

        logger.info(f"Data directory set to: {self.data_dir} with cloud option {self._store_cloud}")

        # Build granule_dir
        collection_path = (
            f"{self.data_dir}/{self.collection_name}" if store_cloud else self.data_dir / self.collection_name
        )

        if self.bbox:
            bbox_str = "_".join(format(x, ".1f") for x in self.bbox)
            collection_path = f"{collection_path}/{bbox_str}" if store_cloud else collection_path / bbox_str

        self.granule_dir = f"{collection_path}/raw_data" if store_cloud else collection_path / "raw_data"

        if not store_cloud:
            self.granule_dir.mkdir(parents=True, exist_ok=True)

        if zarr_path and not store_cloud:
            self.zarr_path = zarr_path
        elif not store_cloud:
            self.zarr_path = self.data_dir / f"{self.short_name}_dataset.zarr"
        elif store_cloud and zarr_path:
            raise NotImplementedError("Zarr output not implemented for cloud storage.")

        self._minio_client = minio_client() if store_cloud else None

        if store_cloud:
            self.minio_bucket = os.getenv('MINIO_BUCKET')
            self.minio_prefix = os.getenv('MINIO_PREFIX', 'modis')  # default if not set

    def _zarr_index_path(self):
        """Return Path to a small JSON index storing processed dates (local only)."""
        if self._store_cloud:
            return None
        try:
            return self.zarr_path.with_name(f"{self.zarr_path.name}.index.json")
        except Exception:
            return None

    def _load_zarr_index(self):
        p = self._zarr_index_path()
        if not p:
            return set()
        try:
            if p.exists():
                with open(p, "r") as fh:
                    data = json.load(fh)
                # Handle both old (simple dict) and new (dict with metadata) formats
                if isinstance(data, dict):
                    return set(data.get("dates", []))
                elif isinstance(data, list):
                    # Backward compat: old format was just dates list
                    return set(data)
        except Exception:
            logger.debug(f"Failed to read zarr index {p}")
        return set()

    def _save_zarr_index(self, dates_set):
        p = self._zarr_index_path()
        if not p:
            return
        tmp = p.with_suffix(".tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            # Include bbox and other metadata in index
            index_data = {
                "dates": sorted(list(dates_set)),
                "bbox": list(self.bbox) if hasattr(self, "bbox") and self.bbox else None,
                "collection": self.collection_name,
                "product": self.short_name,
            }
            with open(tmp, "w") as fh:
                json.dump(index_data, fh)
            tmp.replace(p)
        except Exception as e:
            logger.debug(f"Failed to write zarr index {p}: {e}")

    def _update_zarr_index(self, dates_set):
        """Merge new dates into existing index and persist."""
        if self._store_cloud:
            return
        existing = self._load_zarr_index()
        merged = set(existing) | set(dates_set)
        self._save_zarr_index(merged)

    def _check_disk_space(self, min_gb=1, path=None):
        """Return True if free disk space >= min_gb on the filesystem containing `path`.
        If `path` is None, use DATA_PATH.
        """
        try:
            target = Path(path) if path else Path(DATA_PATH)
            stat = shutil.disk_usage(str(target))
            free_gb = stat.free / (1024 ** 3)
            return free_gb >= float(min_gb)
        except Exception:
            return True

    def _check_reproj_lib(self, reproj_lib):
        valid_libs = ["rioxarray", "xesmf"]
        if reproj_lib not in valid_libs:
            raise ValueError(f"Invalid reproj_lib '{reproj_lib}'. Must be one of {valid_libs}.")
        logger.info(f"Using reproj_lib: {reproj_lib}")
        return reproj_lib

    def _check_reproj_method(self, reproj_method):
        valid_methods = ["nearest", "bilinear"]
        if reproj_method not in valid_methods:
            raise ValueError(f"Invalid reproj_method '{reproj_method}'. Must be one of {valid_methods}.")
        logger.info(f"Using reproj_method: {reproj_method} with {self._crs}")
        return reproj_method

    def _rio_resampling(self):
        """
        Convert CLI/string resampling method to rasterio Resampling enum.
        rioxarray expects rasterio.enums.Resampling, not a plain string.
        """
        from rasterio.enums import Resampling

        resampling_dict = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
        }

        try:
            return resampling_dict[self._reproj_method]
        except KeyError:
            raise ValueError(
                f"Invalid reproj_method '{self._reproj_method}'. "
                f"Must be one of {list(resampling_dict.keys())}."
            )

    def _get_utm_crs(self, bbox=None):
        """
        Compute UTM CRS from a bounding box in EPSG:4326.

        Parameters
        ----------
        bbox : tuple or list
            (minx, miny, maxx, maxy) in EPSG:4326

        Returns
        -------
        str
            EPSG code string for the UTM zone.
        """

        # If bbox not provided, fallback to stored bbox
        if bbox is None:
            bbox = self.bbox

        if not bbox:
            return "EPSG:6933"  # fallback

        minx, miny, maxx, maxy = bbox

        # Center of the bounding box
        lon = (minx + maxx) / 2
        lat = (miny + maxy) / 2

        # Compute UTM zone
        zone = int((lon + 180) / 6) + 1

        # Determine hemisphere
        if lat >= 0:
            epsg = 32600 + zone  # Northern hemisphere
        else:
            epsg = 32700 + zone  # Southern hemisphere

        return f"EPSG:{epsg}"

    def _login_earthaccess(self):
        load_dotenv()
        # Login once
        earthaccess.login(strategy="environment")

    def _check_dates(self, date_range):
        if date_range is None or len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start_date, end_date)")
        return date_range


    def _missing_date_ranges(self, start_date, end_date, processed_dates):
        """
        Returns a list of (start, end) date tuples (YYYY-MM-DD)
        representing contiguous missing date intervals.
        """

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        processed = set(
            datetime.strptime(d, "%Y-%m-%d") for d in processed_dates
        )

        all_dates = []
        d = start
        while d <= end:
            if d not in processed:
                all_dates.append(d)
            d += timedelta(days=1)

        if not all_dates:
            return []

        ranges = []
        range_start = all_dates[0]
        prev = all_dates[0]

        for d in all_dates[1:]:
            if d == prev + timedelta(days=1):
                prev = d
            else:
                ranges.append((range_start, prev))
                range_start = d
                prev = d

        ranges.append((range_start, prev))

        return [
            (r[0].strftime("%Y-%m-%d"), r[1].strftime("%Y-%m-%d"))
            for r in ranges
        ]

    # ------------------------
    # Search and Download
    # ------------------------
    def _search_and_download(self, date_range=None):
        """
        Search and download data for missing dates within a given date range.
        """

        if date_range is None:
            date_range = self.date_range

        start_date, end_date = date_range

        # Build list of all dates in the requested range (YYYY-MM-DD)
        d = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        all_dates = []
        while d <= end_dt:
            all_dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)

        if self._add_new_variables:
            # Only skip dates where ALL requested variables are already written in the zarr.
            # This lets re-runs continue from where they left off instead of re-downloading
            # everything from scratch, while still processing dates that the new variables
            # haven't been written for yet.
            processed_dates = self._get_processed_dates_for_variables(dates_to_check=all_dates)
        else:
            # Ask the fast path to probe only these dates (MajorTOM stores are fast this way)
            processed_dates = self._get_processed_dates(dates_to_check=all_dates)

        done_dates_range = [d for d in all_dates if d in processed_dates]
        if len(done_dates_range) > 0:
            logger.info(f"Skipping {len(done_dates_range)} already processed dates.")

        missing_ranges = self._missing_date_ranges(
            start_date, end_date, processed_dates
        )

        if not missing_ranges:
            logger.info("✅ No missing dates to download.")
            self.files = None
            return

        for sub_range in missing_ranges:
            logger.info(
                f"🔍 Searching for {self.short_name} data from {sub_range[0]} to {sub_range[1]}..."
            )

            results = earthaccess.search_data(
                short_name=self.short_name,
                bounding_box=(
                    self.bbox[0], self.bbox[1],
                    self.bbox[2], self.bbox[3]
                ),
                temporal=sub_range,
            )

            logger.info(f"⬇️ Downloading {len(results)} files to {self.granule_dir}")

            if results:
                if self._store_cloud:
                    earthaccess.download(results, self.temp_dir)
                else:
                    earthaccess.download(results, str(self.granule_dir))

        # After download, refresh file list and filter by date range
        if not self._store_cloud:
            all_files = sorted(glob.glob(str(self.granule_dir / f"{self.short_name}*.{self.raw_data_type}")))
        else:
            all_files = sorted(glob.glob(str(self.temp_dir / f"{self.short_name}*.{self.raw_data_type}")))

        # Filter files to only keep those within the requested date range
        start_date, end_date = self.date_range if date_range is None else date_range
        range_start = datetime.strptime(start_date, "%Y-%m-%d")
        range_end = datetime.strptime(end_date, "%Y-%m-%d")

        self.files = []
        for f in all_files:
            file_date = self._get_date(f)
            if range_start <= file_date <= range_end:
                self.files.append(f)


    def _custom_preprocess(self, da, band_name=None):

        if band_name is None:
            band_name = da.name

        # ---------- MODIS LST ----------
        if self.short_name == "MOD11A1":
            da = da.where(da > 0).astype("float32") * 0.02 - 273.15
            da.name = "LST_Day_Celsius"

        # ---------- MODIS NDVI ----------
        elif "MOD13" in self.short_name:
            da = da.where(da != -3000).astype("float32") * 0.0001
            da.name = "NDVI"

        # ---------- VIIRS NIGHT LIGHTS ----------
        elif "VNP46A" in self.short_name:

            if band_name in ["NearNadir_Composite_Snow_Free", "NearNadir_Composite_Snow_Free_Std"]:
                da = da.where(da != 65535)
                da = da.astype("float32")

                # attrs are already stripped — apply known VIIRS scale factor directly
                # VNP46A2 radiance scale is 0.1, offset 0
                da = da * 0.1

                da = da.rio.write_nodata(np.nan)
                da.name = "Radiance" if "Std" not in band_name else "Radiance_Std"

            elif band_name in ["NearNadir_Composite_Snow_Free_Quality", "NearNadir_Composite_Snow_Free_Num"]:
                # Categorical band → keep integer
                da = da.where(da != 65535, 0)
                da = da.astype("uint8")
                da = da.rio.write_nodata(0)
                da.name = "Quality_Flag"

        # ---------- DEFAULT ----------
        else:
            da = da.astype("float32")
            da = da.rio.write_nodata(np.nan)
            da.name = "band_data"

        return da

    def _get_date(self, filename):
        match = re.search(r'\.A(\d{7})\.', filename)
        return pd.to_datetime(match.group(1), format="%Y%j")

    def _get_hv(self, filename):
        match = re.search(r'\.h(\d+)v(\d+)\.', filename)
        return int(match.group(1)), int(match.group(2))

    def _validate_latlon_bounds(self, lat, lon, tol=1e-6):
        """
        Validate that the lat/lon arrays define a meaningful bounding box.

        Parameters
        ----------
        lat : np.ndarray
            2D array of latitude values.
        lon : np.ndarray
            2D array of longitude values.
        tol : float
            Minimum required span; values smaller are treated as collapsed.

        Raises
        ------
        ValueError
            If the lat/lon bounding box is degenerate (collapsed to a line/point).
        """

        min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
        min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)

        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon

        errors = []

        if lat_span < tol:
            errors.append(
                f"Latitude range is too small: span={lat_span:.6f}. "
                "This suggests a collapsed latitude dimension or swapped axes."
            )

        if lon_span < tol:
            errors.append(
                f"Longitude range is too small: span={lon_span:.6f}. "
                "This suggests a collapsed longitude dimension or swapped axes."
            )

        if errors:
            raise ValueError("\n".join(errors))

        return True

    def _regridding_with_xe_modis(self,
        ds_src: xr.Dataset,
        method: str = "bilinear",
        lon_res: float | None = None,
        lat_res: float | None = None,
        ):

        import xesmf as xe
        """
        Regrid a Dataset from EPSG:6933 to EPSG:4326 using xESMF.
        If lon_res / lat_res are None, infer from rio.resolution().
            """
        # Ensure CRS metadata
        if hasattr(ds_src, "rio"):
            ds_src = ds_src.rio.write_crs(self._crs)

        # ---- Create 2D lon/lat coordinates for xESMF ----
        if "x" in ds_src.coords and "y" in ds_src.coords:
            # Projected coordinates
            x = ds_src.coords["x"].values
            y = ds_src.coords["y"].values

            # Clip x/y to valid EPSG:6933 range to avoid lat warning
            x_min, x_max = -3.5e7, 3.5e7
            y_min, y_max = -3.5e7, 3.5e7
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)

            # Build 2D meshgrid (shape y,x)
            X, Y = np.meshgrid(x, y, indexing="xy")

            # Transform to geographic coordinates
            transformer = Transformer.from_crs("EPSG:6933", self._crs, always_xy=True)
            lon2d, lat2d = transformer.transform(X, Y)

            # Optional safety clip
            lon2d = np.clip(lon2d, -180, 180)
            lat2d = np.clip(lat2d, -90, 90)

            # Assign CF-compliant coordinates
            ds_src = ds_src.assign_coords(
                lon=(("y", "x"), lon2d),
                lat=(("y", "x"), lat2d)
            )

        # Infer resolution if not provided
        if lon_res is None or lat_res is None:
            res_x, res_y = ds_src.rio.resolution()
            lon_res = abs(lon_res or res_x)
            lat_res = abs(lat_res or res_y)

        # Define target lat/lon grid based on source bounds
        bounds = ds_src.rio.bounds()
        lon_target = np.arange(bounds[0], bounds[2] + lon_res, lon_res)
        lat_target = np.arange(bounds[1], bounds[3] + lat_res, lat_res)

        ds_out = xr.Dataset(
            coords={
                "lon": (["lon"], lon_target),
                "lat": (["lat"], lat_target)
            }
        )

        # Regrid using xESMF
        regridder = xe.Regridder(ds_src, ds_out, method=method, reuse_weights=False)
        ds_regridded = regridder(ds_src)

        # ---- Mask: keep only coordinates with non-null data ----
        src_mask = xr.where(ds_src.isnull(), 0, 1).mean(dim=list(ds_src.data_vars)).fillna(0)
        mask_regridded = regridder(src_mask)
        ds_regridded = ds_regridded.where(mask_regridded > 0)

        # Attach CRS metadata
        ds_regridded.attrs["crs"] = self._crs

        return ds_regridded

    def _get_processed_dates_for_variables(self, dates_to_check=None) -> set:
        """
        Like _get_processed_dates but scoped to the variables currently being downloaded.
        A date is considered processed only if ALL of self.variable are already present
        in the zarr for that date.  Used in add_new_variables mode so that re-runs skip
        dates that are fully written while still processing dates that are missing any
        of the new variables.
        """
        variables = [self.variable] if isinstance(self.variable, str) else list(self.variable)
        processed_dates = set()

        try:
            zarr_path = self.zarr_path
            # use_consolidated=False: zarr v3 auto-uses .zmetadata by default, which can
            # contain stale entries for variables whose chunk files were never written
            # (e.g. after a partial/failed run).  Read the real filesystem state instead.
            store = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)

            if "patches" in store:
                patches_grp = store["patches"]
                # Collect, per variable, which of the requested dates are present
                per_var: list[set] = []
                for var in variables:
                    if var in patches_grp:
                        var_grp = patches_grp[var]
                        if dates_to_check:
                            per_var.append({d for d in dates_to_check if d in var_grp})
                        else:
                            per_var.append(set(var_grp.group_keys()))
                    else:
                        per_var.append(set())   # variable not yet in store → nothing processed

                if per_var:
                    processed_dates = set.intersection(*per_var)
            else:
                # Regular time-indexed zarr: variable must exist with those time steps
                ds = xr.open_zarr(str(zarr_path))
                if "time" in ds.dims:
                    existing_dates = set(pd.to_datetime(ds.time.values).strftime("%Y-%m-%d"))
                    missing_vars = [v for v in variables if v not in ds.data_vars]
                    if missing_vars:
                        # At least one variable is entirely absent → nothing processed
                        processed_dates = set()
                    else:
                        if dates_to_check:
                            processed_dates = {d for d in dates_to_check if d in existing_dates}
                        else:
                            processed_dates = existing_dates
        except Exception:
            pass  # zarr doesn't exist or can't be read → treat everything as unprocessed

        return processed_dates

    def _get_processed_dates(self, dates_to_check=None):
        """
        Get processed dates. If `dates_to_check` is provided (iterable of YYYY-MM-DD strings),
        probe only those dates in the Zarr store (fast for MajorTOM sparse stores).
        Falls back to scanning available dates and persists a small JSON index for local stores.
        """
        processed_dates = set()

        if self._output_format.lower() == "zarr":
            zarr_path = self.zarr_path
            if self._store_cloud:
                zarr_path = f"/vsis3/{self.minio_bucket}/{self.minio_prefix}/{self.collection_name}_dataset.zarr"

            try:
                # Use index if up-to-date
                index_path = self._zarr_index_path()
                if index_path and index_path.exists() and not self._store_cloud:
                    try:
                        z_arr_path = Path(zarr_path)
                        z_mtime = z_arr_path.stat().st_mtime if z_arr_path.exists() else 0
                        if index_path.stat().st_mtime >= z_mtime:
                            cached = self._load_zarr_index()
                            if dates_to_check:
                                processed_dates.update(d for d in dates_to_check if d in cached)
                                if set(dates_to_check).issubset(cached):
                                    logger.debug("Using up-to-date zarr index for processed dates")
                                    return processed_dates
                            else:
                                processed_dates.update(cached)
                                logger.info(f"Loaded {len(processed_dates)} processed dates from index: {index_path}")
                                return processed_dates
                    except Exception:
                        logger.debug("Failed to use zarr index, will probe store")

                # Open Zarr using consolidated metadata if available
                store = None
                try:
                    if (not self._store_cloud) and (Path(zarr_path) / ".zmetadata").exists():
                        store = zarr.open_consolidated(str(zarr_path))
                    else:
                        store = zarr.open_group(str(zarr_path), mode="r")
                except Exception:
                    store = None

                if store is not None and "patches" in store:
                    # MajorTOM sparse store: probe only the requested dates (fast)
                    patches_grp = store["patches"]
                    first_var = next(iter(patches_grp), None)
                    if first_var:
                        var_grp = patches_grp[first_var]
                        if dates_to_check:
                            for d in dates_to_check:
                                if d in var_grp:
                                    processed_dates.add(d)
                        else:
                            processed_dates.update(list(var_grp.group_keys()))
                    # Save index for faster future lookups
                    if not self._store_cloud and processed_dates:
                        self._update_zarr_index(processed_dates)
                else:
                    # Regular consolidated Zarr with a time dimension — use xarray
                    try:
                        ds = xr.open_zarr(str(zarr_path))
                        if "time" in ds.dims:
                            dates = pd.to_datetime(ds.time.values).strftime("%Y-%m-%d")
                            if dates_to_check:
                                processed_dates.update(d for d in dates if d in dates_to_check)
                            else:
                                processed_dates.update(dates)
                            if not self._store_cloud and processed_dates:
                                self._update_zarr_index(processed_dates)
                    except Exception:
                        logger.info(f"No existing Zarr file found at {zarr_path}, starting fresh.")
            except Exception:
                logger.info(f"No existing Zarr file found at {zarr_path}, starting fresh.")

        else:
            # Original TIFF logic
            if self._store_cloud:
                # GDAL /vsis3/ path
                tif_out_dir = f"{self.granule_dir.rsplit('/', 1)[0]}/tiffs"
                files = gdal.ReadDir(tif_out_dir)
                if files:
                    for filename in files:
                        if filename.endswith(".tif"):
                            parts = filename.split("_")
                            if len(parts) >= 2:
                                date_str = parts[-1].replace(".tif", "")
                                try:
                                    date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                                    processed_dates.add(date_fmt)
                                except ValueError:
                                    logger.debug(f"Skipping S3 file with non-matching date: {filename}")
                logger.info(f"Found {len(processed_dates)} processed dates in MinIO: {tif_out_dir}")

            else:
                # Local filesystem
                tif_out_dir = Path(self.granule_dir).parent / "tiffs"
                if tif_out_dir.exists():
                    for tif_file in tif_out_dir.glob("*.tif"):
                        date_str = tif_file.stem.split('_')[-1]
                        try:
                            date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                            processed_dates.add(date_fmt)
                        except ValueError:
                            logger.debug(f"Skipping local file with non-matching date: {tif_file.name}")
                logger.info(f"Found {len(processed_dates)} processed dates locally: {tif_out_dir}")

        return processed_dates

    def _exclude_processed_files(self, files):
        processed_dates = self._get_processed_dates()

        # Step 2: Filter self.files
        filtered_files = []
        for h5_file in files:
            h5_stem = Path(h5_file).stem
            # Extract the AYYYYDDD part
            doy_part = [part for part in h5_stem.split('.') if part.startswith('A')]
            if not doy_part:
                continue  # skip if no AYYYYDDD found
            doy_str = doy_part[0][1:]  # remove the 'A', e.g., '2012001'
            # Convert YYYYDDD to YYYYMMDD
            date_obj = datetime.strptime(doy_str, "%Y%j")
            date_str = date_obj.strftime("%Y-%m-%d")

            if date_str not in processed_dates:
                filtered_files.append(h5_file)

        pattern = re.compile(r'\.h(\d+)v(\d+)\.')
        files_sorted = sorted(filtered_files, key=lambda x: tuple(map(int, pattern.search(x).groups())))

        return files_sorted


    def _mosaic_daily(self, max_workers=1):
        """
        Process all tiles, mosaic and reproject them per day.
        Export either as Zarr or TIFF based on self.arg.output.
        """
        logger.info("🧩 Grouping tiles by date...")

        # --- Group files by date ---
        files_by_date = defaultdict(list)
        for f in self.files:
            files_by_date[self._get_date(f)].append(f)

        variables = [self.variable] if isinstance(self.variable, str) else list(self.variable)

        if self._store_cloud is False:
            self.tif_out_dir = self.granule_dir.parent / "tiffs"
        else:
            self.tif_out_dir = self.temp_dir / "tiffs"

        self.tif_out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------
        # Helper: process one day
        # ------------------------
        def process_one_day(date_files):
            date, files = date_files
            variable_data = {}

            # --- Cache subdatasets for all tiles ---
            subdatasets_cache = {}
            for f in files:
                ds = gdal.Open(str(f))
                if ds is None:
                    logger.warning(f"Could not open {f}, skipping.")
                    if f.endswith(self.raw_data_type):
                        Path(f).unlink(missing_ok=True)
                    continue
                subdatasets_cache[f] = ds.GetSubDatasets()

            # --- Process each variable ---
            for var in variables:
                subdatasets = [
                    s[0]
                    for f in files
                    for s in subdatasets_cache.get(f, [])
                    if var in s[0]
                ]
                if not subdatasets:
                    logger.warning(f"No subdatasets for '{var}' on {date:%Y-%m-%d}")
                    continue

                da = self._build_mosaic(subdatasets, var)
                if da is None:
                    continue

                # --- Custom preprocessing ---
                # da = self._custom_preprocess(da)
                da = da.expand_dims(time=[date])
                da.name = var
                variable_data[var] = da

            # --- Merge variables ---
            if not variable_data:
                return None

            ds_date = self._merge_dataarrays(list(variable_data.values()))
            logger.info(f"Reprojecting dataset for {date:%Y-%m-%d}...")
            # ds_date = self._reproject(ds_date)

            # --- Export tiff---
            if self._output_format.lower() == "tiff":
                self._export_data(ds_date, date)
            return ds_date

        # ------------------------
        # Parallel or sequential execution
        # ------------------------
        mosaics = []
        items = sorted(files_by_date.items())
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_one_day, item): item[0] for item in items}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing dates"):
                    result = fut.result()
                    if result is not None:
                        mosaics.append(result)
        else:
            for item in tqdm(items, desc="Processing dates"):
                ds_date = process_one_day(item)
                if ds_date is not None:
                    mosaics.append(ds_date)
                    del ds_date
                    gc.collect()

        # --- If Zarr is requested, merge all days into a single dataset ---
        if self._output_format.lower() == "zarr":
            if not mosaics:
                logger.warning("No daily mosaics produced — nothing to export to Zarr.")
                return
            ds = xr.concat(mosaics, dim="time")
            ds = prepare(ds)
            ds = ds.chunk({"time": 1, "lat": 1024, "lon": 1024})
            self._export_to_zarr(ds)


    # ------------------------
    # Helper: build mosaic for one variable
    # ------------------------
    def _gdal_subdataset_to_dataarray(self, subdataset_path: str) -> "xr.DataArray":
        """
        Open a GDAL subdataset path (e.g. HDF4 EOS) directly via GDAL and return
        an xarray DataArray with CRS, transform and nodata set.

        rasterio/rioxarray cannot open HDF4 EOS paths like
        HDF4_EOS:EOS_GRID:"...":Grid:Band directly, but gdal.Open() handles them
        fine.  Reading through GDAL and constructing the DataArray manually avoids
        any extra disk I/O.
        """
        from affine import Affine
        from rasterio.crs import CRS as RasterioCRS

        gdal_ds = gdal.Open(subdataset_path)
        if gdal_ds is None:
            raise RuntimeError(f"gdal.Open returned None for: {subdataset_path}")

        band = gdal_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        gt = gdal_ds.GetGeoTransform()
        proj = gdal_ds.GetProjection()
        nodata = band.GetNoDataValue()
        gdal_ds = None  # release file handle

        transform = Affine.from_gdal(*gt)
        ny, nx = data.shape
        xs = np.array([transform.c + (j + 0.5) * transform.a for j in range(nx)])
        ys = np.array([transform.f + (i + 0.5) * transform.e for i in range(ny)])

        da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
        da = da.rio.write_crs(RasterioCRS.from_wkt(proj))
        da = da.rio.write_transform(transform)
        # write_nodata without encoded=True stores in attrs (not encoding),
        # which avoids a rioxarray bug where encoding._FillValue as np.int16
        # gets re-cast through _ensure_nodata_dtype and raises ValueError.
        if nodata is not None and not np.isnan(float(nodata)):
            da = da.rio.write_nodata(int(nodata))

        return da

    def _build_mosaic(self, subdatasets, var):
        from osgeo import gdal
        import rioxarray

        try:
            gdal.UseExceptions()

            # 1. Reproject to self._crs — no preprocessing yet
            temp_tifs = []
            for i, subdataset in enumerate(subdatasets):
                try:
                    da = self._gdal_subdataset_to_dataarray(subdataset)
                except Exception as e:
                    logger.warning(f"Skipping subdataset (cannot open): {subdataset} — {e}")
                    continue

                dest_crs = self._crs

                if not dest_crs.startswith("EPSG"):
                    raise ValueError(f"Invalid CRS: {dest_crs}")

                da.attrs.pop("scale_factor", None)
                da.attrs.pop("add_offset", None)

                da_utm = da.rio.reproject(
                    dst_crs=dest_crs,
                    resampling=self._rio_resampling(),
                    resolution=self._resolution,
                )

                da_utm.attrs.pop("scale_factor", None)
                da_utm.attrs.pop("add_offset", None)

                temp_tif = self.temp_dir / f"temp_{var}_{i}.tif"
                da_utm.rio.to_raster(str(temp_tif), driver="GTiff")
                temp_tifs.append(str(temp_tif))

            if not temp_tifs:
                logger.warning(f"No valid tiles for '{var}' — all subdatasets failed to open")
                return None

            # 2. Group tiles by dtype before mosaicking
            dtype_groups = {}
            for tif in temp_tifs:
                ds = gdal.Open(tif)
                dtype = ds.GetRasterBand(1).DataType  # e.g. GDT_Float32, GDT_Byte
                ds = None
                dtype_groups.setdefault(dtype, []).append(tif)

            if len(dtype_groups) > 1:
                # Pick the dominant group (most tiles), warn about the rest
                dominant_dtype, dominant_tifs = max(dtype_groups.items(), key=lambda x: len(x[1]))
                logger.warning(
                    f"Mixed dtypes detected for {var}: "
                    f"{ {gdal.GetDataTypeName(k): len(v) for k, v in dtype_groups.items()} }. "
                    f"Using dtype '{gdal.GetDataTypeName(dominant_dtype)}' group only."
                )
                tifs_to_mosaic = dominant_tifs
            else:
                tifs_to_mosaic = temp_tifs

            # 3. Build VRT + translate with homogeneous dtype group
            vrt_path = self.temp_dir / f"{self.short_name}_{var}.vrt"
            tif_path = self.temp_dir / f"{self.short_name}_{var}.tif"

            vrt = gdal.BuildVRT(str(vrt_path), tifs_to_mosaic, creationOptions=["NUM_THREADS=ALL_CPUS"])
            if vrt is None:
                raise RuntimeError("BuildVRT returned None")

            gdal.Translate(
                str(tif_path),
                vrt,
                creationOptions=["TILED=YES", "BIGTIFF=YES", "COMPRESS=LZW", "NUM_THREADS=ALL_CPUS"]
            )
            vrt = None

            # 4. Preprocess ONCE on the full mosaic — much faster
            da_mosaic = rioxarray.open_rasterio(tif_path, chunks=True).squeeze("band", drop=True)
            da_mosaic = self._custom_preprocess(da_mosaic, band_name=os.path.basename(str(subdatasets[0])))
            return da_mosaic

        except Exception as e:
            logger.warning(f"GDAL mosaic failed for {var}: {e}")
            return None


    def _drop_empty_coordinates(self, ds:xr.Dataset):
        # Only drop if coordinates exist
        if "lat" in ds.coords:
            ds = ds.dropna(dim="lat", how="all")
        if "lon" in ds.coords:
            ds = ds.dropna(dim="lon", how="all")
        if "x" in ds.coords:
            ds = ds.dropna(dim="x", how="all")
        if "y" in ds.coords:
            ds = ds.dropna(dim="y", how="all")
        return ds


    def _export_multiband_raster(
        self,
        ds: xr.Dataset | xr.DataArray,
        date,
        tile_dir: str = None,
        dtype: str = "int16",
    ):
        """
        Export a multiband GeoTIFF either locally or directly to MinIO.
        Storage location is determined by self._store_cloud flag.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataArray
            Dataset to export
        date : datetime
            Date for the filename
        tile_dir : str, optional
            Local directory path (required if store_cloud=False)
        dtype : str
            Data type for output, default "int16"
        """

        # Construct output path
        filename = f"{self.short_name}_{date:%Y%m%d}.tif"

        if self._store_cloud:
            # Cloud: full /vsis3/ path as string
            out_tif = f"{self.tif_out_dir}/{filename}"
        else:
            # Local filesystem
            if tile_dir is None:
                tile_dir = self.tif_out_dir

            tile_dir = Path(tile_dir)
            tile_dir.mkdir(parents=True, exist_ok=True)
            out_tif = tile_dir / filename

        var_names = list(ds.data_vars)
        first = ds[var_names[0]]

        time, height, width = first.shape
        transform = first.rio.transform()
        crs = first.rio.crs

        is_float = np.issubdtype(np.dtype(dtype), np.floating)
        predictor = 3 if is_float else 2

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(var_names),
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "DEFLATE",
            "ZLEVEL": 5,
            "TILED": True,
            "BLOCKXSIZE": 256,
            "BLOCKYSIZE": 256,
            "PREDICTOR": predictor,
            "BIGTIFF": "YES" if (height * width * len(var_names) * 2 > 4e9) else "NO",
        }

        with rasterio.open(str(out_tif), "w", **profile) as dst:
            for i, var in enumerate(var_names, start=1):
                arr = ds[var].data.squeeze()
                dst.write(arr.astype(dtype), i)
                dst.set_band_description(i, var)

        if self._store_cloud:
            self._minio_client.fput_object(
            os.getenv('MINIO_BUCKET'),
            f"{extract_object_from_minio(self.granule_dir)}/tiffs/{filename}",
            str(out_tif),
        )


    def _export_single_bands(self, ds_date, date, tile_dir=None):
        """
        Export single-band rasters, either locally or to MinIO.
        """
        if tile_dir is None and not self._store_cloud:
            tile_dir = self.tif_out_dir

        for var in ds_date.data_vars:
            filename = f"{self.short_name}_{var}_{date:%Y%m%d}.tif"

            if not self._store_cloud:
                out_tif = Path(tile_dir) / filename
                Path(tile_dir).mkdir(parents=True, exist_ok=True)

            da = ds_date[var]

            # Determine appropriate predictor based on data type
            # PREDICTOR=3 (float prediction) only for float dtypes
            # PREDICTOR=2 (horizontal differencing) for integer dtypes
            is_float = np.issubdtype(da.dtype, np.floating)
            predictor = 3 if is_float else 2

            # Optimized GeoTIFF options for single-band export
            rio_kwargs = {
                "driver": "GTiff",
                "compress": "DEFLATE",
                "ZLEVEL": 5,              # Balance compression vs speed (~2-3x faster than ZLEVEL=9)
                "TILED": True,
                "BLOCKXSIZE": 256,
                "BLOCKYSIZE": 256,
                "PREDICTOR": predictor,   # 3 for float, 2 for integer
                "dtype": str(da.dtype)
            }

            da.rio.to_raster(str(out_tif), **rio_kwargs)

            storage_type = "MinIO" if self._store_cloud else "local"
            logger.debug(f"✅ Exported {var} to {storage_type}: {out_tif}")

    # ------------------------
    # Helper: export dataset according to arg.output
    # ------------------------
    def _export_data(self, ds_date:xr.Dataset | xr.DataArray, date:str):
        """ Export daily dataset according to self._output_format """
        if self._output_format.lower() == "tiff":
            self._export_multiband_raster(ds_date, date)
        else:
            raise ValueError(f"Unknown output format '{self._output_format}'")

    def _merge_dataarrays(self, dataarrays: list[xr.DataArray]) -> xr.Dataset:
        """
        Merge a list of xarray DataArrays into a Dataset.
        Checks for differing spatial resolutions and issues a warning.
        For MOD09GA, resamples state_1km to match the resolution of other bands (500m).
        """
        if not dataarrays:
            return xr.Dataset()

        # Check spatial resolutions
        resolutions = []
        for da in dataarrays:
            if hasattr(da, 'rio') and da.rio.crs is not None:
                res = da.rio.resolution()
                resolutions.append(res)
            else:
                resolutions.append(None)

        unique_res = set(tuple(r) if r is not None else None for r in resolutions)
        if len(unique_res) > 1:
            logger.warning(f"Different spatial resolutions detected among DataArrays: {unique_res}")

        # For MOD09GA, resample state_1km to 500m if necessary
        if self.short_name == "MOD09GA":
            target_da = None
            for da in dataarrays:
                if da.name != "state_1km":
                    target_da = da
                    break
            if target_da is not None:
                for i, da in enumerate(dataarrays):
                    if da.name == "state_1km":
                        # Resample state_1km to match target_da resolution
                        dataarrays[i] = da.rio.reproject_match(
                            target_da,
                            resampling=self._rio_resampling(),
                        )
                        logger.info("Resampled state_1km to match other bands' resolution.")
                        break

        # Merge the DataArrays
        return xr.merge(dataarrays)

    def _convert_crs(self, ds: xr.Dataset | xr.DataArray, dst_crs: str, bbox_deg: tuple[float, float, float, float]) -> xr.Dataset | xr.DataArray:
        """
        Transform bounding box to dataset CRS and clip the dataset.
        """
        if isinstance(ds, xr.DataArray):
            ds_repr = ds.to_dataset(name=ds.name)
        else:
            ds_repr = ds

        project = pyproj.Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
        minx, miny = project(bbox_deg[0], bbox_deg[1])
        maxx, maxy = project(bbox_deg[2], bbox_deg[3])

        return [minx, miny, maxx, maxy]

    def _reproject(self, ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:

        if isinstance(ds, xr.DataArray):
            ds_repr = ds.to_dataset(name=ds.name)
        else:
            ds_repr = ds

        if self._reproj_lib == "rioxarray":

            ds_repr = ds_repr.rio.reproject(
                dst_crs=self._crs,
                resampling=self._rio_resampling(),
                resolution=self._resolution,
            )

            if (self._crs == "EPSG:6933" or self._crs.startswith("EPSG:326") or self._crs.startswith("EPSG:327")) and self.bbox is not None:
                # Convert bbox to dataset CRS
                bbox_projected = self._convert_crs(ds_repr, self._crs, self.bbox)
                minx, miny, maxx, maxy = bbox_projected
            elif self._crs == "EPSG:4326" and self.bbox is not None:
                minx, miny, maxx, maxy = self.bbox
            else:
                minx, miny, maxx, maxy = ds_repr.rio.bounds()

            ds_tmp = ds_repr.rio.clip_box(
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,
                    allow_one_dimensional_raster=True,
                )

            # Drop degenerate clips
            if ds_tmp.sizes.get("x", 0) <= 1 or ds_tmp.sizes.get("y", 0) <= 1:
                logger.warning(
                    f"⚠️ Clipped raster collapsed to 1D "
                    f"(x={ds_tmp.sizes.get('x')}, y={ds_tmp.sizes.get('y')}), skipping."
                )
                return ds_repr

        elif self._reproj_lib == "xesmf":
            ds_repr = self._regridding_with_xe_modis(
                ds_repr,
                method=self._reproj_method
            )
        else:
            raise ValueError(f"Unknown reproj_lib '{self._reproj_lib}'")

        if isinstance(ds, xr.DataArray):
            return ds_repr[ds.name]
        else:
            return ds_repr

    # ------------------------
    # Export to Zarr
    # ------------------------
    def _export_to_zarr(self, ds):
        """
        Export dataset to Zarr format, either locally or to MinIO.
        Handles both creating new stores and appending to existing ones.
        Logs bbox and collection metadata.
        """
        if self._store_cloud:
            zarr_path = f"/vsis3/{self.minio_bucket}/{self.minio_prefix}/{self.collection_name}_dataset.zarr"
        else:
            zarr_path = self.zarr_path

        logger.info(f"💾 Exporting data to {zarr_path} for bbox {self.bbox if hasattr(self, 'bbox') else 'N/A'}")

        if self._store_cloud:
            # For cloud storage, we need to check existence differently
            try:
                ds_existing = xr.open_zarr(zarr_path)
                if self._add_new_variables:
                    # Add new variable arrays without touching existing time steps
                    ds.to_zarr(zarr_path, mode="a")
                else:
                    ds.to_zarr(zarr_path, mode="a", append_dim="time")
            except (FileNotFoundError, KeyError, ValueError):
                # Create new
                ds.to_zarr(zarr_path, mode="w")
        else:
            # Ensure some free space before attempting write
            if not self._check_disk_space(min_gb=1):
                logger.warning("Low disk space before exporting Zarr — attempting cleanup")
                self._cleanup_raw_files()

            # Determine if store already exists to choose the right mode
            zarr_exists = Path(zarr_path).exists()
            target_mode = "a" if zarr_exists else "w"
            logger.debug(f"Zarr store at {zarr_path} exists={zarr_exists}, using mode='{target_mode}'")

            # Retry on ENOSPC a few times after attempting cleanup
            for attempt in range(3):
                try:
                    if zarr_exists:
                        if self._add_new_variables:
                            # Add new variable arrays without touching existing time steps
                            ds.to_zarr(zarr_path, mode="a")
                        else:
                            ds.to_zarr(zarr_path, mode="a", append_dim="time")
                    else:
                        ds.to_zarr(zarr_path, mode="w")
                    break
                except OSError as e:
                    if getattr(e, 'errno', None) == errno.ENOSPC:
                        logger.warning(f"ENOSPC encountered while writing zarr (attempt {attempt+1})")
                        self._cleanup_raw_files()
                        try:
                            shutil.rmtree(self.temp_dir, ignore_errors=True)
                            self.temp_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            logger.debug("Failed to recreate temp_dir after cleanup")
                        time.sleep(2)
                        continue
                    else:
                        raise

            # Consolidate metadata for faster opens next time
            try:
                zarr.consolidate_metadata(str(zarr_path))
            except Exception:
                logger.debug("Failed to consolidate zarr metadata; continuing")

            # Update small index of dates for fast lookups
            try:
                if hasattr(ds, "time") and not self._store_cloud:
                    dates = pd.to_datetime(ds.time.values).strftime("%Y-%m-%d")
                    self._update_zarr_index(dates)
                    logger.debug(f"Updated zarr index with {len(dates)} new dates for bbox {self.bbox if hasattr(self, 'bbox') else 'N/A'}")
            except Exception:
                logger.debug("Failed to update zarr index after export")

    # ------------------------
    # Cleanup
    # ------------------------
    def _cleanup_raw_files(self):
        """Delete raw downloaded files (HDF/H5) to free disk space."""
        raw_extensions = (self.raw_data_type.lstrip("."), self.raw_data_type)
        deleted_count = 0

        if self.granule_dir.exists():
            for f in os.listdir(self.granule_dir):
                if f.endswith(raw_extensions):
                    file_path = os.path.join(self.granule_dir, f)
                    try:
                        Path(file_path).unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"🧹 Deleted {deleted_count} raw data files from {self.granule_dir}")

    def cleanup(self, clean_nontemp=False):
        logger.info(f"🧹 Cleaning up temporary files in {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if clean_nontemp:
            self._cleanup_raw_files()

    # ------------------------
    # Full pipeline with batch support
    # ------------------------
    def _split_date_range(self, start_date, end_date, batch_days=30):
        """
        Split a date range into batches of specified days.

        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        batch_days : int
            Number of days per batch

        Returns
        -------
        list of tuples
            List of (batch_start, batch_end) date ranges
        """
        from datetime import datetime, timedelta

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        batches = []

        current = start
        while current <= end:
            batch_end = min(current + timedelta(days=batch_days - 1), end)
            batches.append((current.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")))
            current = batch_end + timedelta(days=1)

        return batches

    def build_tile_to_grid_lookup(self, grid, cache_path):
        """
        Build mapping:
            (h, v) -> list of (grid_id, lat, lon)
        using majortom.Grid points.
        """
        import pickle
        from tqdm import tqdm
        from transform.majortom import CalculationsMajorTom
        calculations = CalculationsMajorTom()
        tile_lookup = {}

        points = grid.points  # GeoDataFrame

        for _, row in tqdm(points.iterrows(), total=len(points)):
            grid_id = row["name"]
            lon = row.geometry.x
            lat = row.geometry.y

            # lat/lon → MODIS sinusoidal
            x, y = calculations.latlon_to_sinu(lat, lon)
            h, v = calculations.sinu_to_tile(x, y)

            tile_lookup.setdefault((h, v), []).append(
                (grid_id, lat, lon)
            )

        with open(cache_path, "wb") as f:
            pickle.dump(tile_lookup, f)

        return tile_lookup

    def _generate_global_aoi(self,
            pixel_resolution_m:int,
            patch_size_px:int,
            target_grid_km:bool=None,
            generate_global:bool=False
        ):

        import pickle

        L = patch_size_px * pixel_resolution_m  # patch size in meters

        if target_grid_km is None:
            D = L
        else:
            D = target_grid_km * 1_000

        overlap_m = max(0, L - D)
        gap_m = max(0, D - L)

        calculations = {
            "patch_size_km": L / 1_000,
            "grid_spacing_km": D / 1_000,
            "overlap_km": overlap_m / 1_000,
            "gap_km": gap_m / 1_000,
        }

        logger.info(f"🗺️ Generating global AOI with patch size {calculations['patch_size_km']:.2f} km, grid spacing {calculations['grid_spacing_km']:.2f} km, overlap {calculations['overlap_km']:.2f} km, gap {calculations['gap_km']:.2f} km")

        self.grid = Grid(dist=calculations["grid_spacing_km"])

        if not generate_global:
            return None

        grid_file = Path(DATA_PATH) / f"tile_to_grid_global_{pixel_resolution_m}_{patch_size_px}_{target_grid_km}.pkl"

        if grid_file.exists():
            logger.info(f"Loading cached global tile-to-grid mapping from {grid_file}")
            with open(grid_file, "rb") as f:
                return pickle.load(f)

        tile_to_grid = self.build_tile_to_grid_lookup(
            self.grid,
            cache_path=grid_file,
        )
        return tile_to_grid

    def run(self, batch_days=30,
            majortom_grid: bool = False,
            pixel_size=250,
            patch_size=64,
            reference_band=None):

        if majortom_grid:
            from transform import CalculationsMajorTom
            self.calculations =  CalculationsMajorTom(pixel_size=pixel_size)
            tile_to_grid = self._generate_global_aoi(generate_global=True,
                                                     pixel_resolution_m=pixel_size,
                                                     patch_size_px=patch_size,
                                                     target_grid_km=100)
            if reference_band is None:
                reference_band = self.variable if isinstance(self.variable, str) else self.variable[0]
                logger.info(f"No reference_band provided, using '{reference_band}' as default for grid alignment.")
        else:
            tile_to_grid = None

        start_date, end_date = self.date_range
        batches = self._split_date_range(start_date, end_date, batch_days=batch_days)

        if not self._store_cloud:
            self._cleanup_raw_files()

        logger.info(f"📦 Processing {len(batches)} batches of ~{batch_days} days")

        for i, (batch_start, batch_end) in enumerate(batches, 1):
            logger.info(f"Batch {i}/{len(batches)}: {batch_start} → {batch_end}")

            self._search_and_download(date_range=(batch_start, batch_end))

            if not self.files:
                logger.info("No new files — skipping batch")
                continue

            if majortom_grid:
                self.build_or_update_majortom_zarr(
                    tile_to_grid=tile_to_grid,
                    patch_size=patch_size,
                    reference_band=reference_band
                )
            else:
                self._mosaic_daily()

            if not self._store_cloud:
                self._cleanup_raw_files()
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✅ All batches completed")

    def _init_or_open_sparse_zarr(self, patch_size):
        PATCH_SIZE = 2 * patch_size
        zarr_path = Path(self.zarr_path)

        # Remove stale consolidated metadata if it exists, so zarr doesn't try to use
        # outdated metadata when adding new dates in subsequent batches
        zmetadata_path = zarr_path / ".zmetadata"
        if zmetadata_path.exists():
            try:
                zmetadata_path.unlink()
                logger.debug(f"Removed stale consolidated metadata at {zmetadata_path}")
            except Exception as e:
                logger.debug(f"Could not remove .zmetadata: {e}")

        store = zarr.open_group(zarr_path, mode="a", zarr_format=2)

        from numcodecs import Blosc

        compressor = Blosc(
            cname="zstd",     # best general-purpose choice
            clevel=5,         # 1–5 is usually optimal
            shuffle=Blosc.BITSHUFFLE,
        )

        variables = (
            self.variable
            if isinstance(self.variable, (list, tuple))
            else [self.variable]
        )

        # --- metadata ---
        store.attrs.setdefault("grid", "MajorTOM")
        store.attrs.setdefault("projection", "MODIS sinusoidal")
        store.attrs.setdefault("resolution_m", self._resolution)
        store.attrs.setdefault("patch_size", PATCH_SIZE)

        # --- index arrays ---
        if "grid_id" not in store:
            store.create(
                "grid_id",
                shape=(0,),
                chunks=(1024,),
                dtype="int32",
            )

        if "time" not in store:
            store.create(
                "time",
                shape=(0,),
                chunks=(1024,),
                dtype="datetime64[D]",
            )

        # --- variable arrays ---
        patches_grp = store.require_group("patches")

        # create sub-groups for variables only once
        for var in variables:
            if var not in patches_grp:
                patches_grp.require_group(var)

        return store, patches_grp, compressor

    def _gdal_band_preprocess(self, data, band, description:str):

        raw_name = description.split(":")[-1].lower()
        band_name = re.sub(r"_\d+$", "", raw_name)

        if band_name in REFL_BANDS:
            data = data.astype(np.float16)
            nodata = band.GetNoDataValue()
            if nodata is not None:
                data[data == nodata] = np.nan
            scale = band.GetScale() or 1.0
            offset = band.GetOffset() or 0.0
            data = data * scale + offset

        elif band_name in QC_BANDS:
            data = data.astype(np.uint16)
            data = MODISQCMask.get_mask(data, band_name)

        elif band_name in CLOUD_BANDS:
            data = data.astype(np.uint16)  # store raw QA bits; decode at load time

        return data, band_name

    def build_or_update_majortom_zarr(
        self,
        tile_to_grid,
        patch_size=64,
        reference_band="sur_refl_b01",
        max_null_fraction=0.1,
        ):

        if not tile_to_grid:
            logger.warning("No MajorTOM tile mapping provided")
            return

        store, patches_grp, compressor = self._init_or_open_sparse_zarr(patch_size)

        if self.short_name =='MOD35_L2':
            written_dates =  self._stream_mod35_swath(
                patches_grp,
                compressor,
                tile_to_grid,
                patch_size,
                max_null_fraction,
            )
        else:
            written_dates = self._stream_modis_tiles(
                patches_grp=patches_grp,
                compressor=compressor,
                tile_to_grid=tile_to_grid,
                patch_size=patch_size,
                reference_band=reference_band,
            )

        # Update index and consolidate metadata for faster subsequent opens
        if written_dates and not self._store_cloud:
            try:
                self._update_zarr_index(written_dates)
                zarr.consolidate_metadata(str(self.zarr_path))
            except Exception:
                logger.debug("Failed to update zarr index or consolidate metadata")

    def _stream_mod35_swath(
        self,
        patches_grp,
        compressor,
        tile_to_grid,
        patch_size,
        max_null_fraction,
    ):
        HALF = patch_size
        PATCH_SIZE = 2 * HALF
        # Keep the intermediate lat/lon grid at MOD35's native ~1 km resolution
        # (0.01° ≈ 1113 m) so pyresample stays fast.  The geographic footprint is
        # matched to the co-registered product (e.g. MOD09GQ at 250 m) by extracting
        # a proportionally smaller sub-patch and upsampling it to PATCH_SIZE×PATCH_SIZE.
        self.grid_res = 0.01                                     # degrees, ~1 km native
        _grid_res_m   = self.grid_res * 111_320                  # metres per native pixel
        # Half-width in native pixels that spans the same area as HALF * self._resolution
        _native_half  = max(1, round(HALF * self._resolution / _grid_res_m))
        # Integer upsampling factor (ceiling) so np.kron output is ≥ PATCH_SIZE
        _upsample_k   = -(-PATCH_SIZE // (2 * _native_half))    # ceiling division

        logger.info(f"🌥️ Streaming {len(self.files)} MOD35 swaths")

        mod35 = MOD35SwathProcessor(
            radius_of_influence=1500,
            fill_value=np.nan,
            nprocs=4,
        )

        # Flatten all grid cells once — reused for every file
        grid_cells = sum(tile_to_grid.values(), [])
        all_gids  = np.array([gid       for gid, _, _  in grid_cells])
        all_lats  = np.array([lat       for _, lat, _   in grid_cells])
        all_lons  = np.array([lon       for _, _, lon   in grid_cells])

        from collections import defaultdict
        written_counts = defaultdict(int)

        for file_path in tqdm(self.files, desc="MOD35 swaths"):
            try:
                date_str = str(np.datetime64(self._get_date(file_path), "D"))

                # --------------------------------------------------
                # 1. Read swath (lat/lon needed before resampling)
                # --------------------------------------------------
                cloud_raw, swath_lat, swath_lon = mod35._read_swath(file_path)

                # --------------------------------------------------
                # 2. Compute swath bbox and filter grid cells
                # --------------------------------------------------
                lon_min, lon_max, lat_min, lat_max = mod35._get_swath_bbox(
                    swath_lat, swath_lon, padding=2*self.grid_res
                )

                in_bbox = (
                    (all_lons >= lon_min) & (all_lons <= lon_max) &
                    (all_lats >= lat_min) & (all_lats <= lat_max)
                )

                if not in_bbox.any():
                    logger.debug(f"No grid cells in swath footprint, skipping {file_path}")
                    continue

                # relevant_gids = all_gids[in_bbox]
                relevant_lats = all_lats[in_bbox]
                relevant_lons = all_lons[in_bbox]
                relevant_gids = all_gids[in_bbox]

                # --------------------------------------------------
                # 3. Build small target grid for this swath only
                # --------------------------------------------------
                target_def = mod35.build_regular_latlon_grid(
                    bbox=[lon_min, lon_max, lat_min, lat_max],
                    res_deg=self.grid_res)


                # --------------------------------------------------
                # 4. Extract 4-level quality + build swath definition
                # Bits 1-2 of Cloud_Mask byte:
                #   0=confident cloudy, 1=probably cloudy,
                #   2=probably clear,   3=confident clear
                # --------------------------------------------------
                cloud_quality = ((cloud_raw.astype(np.uint8) >> 1) & 0b11).astype(np.float32)

                # Ensure lat/lon match cloud_quality resolution
                swath_def, cloud_quality = mod35._build_swath_def(
                    swath_lat,
                    swath_lon,
                    cloud_quality
                )

                # --------------------------------------------------
                # 5. Resample onto filtered MajorTOM grid
                # --------------------------------------------------
                cloud_grid = mod35._resample(
                    swath_def,
                    target_def,
                    cloud_quality
                )

                if cloud_grid is None:
                    continue

                if len(relevant_gids) == 0:
                    continue

                # --------------------------------------------------
                # 6. Prepare Zarr groups
                # --------------------------------------------------
                # cloud_mask stores the 4-level MOD35 quality (0-3, uint8, fill=99).
                # Decode at load time with MODISQCMask.decode_mod35().
                var_grp  = patches_grp.require_group("cloud_mask")
                date_grp = var_grp.require_group(date_str)

                # --------------------------------------------------
                # 7. Write values (Patches)
                # --------------------------------------------------
                for gid, lat, lon in zip(relevant_gids, relevant_lats, relevant_lons):

                    # Convert lat/lon → row/col in bbox grid
                    row = int(round((lat - lat_min) / self.grid_res))
                    col = int(round((lon - lon_min) / self.grid_res))

                    # Clamp (important for safety)
                    row = max(0, min(row, cloud_grid.shape[0] - 1))
                    col = max(0, min(col, cloud_grid.shape[1] - 1))

                    # Skip edges (use native-resolution half-width for bounds)
                    if (
                        row - _native_half < 0 or row + _native_half > cloud_grid.shape[0] or
                        col - _native_half < 0 or col + _native_half > cloud_grid.shape[1]
                    ):
                        continue

                    # Extract native-resolution sub-patch; flip N-S so row 0 = north
                    patch_native = cloud_grid[
                        row - _native_half : row + _native_half,
                        col - _native_half : col + _native_half
                    ][::-1, :]

                    # Upsample to PATCH_SIZE×PATCH_SIZE with nearest-neighbour (np.kron)
                    # then crop the ceiling-overshoot so the stored array is exactly PATCH_SIZE.
                    patch = np.kron(
                        patch_native,
                        np.ones((_upsample_k, _upsample_k), dtype=patch_native.dtype),
                    )[:PATCH_SIZE, :PATCH_SIZE]

                    # Shape check
                    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                        continue

                    # Quality filter
                    if np.isnan(patch).mean() > max_null_fraction:
                        continue

                    # Convert dtype; fill NaN (out-of-swath) with sentinel 99
                    patch = np.where(np.isfinite(patch), patch, 99).astype(np.uint8)
                    patch = np.ascontiguousarray(patch)

                    gid_str = str(gid)

                    # Write cloud_mask patch (4-level quality: 0-3, fill=99)
                    if gid_str in date_grp:
                        date_grp[gid_str][:] = patch
                    else:
                        arr = date_grp.create(
                            name=gid_str,
                            shape=patch.shape,
                            chunks=patch.shape,
                            dtype="uint8",
                            compressor=compressor,
                        )
                        arr[:] = patch

                    written_counts[date_str] += 1

            except Exception as e:
                logger.error(f"❌ MOD35 error {file_path}: {e}", exc_info=True)

        for d, cnt in written_counts.items():
            logger.info(f"Written {cnt} patches for date {d}")

        return set(written_counts.keys())

    def _stream_modis_tiles(
        self,
        patches_grp,
        compressor,
        tile_to_grid,
        patch_size=64,
        reference_band="sur_refl_b01",  # 👈 choose your gatekeeper band
        max_null_fraction=0.1,
    ):
        """
        Stream raw MODIS tiles into a sparse MajorTOM Zarr store.

        High-level steps:
        1. Prepare constants and variable list.
        2. Loop over downloaded HDF files and parse the acquisition date and tile indices.
        3. For each HDF, open subdatasets and load needed bands into memory once.
        4. For every MajorTOM grid cell that falls inside the HDF tile, compute the pixel
           coordinates and extract a PATCH centered at that location.
        5. Use a reference band (typically a reflectance band) to gate writes based on
           null-fraction and spatial coverage.
        6. Write patches into the sparse Zarr `patches` group under [variable][date][grid_id].
           Writes include retry-on-ENOSPC logic and per-date success counts.
        7. Return the set of dates that received at least one successful write.

        Notes on performance and reliability:
        - Bands are loaded once per HDF to avoid repeated I/O.
        - Writing is retried a small number of times on ENOSPC after attempting cleanup.
        - Per-date summary logging (instead of listing grid ids) reduces log noise and I/O.
        """

        HALF = patch_size
        PATCH_SIZE = 2 * HALF

        # Normalize variables to a list
        variables = (
            self.variable
            if isinstance(self.variable, (list, tuple))
            else [self.variable]
        )

        if reference_band not in variables:
            raise ValueError(
                f"Reference band '{reference_band}' must be in variables list {variables}"
            )

        logger.info(
            f"🧬 Streaming {len(self.files)} MODIS tiles into sparse Zarr "
            f"(reference band = {reference_band})"
        )

        # Track which dates we wrote to and counts per date for lightweight logging
        written_dates = set()
        from collections import defaultdict as _defaultdict
        written_counts = _defaultdict(int)

        # Process each downloaded HDF file sequentially (keeps memory usage bounded)
        for file_path in tqdm(self.files, desc="MODIS tiles"):
            try:
                # Convert MODIS granule filename -> calendar date string YYYY-MM-DD
                date_str = str(np.datetime64(self._get_date(file_path), "D"))

                # --- parse MODIS tile indices from filename (hXXvYY) ---
                fname = Path(file_path).name
                m = re.search(r"h(\d+)v(\d+)", fname)
                if not m:
                    # Skip files that don't match expected naming convention
                    continue
                h, v = int(m.group(1)), int(m.group(2))

                # Map MODIS tile -> list of MajorTOM grid cells
                cells = tile_to_grid.get((h, v))
                if not cells:
                    continue

                # --- open HDF and read available subdatasets ---
                hdf = gdal.Open(str(file_path))
                if hdf is None:
                    continue

                subdatasets = dict(hdf.GetSubDatasets())

                # --- load each requested band once into memory ---
                band_data = {}
                band_types = {}
                ny = nx = None

                for var in variables:
                    var_path = next(
                        (p for p, d in subdatasets.items() if var in d),
                        None,
                    )
                    if var_path is None:
                        band_data[var] = None
                        continue

                    ds = gdal.Open(var_path)
                    if ds is None:
                        band_data[var] = None
                        continue

                    # Read array and metadata, then apply product-specific transforms
                    desc = ds.GetDescription()
                    data = ds.ReadAsArray()
                    band = ds.GetRasterBand(1)

                    data, band_name = self._gdal_band_preprocess(data, band, desc)

                    ny, nx = data.shape
                    band_data[var] = data
                    band_types[var] = band_name
                    ds = None

                # Safety: ensure reference band is reflectance (used to gate writes)
                if band_types.get(reference_band) not in REFL_BANDS:
                    logger.warning(
                        f"Reference band {reference_band} is not reflectance: {band_types.get(reference_band)}"
                    )

                # Ensure date groups exist for each variable in the Zarr `patches` group.
                # Use require_group but catch if group already exists (safe on append).
                for var in variables:
                    var_grp = patches_grp.require_group(var)
                    try:
                        var_grp.require_group(date_str)
                    except Exception as e:
                        # If date group already exists, we can safely open it for writing new grid_ids
                        logger.debug(f"Date group {date_str} already exists for {var}, will append: {e}")

                # Mark this date as being written to (we may later filter by successful writes)
                written_dates.add(date_str)

                # --- iterate candidate MajorTOM grid cells that fall within this MODIS tile ---
                for grid_id, lat, lon in cells:
                    # Convert grid lat/lon -> sinusoidal x,y -> tile indices and pixel coords
                    x, y = self.calculations.latlon_to_sinu(lat, lon)
                    tile_h, tile_v = self.calculations.sinu_to_tile(x, y)
                    if tile_h != h or tile_v != v:
                        # Grid point does not fall into this MODIS tile
                        continue

                    row, col = self.calculations.xy_to_pixel(x, y, h, v)

                    # Skip cells where patch would exceed tile bounds
                    if (
                        row - HALF < 0 or col - HALF < 0
                        or row + HALF > ny or col + HALF > nx
                    ):
                        continue

                    # --- reference band gating: reject patches with too many nulls ---
                    ref_data = band_data.get(reference_band)
                    if ref_data is None:
                        continue

                    ref_patch = ref_data[row - HALF : row + HALF, col - HALF : col + HALF]
                    if ref_patch.shape != (PATCH_SIZE, PATCH_SIZE):
                        continue

                    null_fraction = np.isnan(ref_patch).mean()
                    if null_fraction > max_null_fraction:
                        # Reject this grid cell for being mostly invalid
                        continue

                    # Debug info for first cell in this tile
                    if grid_id == cells[0][0]:
                        logger.debug(
                            f"DEBUG {reference_band}: nan_frac={np.isnan(ref_patch).mean():.3f}, band_type={band_types[reference_band]}"
                        )

                    # --- write patches for all requested variables ---
                    for var in variables:
                        data = band_data.get(var)
                        if data is None:
                            continue

                        patch = data[row - HALF : row + HALF, col - HALF : col + HALF]
                        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                            continue

                        patch = np.ascontiguousarray(patch, dtype=np.float16)

                        date_grp = patches_grp[var][date_str]
                        gid = str(grid_id)

                        band_name = band_types[var]
                        if band_name in REFL_BANDS:
                            dtype = "float16"
                            patch = np.ascontiguousarray(patch, dtype=np.float16)
                        elif band_name in QC_BANDS:
                            # decoded QC mask stored as uint8 (boolean)
                            patch = np.ascontiguousarray(patch.astype(np.uint8))
                            dtype = "uint8"
                        elif band_name in CLOUD_BANDS:
                            # raw QA bits stored as uint16 for load-time decoding
                            patch = np.ascontiguousarray(patch.astype(np.uint16))
                            dtype = "uint16"
                        else:
                            # Unsupported band type for writing
                            continue

                        # Perform write with a short retry loop on ENOSPC (disk full).
                        # On ENOSPC we attempt lightweight cleanup and retry a few times.
                        success = False
                        for attempt_write in range(3):
                            try:
                                if gid in date_grp:
                                    date_grp[gid][:] = patch
                                else:
                                    arr = date_grp.create(
                                        name=gid,
                                        shape=patch.shape,
                                        chunks=patch.shape,
                                        dtype=dtype,
                                        compressor=compressor,
                                    )
                                    arr[:] = patch
                                success = True
                                written_counts[date_str] += 1
                                break
                            except OSError as e:
                                # Only handle ENOSPC here; other errors should surface
                                if getattr(e, "errno", None) == errno.ENOSPC:
                                    logger.warning(
                                        f"ENOSPC while writing patch {gid} for {date_str} (attempt {attempt_write+1})"
                                    )
                                    # Try to free some space and retry
                                    try:
                                        self._cleanup_raw_files()
                                        shutil.rmtree(self.temp_dir, ignore_errors=True)
                                        self.temp_dir.mkdir(parents=True, exist_ok=True)
                                    except Exception:
                                        logger.debug("Cleanup attempt failed during ENOSPC handling")
                                    time.sleep(2)
                                    continue
                                else:
                                    raise
                        if not success:
                            logger.error(f"Failed to write patch {gid} for {date_str} after retries")

            except Exception as e:
                # Log and continue with next file — do not stop the entire streaming process
                logger.error(f"❌ Error processing {file_path}: {e}", exc_info=True)

        # Summarize writes per date (keeps logs compact and informative)
        for d, cnt in written_counts.items():
            logger.info(f"Written {cnt} patches for date {d}")

        logger.info("✅ Sparse MajorTOM streaming completed")
        # Return the set of dates that received at least one successful write
        return set(written_counts.keys())

import numpy as np


class MODISQCMask:
    # -----------------------------
    # Utility: extract bits
    # -----------------------------
    @staticmethod
    def _bits(arr: np.ndarray, start: int, length: int = 1) -> np.ndarray:
        """Extract bits from integer array."""
        return (arr >> start) & ((1 << length) - 1)


    # -----------------------------
    # QC_250m / QC_500m
    # -----------------------------
    @staticmethod
    def qc_reflectance_mask(QA: np.ndarray) -> np.ndarray:
        """
        Reflectance quality mask (QC_250m / QC_500m).

        Returns:
            True = BAD pixel (low quality)
        """
        modland = MODISQCMask._bits(QA, 0, 2)
        band1   = MODISQCMask._bits(QA, 2, 2)
        band2   = MODISQCMask._bits(QA, 4, 2)

        good = (modland == 0) & (band1 == 0) & (band2 == 0)

        return ~good  # True = bad


    # -----------------------------
    # MOD35 cloud mask
    # -----------------------------
    @staticmethod
    def decode_mod35(raw: np.ndarray) -> tuple:
        """
        Decode the stored MOD35 cloud_mask band (4-level quality, uint8).

        Stored values:
            0 = confident cloudy, 1 = probably cloudy,
            2 = probably clear,   3 = confident clear, 99 = fill

        Binned algorithm: quality < 2 → cloud, quality >= 2 → clear.

        Returns:
            cloud_mask : bool array, True = cloudy (fill pixels → False)
            confidence : uint8 array 0-3 (fill pixels remain 99)
        """
        raw = np.asarray(raw, dtype=np.uint8)
        valid = raw != 99
        cloud_mask = np.where(valid, raw < 2, False)
        return cloud_mask, raw.copy()

    @staticmethod
    def mod35_cloud_mask(QA: np.ndarray) -> np.ndarray:
        """
        Binary cloud mask from the stored MOD35 4-level quality band.

        Stored values: 0=confident cloudy, 1=probably cloudy,
                       2=probably clear,   3=confident clear, 99=fill.

        Binned algorithm: quality < 2 → cloudy, quality >= 2 → clear.

        Returns:
            True = cloudy (fill treated as clear)
        """
        QA = np.asarray(QA, dtype=np.uint8)
        valid = QA != 99
        return np.where(valid, QA < 2, False)


    # -----------------------------
    # MOD09 state_1km mask
    # -----------------------------
    def state_1km_cloud_mask(
        QA: np.ndarray,
        algorithm: Literal["strict", "internal", "cloud_state", "binned"] = "strict"
        ) -> np.ndarray:

        cloud_state = MODISQCMask._bits(QA, 0, 2)
        shadow      = MODISQCMask._bits(QA, 2, 1)
        aerosol     = MODISQCMask._bits(QA, 6, 2)
        cirrus      = MODISQCMask._bits(QA, 8, 1)
        int_cloud   = MODISQCMask._bits(QA, 10, 1)

        if algorithm == "strict":
            return (
                (cloud_state == 1) | (cloud_state == 2) |
                (shadow == 1) |
                (aerosol >= 2) |
                (cirrus == 1) |
                (int_cloud == 1)
            )
        elif algorithm == "cloud_state":
            return (cloud_state == 1) | (cloud_state == 2)

        elif algorithm == "binned":
            return (cloud_state == 2) | (cloud_state == 3)

        elif algorithm == "internal":
            return int_cloud == 1

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # -----------------------------
    # 1. Cloud mask from bits 0–1
    # -----------------------------
    @staticmethod
    def cloud_mask_bits_0_1(QA: xr.DataArray) -> xr.DataArray:
        """
        Cloud mask using bits 0–1 (cloud state).

        Returns:
            True = cloudy (cloudy + mixed)
            False = clear
        """
        cloud_state = QA & 0b11  # extract bits 0–1

        # 01 = cloudy, 10 = mixed
        mask = (cloud_state == 0b01) | (cloud_state == 0b10)

        return mask


    # -----------------------------
    # 2. Internal cloud mask (bit 10)
    # -----------------------------

    @staticmethod
    def cloud_mask_internal(QA: xr.DataArray) -> xr.DataArray:
        """
        MOD09 internal cloud mask (bit 10).

        Returns:
            True = cloudy
            False = clear
        """
        return (QA & (1 << 10)) != 0

    @staticmethod
    def mod35_categories(QA: np.ndarray) -> np.ndarray:
        """
        Return the stored MOD35 4-level quality value directly.

        Stored values (already extracted from bits 1-2 during download):
            0 = confident cloudy, 1 = probably cloudy,
            2 = probably clear,   3 = confident clear, 99 = fill
        """
        return np.asarray(QA, dtype=np.uint8)

    # -----------------------------
    # Binned MOD35 cloud mask (as in paper)
    # -----------------------------
    @staticmethod
    def cloud_mask_mod35_binned(QA: np.ndarray) -> np.ndarray:
        """
        MOD35 cloud mask following MODIS science team recommendation:

        Clear  = probably clear (2) + confident clear (3)
        Cloudy = confident cloudy (0) + probably cloudy (1)

        Returns:
            True = cloudy, False = clear (fill treated as clear)
        """
        QA = np.asarray(QA, dtype=np.uint8)
        valid = QA != 99
        return np.where(valid, QA < 2, False)


    # -----------------------------
    # Unified interface
    # -----------------------------
    @staticmethod
    def get_mask(QA: np.ndarray, band_name: str) -> np.ndarray:
        """
        Unified interface for MODIS QA masks.

        Returns:
            True = BAD pixel
        """
        if band_name in {"qc_250m", "qc_500m"}:
            return MODISQCMask.qc_reflectance_mask(QA)

        elif band_name == "cloud_mask":
            return MODISQCMask.mod35_cloud_mask(QA)

        elif band_name == "state_1km":
            return MODISQCMask.state_1km_cloud_mask(QA)

        else:
            raise ValueError(f"Unknown band: {band_name}")


"""
Helper function to create bitmask for a range of bits (inclusive)
"""




def bit_range_mask(start: int, end: int) -> int:
    return sum(1 << i for i in range(start, end + 1))

def decode_qc_250m(QA: xr.DataArray) -> xr.DataArray:
    """
    MOD09GA QC_250m
    Keep only pixels with:
    - MODLAND QA = 00 (ideal)
    - Band 1 quality = 00
    - Band 2 quality = 00
    """
    mask = (
        bit_range_mask(0, 1) |  # MODLAND
        bit_range_mask(2, 3) |  # Band 1 quality
        bit_range_mask(4, 5)    # Band 2 quality
    )

    filters = (
        (0b00 << 0) |
        (0b00 << 2) |
        (0b00 << 4)
    )

    QA = QA.astype(np.int32)
    return QA.where((QA & mask) == filters)



def decode_state_1km(QA: xr.DataArray) -> xr.DataArray:
    """
    MOD09GA state_1km
    Clear-sky, land, low aerosol, no snow/ice
    """
    mask = (
        bit_range_mask(0, 1) |   # cloud state
        bit_range_mask(2, 2) |   # cloud shadow
        bit_range_mask(3, 5) |   # land/water
        bit_range_mask(6, 7) |   # aerosol
        bit_range_mask(8, 8) |   # cirrus
        bit_range_mask(10,10) |  # internal cloud
        bit_range_mask(12,12)    # snow/ice
    )

    filters = (
        (0b00 << 0) |   # confident clear
        (0b0  << 2) |
        (0b001 << 3) |  # land
        (0b01 << 6) |   # low aerosol
        (0b0  << 8) |
        (0b0  << 10) |
        (0b0  << 12)
    )

    QA = QA.astype(np.int32)
    return QA.where((QA & mask) == filters)


def apply_modis_qa_mask(
        QA: xr.DataArray,
        description: str
    ) -> xr.DataArray:
        """
        Dispatch MODIS QA decoding based on GDAL subdataset description.
        """
        band_name = description.split(":")[-1].lower()

        if band_name == "qc_250m":
            return decode_qc_250m(QA)

        if band_name == "qc_500m":
            # Same logic as QC_250m, just different resolution
            return decode_qc_250m(QA)

        if band_name == "state_1km":
            return decode_state_1km(QA)

        raise ValueError(f"Unsupported MODIS QA band: {band_name}")

"""
This function has not been tested yet
"""
class StacModisTileProcessor:
    def __init__(self,
                 stac_url,
                 collection,
                 time_range,
                 bands,
                 crs="EPSG:6933",
                 resolution=1000,
                 tile_size=256,
                 stride=256,
                 max_missing=0.5):

        self.stac_url = stac_url
        self.collection = collection
        self.time_range = time_range
        self.bands = bands
        self.crs = crs
        self.resolution = resolution
        self.tile_size = tile_size
        self.stride = stride
        self.max_missing = max_missing
        self.catalog = Client.open(stac_url)
        self.ds = None
        self.out_dir = DATA_PATH / "modis"  # Place under data/ per project conventions
        self.out_dir.mkdir(exist_ok=True)

    def load_data(self):
        search = self.catalog.search(collections=[self.collection], datetime=self.time_range)
        items = list(search.get_items())
        if not items:
            raise ValueError("No items found for the given search parameters.")

        self.ds = load(
            items,
            bands=self.bands,
            crs=self.crs,
            resolution=self.resolution,
            groupby="solar_day",
            chunks={"time": 5, "x": 1024, "y": 1024},
        )
        logger.info(f"Loaded dataset: {self.ds}")

    def process_tiles(self):
        if self.ds is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        x_coords = self.ds.x.values
        y_coords = self.ds.y.values
        x_tiles = range(0, len(x_coords) - self.tile_size + 1, self.stride)
        y_tiles = range(0, len(y_coords) - self.tile_size + 1, self.stride)

        tile_id = 0
        for ix in x_tiles:
            for iy in y_tiles:
                tile = self.ds.isel(
                    x=slice(ix, ix + self.tile_size),
                    y=slice(iy, iy + self.tile_size),
                )

                # Skip tiles with too much missing data
                if tile[self.bands[0]].isnull().mean() > self.max_missing:
                    continue

                # Convert to numpy (time, bands, y, x)
                tile_np = np.stack([tile[band].values for band in self.bands], axis=1)

                np.save(self.out_dir / f"tile_{tile_id:06d}.npy", tile_np)
                tile_id += 1
        logger.info(f"Processed {tile_id} tiles.")




class MOD35SwathProcessor:
    def __init__(
        self,
        radius_of_influence=3000,
        fill_value=np.nan,
        nprocs=1,
    ):
        self.radius_of_influence = radius_of_influence
        self.fill_value = fill_value
        self.nprocs = nprocs
        self._session = None

    # ------------------------------------------------------------------
    # Auth — lazy LAADS session via earthaccess token
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        if self._session is None:
            token = os.environ["EARTHDATA_TOKEN"]

            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {token}"})

            self._session = session

        return self._session

    # ------------------------------------------------------------------
    # Filename parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_modis_filename(hdf_path: str) -> dict:
        """
        Parse a standard MODIS filename into its components.
        Format: PRODUCT.AYYYYDDD.HHMM.CCC.PRODDT.hdf
        """
        name = Path(hdf_path).name
        pattern = (
            r"^(?P<product>[A-Z0-9_]+)"
            r"\.A(?P<date>\d{7})"
            r"\.(?P<time>\d{4})"
            r"\.(?P<collection>\d{3})"
            r"\.(?P<production_dt>\d+)"
            r"\.hdf$"
        )
        m = re.match(pattern, name, re.IGNORECASE)
        if not m:
            raise ValueError(
                f"Cannot parse MODIS filename: '{name}'.\n"
                "Expected: PRODUCT.AYYYYDDD.HHMM.CCC.PRODDT.hdf"
            )
        return m.groupdict()

    @staticmethod
    def _build_mod03_shortname(parts: dict) -> str:
        product = parts["product"].upper()
        if product.startswith("MOD"):
            return "MOD03"
        elif product.startswith("MYD"):
            return "MYD03"
        raise ValueError(f"Unrecognised platform prefix in: '{product}'")

    # ------------------------------------------------------------------
    # LAADS direct download (bypasses CMR entirely)
    # ------------------------------------------------------------------

    def _laads_search_mod03(self, parts: dict) -> str:
        """
        Find the exact MOD03 filename on LAADS by listing the day directory.

        LAADS archive structure:
          https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD03/YYYY/DDD/

        Returns the full HTTPS download URL of the matching granule.
        """
        short_name  = self._build_mod03_shortname(parts)
        date_str    = parts["date"]          # YYYYDDD  e.g. 2025152
        time_str    = parts["time"]          # HHMM     e.g. 0105
        collection  = parts["collection"]    # e.g. 061 → archive uses 61

        year = date_str[:4]
        doy  = date_str[4:]                  # day-of-year, 3 digits

        archive_num = str(int(collection))   # "061" → "61"

        # Directory listing URL
        dir_url = (
            f"https://ladsweb.modaps.eosdis.nasa.gov"
            f"/archive/allData/{archive_num}/{short_name}/{year}/{doy}/"
        )

        session = self._get_session()
        resp = session.get(dir_url, timeout=30)
        resp.raise_for_status()

        # LAADS returns an HTML or JSON listing — find the matching filename
        # Filename stem we are looking for: MOD03.AYYYYDDD.HHMM.CCC.
        stem = f"{short_name}.A{date_str}.{time_str}.{collection}."
        matches = re.findall(
            rf'({re.escape(stem)}\d+\.hdf)',
            resp.text
        )

        if not matches:
            raise FileNotFoundError(
                f"No {short_name} granule found in LAADS directory:\n"
                f"  {dir_url}\n"
                f"  Looked for stem: '{stem}'\n"
                "Verify the granule exists on LAADS for this date/time."
            )

        # Pick the most recent production run if duplicates exist
        filename = sorted(matches)[-1]
        return dir_url + filename

    def _download_file(self, url: str, dest_dir: str) -> str:
        """Stream-download a LAADS file to dest_dir, return local path."""
        filename = url.split("/")[-1]
        local_path = os.path.join(dest_dir, filename)

        session = self._get_session()
        with session.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                f.flush()
                os.fsync(f.fileno())

        # --- Diagnostics ---
        file_size = os.path.getsize(local_path)
        logger.debug(f"[MOD03] Downloaded {file_size / 1024 / 1024:.1f} MB → {local_path}")

        with open(local_path, "rb") as f:
            magic = f.read(4)
        logger.debug(f"[MOD03] Magic bytes: {magic!r}  (expected: b'\\x0e\\x03\\x13\\x01')")

        if magic != b"\x0e\x03\x13\x01":
            content = open(local_path, "rb").read(500)
            logger.debug(f"[MOD03] File head: {content!r}")
            raise RuntimeError(
                f"Not a valid HDF4 file. Size={file_size} bytes.\n"
                f"Head: {content[:200]!r}"
            )

        return local_path

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _mod03_file(self, hdf_path: str):
        """
        Locate, download, yield, then delete the matching MOD03 granule.
        """
        parts = self._parse_modis_filename(hdf_path)
        url   = self._laads_search_mod03(parts)
        logger.debug(f"[MOD03] Downloading: {url.split('/')[-1]}")

        with tempfile.TemporaryDirectory(prefix="mod03_") as tmpdir:
            local_path = self._download_file(url, tmpdir)
            try:
                yield local_path
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

    # ------------------------------------------------------------------
    # Swath reading
    # ------------------------------------------------------------------

    def _read_swath(self, hdf_path: str):
        """
        Read Cloud_Mask from MOD35_L2 and 1 km lat/lon from MOD03.
        MOD03 is downloaded automatically and deleted after reading.
        """
        hdf35 = SD(hdf_path, SDC.READ)
        cloud = hdf35.select("Cloud_Mask")[0]   # (2030, 1354)
        hdf35.end()

        with self._mod03_file(hdf_path) as mod03_path:
            hdf03 = SD(mod03_path, SDC.READ)
            lat = hdf03.select("Latitude")[:]
            lon = hdf03.select("Longitude")[:]
            hdf03.end()

        if lat.shape != cloud.shape:
            raise ValueError(
                f"Shape mismatch — Cloud_Mask: {cloud.shape}, "
                f"MOD03 lat/lon: {lat.shape}."
            )

        return cloud, lat, lon

    def _get_swath_bbox(self, lat, lon, padding=0.1):
        """
        Returns (lon_min, lon_max, lat_min, lat_max) for the swath footprint.
        Padding is in degrees.
        """
        return (
            float(np.nanmin(lon)) - padding,
            float(np.nanmax(lon)) + padding,
            float(np.nanmin(lat)) - padding,
            float(np.nanmax(lat)) + padding,
        )

    def build_target_def_for_bbox(self, lon_min, lon_max, lat_min, lat_max, grid_res=0.01):
        """
        Build a pyresample GridDefinition covering only the swath bbox.
        """
        lon_grid = np.arange(lon_min, lon_max, grid_res)
        lat_grid = np.arange(lat_min, lat_max, grid_res)
        lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
        return geometry.GridDefinition(lons=lon2d, lats=lat2d), lon_grid, lat_grid

    def _decode_cloud_mask(self, cloud):
        cloud = cloud.astype(np.uint8)
        # Bit 0:   Cloud Mask Flag (1 = determined, 0 = not determined) — not a quality value
        # Bits 1-2: Unobstructed FOV Quality (00=cloudy, 01=uncertain, 10=prob.clear, 11=clear)
        cloud_quality = (cloud >> 1) & 0b00000011
        cloud_mask = (cloud_quality >= 2).astype(np.float32)  # probably clear or confident clear
        return cloud_mask, cloud_quality.astype(np.uint8)

    def _build_swath_def(self, lat, lon, data=None):

        if data is not None:
            # Upsample lat/lon if needed
            if lat.shape != data.shape:

                factor_y = data.shape[0] // lat.shape[0]
                factor_x = data.shape[1] // lon.shape[1]

                lat = np.repeat(np.repeat(lat, factor_y, axis=0), factor_x, axis=1)
                lon = np.repeat(np.repeat(lon, factor_y, axis=0), factor_x, axis=1)

            # Crop to exact match (handles 1350 vs 1354 issue)
            ny = min(lat.shape[0], data.shape[0])
            nx = min(lon.shape[1], data.shape[1])

            lat = lat[:ny, :nx]
            lon = lon[:ny, :nx]
            data = data[:ny, :nx]

            # Final safety check
            if lat.shape != data.shape:
                raise ValueError(f"Mismatch: lat {lat.shape}, data {data.shape}")

            swath_def = geometry.SwathDefinition(lons=lon, lats=lat)

            return swath_def, data

        else:
            return  geometry.SwathDefinition(lons=lon, lats=lat)

    def build_target_grid_from_latlon(self, lon2d, lat2d):
        return geometry.AreaDefinition(lons=lon2d, lats=lat2d)

    def build_regular_latlon_grid(self, bbox=None, res_deg=0.01):
        if bbox is None:
            lons = np.arange(-180, 180, res_deg)
            lats = np.arange(-90, 90, res_deg)
        else:
            lon_min, lon_max, lat_min, lat_max = bbox
            lons = np.arange(lon_min, lon_max, res_deg)
            lats = np.arange(lat_min, lat_max, res_deg)
        lon2d, lat2d = np.meshgrid(lons, lats)
        return geometry.GridDefinition(lons=lon2d, lats=lat2d)

    def _resample(self, swath_def, target_def, data):
        return resample_nearest(
            swath_def,
            data,
            target_def,
            radius_of_influence=self.radius_of_influence,
            fill_value=self.fill_value,
            nprocs=self.nprocs,
        )

    def process_cloudmodis(self, hdf_path, target_def, return_confidence=False):
        cloud_raw, lat, lon = self._read_swath(hdf_path)
        cloud_mask, cloud_flag = self._decode_cloud_mask(cloud_raw)
        swath_def = self._build_swath_def(lat, lon)
        cloud_mask_resampled = self._resample(swath_def, target_def, cloud_mask)

        if return_confidence:
            cloud_flag_resampled = self._resample(
                swath_def, target_def, cloud_flag.astype(np.float32)
            )
            return cloud_mask_resampled, cloud_flag_resampled.astype(np.uint8)

        return cloud_mask_resampled
