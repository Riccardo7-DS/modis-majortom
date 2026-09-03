import os
import math
import matplotlib.pyplot as plt


def load_land_mask():
    import ee
    return ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select('water_mask')


def bbox_size_km(bbox, pixel_size_m):
    """
    Compute the approximate width and height of a bounding box in km,
    given pixel size in meters, accounting for latitude distortion.

    Parameters
    ----------
    bbox : list or tuple
        [lon_min, lat_min, lon_max, lat_max]
    pixel_size_m : float
        Pixel size in meters

    Returns
    -------
    width_km, height_km : float
        Approximate size of bbox in km
    n_pixels_x, n_pixels_y : int
        Number of pixels along x and y at the given pixel size
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    width_deg = lon_max - lon_min
    height_deg = lat_max - lat_min

    lat_center = (lat_min + lat_max) / 2

    height_m = height_deg * 111_319
    width_m = width_deg * 111_319 * math.cos(math.radians(lat_center))

    n_pixels_x = int(width_m / pixel_size_m)
    n_pixels_y = int(height_m / pixel_size_m)

    width_km = n_pixels_x * pixel_size_m / 1000
    height_km = n_pixels_y * pixel_size_m / 1000

    print("Width (km):", width_km)
    print("Height (km):", height_km)
    print("Pixels X:", n_pixels_x)
    print("Pixels Y:", n_pixels_y)

    return width_km, height_km, n_pixels_x, n_pixels_y


def tile_has_land(bbox, scale=250):
    import ee

    land_mask = load_land_mask()

    """Return True if the bounding box contains land pixels."""
    lon_min, lat_min, lon_max, lat_max = bbox
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    water_stats = land_mask.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=region,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

    if water_stats is None:
        return False

    water_min = water_stats.get('water_mask_min')
    water_max = water_stats.get('water_mask_max')

    if water_min is None or water_max is None:
        return False

    return water_min < 1


def tile_has_min_land_fraction(bbox, min_percentage=10, scale=500, debug=False):
    """
    Return True if at least `min_percentage` of the pixels in the bbox are land.

    Parameters
    ----------
    bbox : list
        [lon_min, lat_min, lon_max, lat_max]
    min_percentage : float
        Minimum percentage of land required (0-100)
    scale : int
        Pixel size in meters (MODIS resolution)
    debug : bool
        If True, print debug info about what's being computed
    """
    import ee
    ee.Authenticate()
    ee.Initialize(project=os.environ["EE_PROJECT"])

    land_mask = load_land_mask()

    lon_min, lat_min, lon_max, lat_max = bbox
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    if debug:
        print(f"\n=== Debugging tile_has_min_land_fraction ===")
        print(f"Bbox: {bbox}")
        print(f"Looking for min_percentage >= {min_percentage}%")

    land_binary = land_mask.eq(0).toFloat()

    stats = land_binary.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=region,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

    if debug:
        print(f"Raw stats dict: {stats}")

    if stats is None:
        if debug:
            print("Stats is None, returning False")
        return False

    land_sum = None
    total_count = None

    possible_sum_keys = ['constant_sum', 'water_mask_sum', 'sum', 'constant']
    possible_count_keys = ['constant_count', 'water_mask_count', 'count']

    for key in possible_sum_keys:
        if key in stats:
            land_sum = stats[key]
            if debug:
                print(f"Found land_sum with key '{key}': {land_sum}")
            break

    for key in possible_count_keys:
        if key in stats:
            total_count = stats[key]
            if debug:
                print(f"Found total_count with key '{key}': {total_count}")
            break

    if land_sum is None or total_count is None or total_count == 0:
        if debug:
            print(f"Missing values: land_sum={land_sum}, total_count={total_count}")
        return False

    land_percentage = (land_sum / total_count) * 100

    if debug:
        print(f"Computed: {land_sum} land pixels / {total_count} total = {land_percentage:.2f}%")
        print(f"Result: {land_percentage >= min_percentage}")

    return land_percentage >= min_percentage


def plot_boxes_on_map(bboxes):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    for bbox in bboxes:
        lon_min, lat_min, lon_max, lat_max = bbox
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                color='red', linewidth=2, transform=ccrs.PlateCarree())

    ax.set_extent([-70, 70, -70, 70], crs=ccrs.PlateCarree())

    plt.title("Bounding Boxes")
    plt.savefig("bounding_boxes_map.png", dpi=300)
    bbox_size_km(bboxes[0], pixel_size_m=250)
