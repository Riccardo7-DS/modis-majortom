import numpy as np

class CalculationsMajorTom():
    TILE_SIZE = 1111950  # metres per MODIS sinusoidal tile
    R = 6371007.181      # Earth radius used by MODIS sinusoidal projection

    # MODIS actual pixels-per-tile for each nominal resolution label.
    # Actual pixel size = TILE_SIZE / pixels_per_tile  (NOT the nominal label).
    # e.g. "250 m" product → 4800 px/tile → 231.65625 m/px (not 250 m).
    _PIXELS_PER_TILE = {250: 4800, 500: 2400, 1000: 1200}

    def __init__(self, pixel_size=250):
        n = self._PIXELS_PER_TILE.get(int(pixel_size))
        if n is not None:
            self.PIXEL_SIZE = self.TILE_SIZE / n   # actual pixel size in metres
        else:
            self.PIXEL_SIZE = float(pixel_size)

    def latlon_to_sinu(self, lat, lon):
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        x = self.R * lon * np.cos(lat)
        y = self.R * lat
        return x, y

    def sinu_to_tile(self, x, y):
        h = int((x + 20015109) // self.TILE_SIZE)
        v = int((10007555 - y) // self.TILE_SIZE)
        return h, v

    def tile_origin(self, h, v):
        x0 = -20015109 + h * self.TILE_SIZE
        y0 =  10007555 - v * self.TILE_SIZE
        return x0, y0

    def xy_to_pixel(self, x, y, h, v):
        x0, y0 = self.tile_origin(h, v)
        col = int((x - x0) / self.PIXEL_SIZE)
        row = int((y0 - y) / self.PIXEL_SIZE)
        return row, col
