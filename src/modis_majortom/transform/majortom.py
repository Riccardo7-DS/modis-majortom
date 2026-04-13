import numpy as np

class CalculationsMajorTom():
    def __init__(self, pixel_size = 250):
        self.PIXEL_SIZE = pixel_size # meters
        self.TILE_SIZE = 1111950  # meters
        self.R = 6371007.181  # Earth radius for MODIS sinusoidal

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
