# Writes packed surface solar radiation forecasts to NetCDF.
# Usage: Imported by run.py.
# Version : CARE-GeoRadCast v1
# Author: Run Ma,Peng Yu

from pathlib import Path

import netCDF4
import numpy as np


def write_product(values, destination, latitude, longitude, origin, targets):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".partial.nc")
    if temporary.exists():
        raise FileExistsError(temporary)
    minimum, maximum = np.inf, -np.inf
    for row in range(0, len(latitude), 32):
        block = values[:, row:row + 32, :]
        finite = block[np.isfinite(block)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
    if not np.isfinite([minimum, maximum]).all():
        minimum = maximum = 0.0
    scale = max(0.01, np.ceil((maximum - minimum) / 65534 * 100) / 100)
    offset = (minimum + maximum) / 2
    with netCDF4.Dataset(temporary, "w", format="NETCDF4", clobber=False) as dst:
        dst.model = "CARE-GeoRadCast"
        dst.forecast_reference_time_utc = origin.isoformat()
        dst.forecast_leads_minutes = np.arange(10, 181, 10, dtype=np.int16)
        dst.forecast_valid_time_range_utc = f"{targets[0].isoformat()} / {targets[-1].isoformat()}"
        for name, length in (("y", len(latitude)), ("x", len(longitude)), ("lead", 18)):
            dst.createDimension(name, length)
        for name, dim, data, units in (("latitude", "y", latitude, "degrees_north"),
                                        ("longitude", "x", longitude, "degrees_east")):
            coordinate = dst.createVariable(name, "f4", (dim,))
            coordinate.units = units
            coordinate[:] = data
        forecast = dst.createVariable("ssr_forecast", "i2", ("y", "x", "lead"),
                                       fill_value=np.int16(-32768), zlib=True,
                                       complevel=4, shuffle=True,
                                       chunksizes=(min(64, len(latitude)), min(64, len(longitude)), 1))
        forecast.units = "W m-2"
        forecast.coordinates = "latitude longitude"
        forecast.scale_factor = np.float64(scale)
        forecast.add_offset = np.float64(offset)
        forecast.set_auto_maskandscale(False)
        for row in range(0, len(latitude), 32):
            block = values[:, row:row + 32, :].transpose(1, 2, 0)
            valid = np.isfinite(block)
            codes = np.rint((block[valid].astype(np.float64) - offset) / scale)
            if np.any((codes < -32767) | (codes > 32767)):
                raise OverflowError("Prediction exceeds the int16 packing range")
            packed = np.full(block.shape, -32768, dtype=np.int16)
            packed[valid] = codes.astype(np.int16)
            forecast[row:row + 32, :, :] = packed
    temporary.rename(destination)
