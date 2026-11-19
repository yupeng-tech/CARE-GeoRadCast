# Crops and combines the Australia case input.
# Usage: python prepare_input.py H8_DIRECTORY --output inputs/rebuilt.nc
# Version : CARE-GeoRadCast v1
# Author: Run Ma,Peng Yu

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

from run import CHANNELS, HALO, INPUT, TIME_UNITS, read_grid


def main():
    parser = argparse.ArgumentParser(description="Build the Australia test input from original H8 files")
    parser.add_argument("source", type=Path, help="Directory holding the original H8 observations")
    parser.add_argument("--output", type=Path, default=INPUT)
    args = parser.parse_args()
    times = pd.date_range("2019-08-10", periods=12, freq="10min", tz="UTC")
    paths = []
    for t in times:
        name = f"NC_H08_{t:%Y%m%d_%H%M}_R21_FLDK.02401_02401.nc"
        matches = list(args.source.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one source file for {name}, found {len(matches)}")
        paths.append(matches[0])
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.nc")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)

    with ExitStack() as stack:
        inputs = [stack.enter_context(netCDF4.Dataset(p)) for p in paths]
        latitude, longitude = (inputs[0][name][:] for name in ("latitude", "longitude"))
        y = np.flatnonzero((latitude >= -45) & (latitude <= -10))
        x = np.flatnonzero((longitude >= 110) & (longitude <= 155))
        ys, xs = (slice(ids[0] - HALO, ids[-1] + HALO + 1) for ids in (y, x))
        with netCDF4.Dataset(temporary, "w", format="NETCDF4", clobber=False) as out:
            out.title = "Himawari-8 R21 FLDK Australia test input"
            out.source_files = "\n".join(p.name for p in paths)
            out.forecast_domain = "110 to 155 degrees east, 45 to 10 degrees south"
            out.input_halo_grid_cells = HALO
            for name, values, section in (("latitude", latitude, ys), ("longitude", longitude, xs)):
                out.createDimension(name, section.stop - section.start)
                v = out.createVariable(name, "f4", (name,))
                v.units = inputs[0][name].units
                v[:] = values[section]
            out.createDimension("time", len(times))
            clock = out.createVariable("time", "i8", ("time",))
            clock.units = TIME_UNITS
            clock[:] = times.asi8 // 1_000_000_000
            for name in CHANNELS:
                first = inputs[0][name]
                attrs = {k: first.getncattr(k) for k in first.ncattrs()}
                var = out.createVariable(name, first.dtype, ("time", "latitude", "longitude"),
                                         fill_value=attrs.get("_FillValue"), zlib=True, complevel=4,
                                         chunksizes=(1, 64, len(out.dimensions["longitude"])))
                var.setncatts({k: value for k, value in attrs.items() if k != "_FillValue"})
                var.set_auto_maskandscale(False)
                for i, ds in enumerate(inputs):
                    src = ds[name]
                    if ds.id != paths[i].name or src.dimensions != ("latitude", "longitude") or src.dtype != first.dtype:
                        raise ValueError(f"Unexpected source variable {name} in {paths[i]}")
                    current = {k: src.getncattr(k) for k in src.ncattrs()}
                    if attrs.keys() != current.keys() or any(not np.array_equal(attrs[k], current[k]) for k in attrs):
                        raise ValueError(f"Source packing differs for {name} in {paths[i]}")
                    src.set_auto_maskandscale(False)
                    var[i] = src[ys, xs]
                var.set_auto_maskandscale(True)
            read_grid(out)
        for ds in inputs:
            if not np.array_equal(ds["latitude"][:], latitude) or not np.array_equal(ds["longitude"][:], longitude):
                raise ValueError("Source grid changed across the 12 observations")
        with netCDF4.Dataset(temporary) as out:
            for name in CHANNELS:
                saved = out[name]
                saved.set_auto_maskandscale(False)
                for i, ds in enumerate(inputs):
                    if not np.array_equal(saved[i], ds[name][ys, xs]):
                        raise ValueError(f"Raw crop differs from source: {name}, frame {i + 1}")
    temporary.rename(path)
    print(f"Verified 12 frames x 20 channels of raw H8 data: {path}")


if __name__ == "__main__":
    main()
