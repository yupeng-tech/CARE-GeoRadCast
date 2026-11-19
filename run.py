# Runs the 18-lead Australia surface solar radiation forecast.
# Usage: python run.py [--input PATH] [--output DIRECTORY]
# Version : CARE-GeoRadCast v1
# Author: Run Ma,Peng Yu

import argparse
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import torch
from pvlib import spa

sys.dont_write_bytecode = True

from model import CAREGeoRadCast
from product import write_product


ROOT = Path(__file__).resolve().parent
CHANNELS = ("SOZ", "SOA", "SAZ", "SAA") + tuple(
    [f"albedo_{i:02d}" for i in range(1, 7)]
    + [f"tbb_{i:02d}" for i in range(7, 17)]
)
ENCODED_CHANNELS = ("SOZ", "sin_SOA", "cos_SOA", "SAZ", "sin_SAA", "cos_SAA") + CHANNELS[4:]
HALO = 8
INPUT = ROOT / "inputs" / "H8_R21_Australia_20190810_0000-0150UTC.nc"
TIME_UNITS = "seconds since 1970-01-01 00:00:00 UTC"


def solar_at_times(target):
    utc = target.astype(np.float64)/1e9
    sidereal, alpha, delta = spa.solar_position_numpy(utc, 0., 0., 0., 1013.25, 12., 67., .5667, 1, sst=True)
    distance, = spa.solar_position_numpy(utc, 0., 0., 0., 1013.25, 12., 67., .5667, 1, esd=True)
    parallax = spa.equatorial_horizontal_parallax(distance)

    def geometry(latitude, longitude):
        lat, lon = latitude.astype(np.float64)[:, None], longitude.astype(np.float64)[:, None]
        local_hour = spa.local_hour_angle(sidereal, lon, alpha)
        uterm = spa.uterm(lat)
        xterm, yterm = spa.xterm(uterm, lat, 0.), spa.yterm(uterm, lat, 0.)
        d_alpha = spa.parallax_sun_right_ascension(xterm, parallax, local_hour, delta)
        delta_prime = spa.topocentric_sun_declination(delta, xterm, yterm, parallax, d_alpha, local_hour)
        hour_prime = spa.topocentric_local_hour_angle(local_hour, d_alpha)
        elevation = spa.topocentric_elevation_angle_without_atmosphere(lat, delta_prime, hour_prime)
        zenith = np.deg2rad(spa.topocentric_zenith_angle(elevation))
        gamma = spa.topocentric_astronomers_azimuth(hour_prime, delta_prime, lat)
        azi = np.deg2rad((spa.topocentric_azimuth_angle(gamma)+180.)%360.-180.)
        return np.stack((np.cos(zenith), np.sin(zenith)*np.sin(azi), np.sin(zenith)*np.cos(azi)),axis=-1).astype(np.float32)

    return geometry


def encode_tile(tile, device):
    encoded = np.empty((tile.shape[0],22,*tile.shape[2:]),np.float32)
    encoded[:,0], encoded[:,3], encoded[:,6:] = tile[:,0],tile[:,2],tile[:,4:]
    for source,target in ((1,1),(3,4)):
        radians = np.deg2rad(tile[:,source])
        encoded[:,target], encoded[:,target+1] = np.sin(radians),np.cos(radians)
    return torch.from_numpy(encoded).to(device)


def read_grid(ds):
    lat = np.ma.asarray(ds["latitude"][:]).filled(np.nan).astype(np.float32)
    lon = np.ma.asarray(ds["longitude"][:]).filled(np.nan).astype(np.float32)
    for axis, first, step, size in ((lat, -9.6, -0.05, 717), (lon, 109.6, 0.05, 917)):
        if axis.shape != (size,) or not np.allclose(axis, first + step * np.arange(size), rtol=0, atol=2e-5):
            raise ValueError("Expected the Australia grid with an eight-cell halo")
    clock = ds["time"]
    if clock.units != TIME_UNITS or np.ma.getmaskarray(clock[:]).any():
        raise ValueError("Invalid observation times")
    stamps = pd.to_datetime(np.asarray(clock[:]), unit="s", utc=True)
    expected = pd.date_range("2019-08-10", periods=12, freq="10min", tz="UTC")
    if not stamps.equals(expected):
        raise ValueError("Expected twelve observations from 00:00 to 01:50 UTC on 2019-08-10")
    for name in CHANNELS:
        if ds[name].dimensions != ("time", "latitude", "longitude") or ds[name].shape != (12, 717, 917):
            raise ValueError(f"Unexpected shape for {name}")
    return lat, lon, stamps


def read_tile(ds, row, stop, width):
    tile = np.empty((12, 20, stop - row + 2 * HALO, width), dtype=np.float32)
    for j, name in enumerate(CHANNELS):
        values = ds[name][:, row - HALO:stop + HALO, :]
        tile[:, j] = np.ma.asarray(values).astype(np.float32).filled(np.nan)
    return tile


def load_model(device):
    weights = torch.load(ROOT / "weights" / "care_georadcast.pt", map_location="cpu", weights_only=True)
    model = CAREGeoRadCast().to(device).eval()
    model.load_state_dict(weights, strict=True)
    model.temporal_encoder.lstm.flatten_parameters()
    with np.load(ROOT / "weights" / "normalization.npz", allow_pickle=False) as z:
        if tuple(z["channels"].tolist()) != ENCODED_CHANNELS:
            raise ValueError("Normalization channel order differs")
        mean = torch.as_tensor(z["x_mean"], dtype=torch.float32, device=device)[None, None, :, None, None]
        scale = torch.as_tensor(z["x_scale"], dtype=torch.float32, device=device)[None, None, :, None, None]
        y_mean, y_scale = float(z["y_mean"]), float(z["y_scale"])
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all() or not np.isfinite([y_mean, y_scale]).all() or y_scale <= 0:
        raise ValueError("Invalid normalization statistics")
    return model, mean, scale, y_mean, y_scale


def predict_tile(tile, latitude, longitude, geometry, model_data, device, batch_size):
    patch_valid = np.lib.stride_tricks.sliding_window_view(np.isfinite(tile).all(axis=(0, 1)), (17, 17))
    rows, cols = np.nonzero(patch_valid.all(axis=(-1, -2)))
    if not len(rows):
        return
    future = geometry(latitude[rows], longitude[cols])
    daytime = future[:, :, 0] > 0
    active = daytime.any(axis=1)
    rows, cols, future, daytime = rows[active], cols[active], future[active], daytime[active]
    if not len(rows):
        return
    model, mean, scale, target_mean, target_scale = model_data
    tile_tensor = (encode_tile(tile, device) - mean.reshape(1, 22, 1, 1)) / scale.reshape(1, 22, 1, 1)
    view = tile_tensor.unfold(2, 17, 1).unfold(3, 17, 1)
    tensor_rows, tensor_cols, future_tensor = (torch.from_numpy(a).to(device) for a in (rows, cols, future))
    for start in range(0, len(rows), batch_size):
        batch = slice(start, start + batch_size)
        patch = view[:, :, tensor_rows[batch], tensor_cols[batch]].permute(2, 0, 1, 3, 4).contiguous()
        pred = (model(patch, future_tensor[batch]) * target_scale + target_mean).cpu().numpy()
        if not np.isfinite(pred).all():
            raise FloatingPointError("Non-finite model output")
        yield rows[batch], cols[batch], np.where(daytime[batch], pred, np.nan).T


@torch.inference_mode()
def forecast(args):
    if min(args.batch_size, args.tile_rows, args.threads) < 1:
        raise ValueError("Batch size, tile rows and threads must be positive")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    model_data = load_model(device)
    begun = time.perf_counter()
    with netCDF4.Dataset(args.input) as ds:
        latitude, longitude, stamps = read_grid(ds)
        origin = stamps[-1]
        targets = origin.value + np.arange(1, 19, dtype=np.int64) * 600_000_000_000
        geometry = solar_at_times(targets)
        out_lat = latitude[HALO:-HALO]
        out_lon = longitude[HALO:-HALO]
        values = np.full((18, len(out_lat), len(out_lon)), np.nan, dtype=np.float32)
        args.output.mkdir(parents=True, exist_ok=False)
        path = args.output / f"CARE-GeoRadCast_SSR_forecast_Australia_{origin:%Y%m%d_%H%M}UTC.nc"
        print(f"Australia | {device} | origin {origin.isoformat()}", flush=True)
        for row in range(HALO, len(latitude) - HALO, args.tile_rows):
            stop = min(row + args.tile_rows, len(latitude) - HALO)
            tile = read_tile(ds, row, stop, len(longitude))
            for rows, cols, prediction in predict_tile(tile, latitude[row:stop], longitude[HALO:-HALO],
                                                       geometry, model_data, device, args.batch_size):
                values[:, row + rows - HALO, cols] = prediction
    write_product(values, path, out_lat, out_lon, origin, [pd.Timestamp(t, tz="UTC") for t in targets])
    print(f"Saved {path} ({time.perf_counter() - begun:.1f} s)", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="CARE-GeoRadCast 18-lead Australia forecast")
    parser.add_argument("--input", type=Path, default=INPUT, help="Australia input NetCDF")
    parser.add_argument("--output", type=Path, help="New output directory; default results or next unused results_N")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "results"
        index = 2
        while args.output.exists():
            args.output = ROOT / f"results_{index}"
            index += 1
    forecast(args)


if __name__ == "__main__":
    main()
