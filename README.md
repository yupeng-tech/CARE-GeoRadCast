
# CARE-GeoRadCast Demo

Source code for satellite-based solar radiation nowcasting.

## About CARE

CARE is an initiative focused on cloud remote sensing and its impacts on Earth's radiation budget. Its satellite datasets include cloud, aerosol, precipitation, water vapor, and radiation products.

CARE-GeoRadCast is one of the algorithms in the CARE family, focusing on multi-step surface solar radiation (SSR) nowcasting.

For more information, visit the [CARE official website](https://slrss.cn/care/).

## Satellite Applicability

The current algorithm is applicable to Himawari-8/9 observations. The included demo input is based on Himawari-8 data.

Future versions are planned to extend the algorithm to additional geostationary satellite observations, including FY-4A/B and GOES-R/ABI.

## Environment and Installation

Python 3.10. For CUDA 11.8, install the matching PyTorch wheel first:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Run

Run the forecast:

```bash
python run.py
```

Use `--device cpu` to run on CPU.

## Demo Input and Output

The bundled NetCDF file contains 12 Himawari-8 (H8) observations. The demo uses these observations to generate a short-term surface solar radiation (SSR) forecast at 0.05-degree spatial resolution.

The output in `results/` contains SSR values (W m-2), latitude, and longitude. The output shape is 701 x 901 x 18, covering forecast lead times from +10 to +180 minutes.

## Rebuild Input from Original H8 Files

To rebuild the input from the original H8 files:

```bash
python prepare_input.py H8_DIRECTORY --output inputs/rebuilt.nc
python run.py --input inputs/rebuilt.nc
```

## Project Structure

- `run.py`: Runs the forecast.
- `model.py`: Defines the model.
- `product.py`: Writes NetCDF.
- `prepare_input.py`: Crops and merges input data.
- `weights/`: Holds the model parameters and normalization.
- `inputs/`: Holds the sample NetCDF.

## Reference

Run Ma, Peng Yu, Huazhe Shang, Su Wang, Ziming Wang, Lesi Wei, Mingju Chen, Hongli Yan, and Husi Letu. "CARE-GeoRadCast: CARE-supervised multi-step surface solar radiation nowcasting from geostationary satellite observations." *Journal of Remote Sensing* (under review).
