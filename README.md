
# CARE-GeoRadCast Demo

Source code for satellite-based solar radiation nowcasting.

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
