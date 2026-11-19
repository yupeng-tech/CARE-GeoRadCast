CARE-GeoRadCast  demo

Python 3.10. For CUDA 11.8, install the matching PyTorch wheel first:
  python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
  python -m pip install -r requirements.txt
  python run.py
Use --device cpu to run on CPU.

The bundled NetCDF contains 12 H8 observations from 2019-08-10 00:00 to
01:50 UTC. The forecast covers Australia and nearby seas (110-155 E,
45-10 S) at 0.05 degrees. The output in results/ contains SSR in W m-2,
latitude, and longitude. Shape: 701 x 901 x 18, from +10 to +180 minutes.

To rebuild the input from the original H8 files:
  python prepare_input.py H8_DIRECTORY --output inputs/rebuilt.nc
  python run.py --input inputs/rebuilt.nc

run.py runs the forecast, model.py defines the model, product.py writes
NetCDF, and prepare_input.py crops and merges input data. The weights/
folder holds the model parameters and normalization. The inputs/ folder
holds the sample NetCDF.
