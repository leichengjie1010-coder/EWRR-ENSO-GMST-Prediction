# A strong 2026/27 El Niño elevates 2027 record-warmth risk

## Overview

This repository contains the Python workflow used to prepare the predictor datasets, fit and validate the event-weighted ridge-regression (EWRR) models, generate the 2026/27 El Niño and 2027 global mean surface temperature (GMST) forecasts, and reproduce the main and supplementary figures. The workflow includes the primary ERSSTv6/CMST2.0 experiments and the SST sensitivity tests based on ERSSTv5, COBE2 and HadISST1.1.

## Requirements

- Python 3.10 or later
- NumPy 2.0 or later
- pandas 2.0 or later
- xarray 2024.0 or later
- Matplotlib 3.8 or later
- A NetCDF backend supported by xarray, such as `netCDF4` or `h5netcdf`

The scripts use only the Python standard library and the packages listed above. Package versions used for the archived analysis were Python 3.10, NumPy 2.2.6, pandas 2.3.3, xarray 2025.6.1 and Matplotlib 3.10.8.

## Data

The scripts expect the climate datasets described in the manuscript and Supplementary Table 5. By default, the canonical data archive is read from `/Users/leichengjie/Desktop/datas`, and model-ready project files are read from `/Users/leichengjie/Desktop/2026ENSO`. These locations can be changed without editing the source code:

```bash
export CODEX_DATA_ROOT=/path/to/climate/data
export ENSO_PROJECT_ROOT=/path/to/2026ENSO
```

Large input datasets are not distributed with this code. Their sources and download links are listed in the Data availability section of the manuscript.

## Usage

Run the scripts in numerical order from this directory.

1. Prepare the annual ENSO and GMST predictor tables:

   ```bash
   python 01_prepare_data.py
   ```

   To recompute the SST indices directly from the source NetCDF files instead of using the archived analysis-ready tables:

   ```bash
   python 01_prepare_data.py --recompute-sst --overwrite
   ```

2. Fit the EWRR models, perform leave-one-event-out validation, and generate the forecasts:

   ```bash
   python 02_fit_validate_models.py
   ```

3. Reproduce the manuscript and supplementary figures:

   ```bash
   python 03_make_figures.py
   ```

Unless alternative paths are supplied through the command-line options, processed tables and model outputs are written to `数据/code_reproduction/`, and figures are written to `图件/code_reproduction/`. Use `python <script_name> --help` to view all available options.

## Code organization

- `01_prepare_data.py` — data checks, anomaly calculation, regional averaging and construction of annual predictor tables.
- `02_fit_validate_models.py` — EWRR fitting, hyperparameter selection, leave-one-event-out validation, uncertainty estimation and final forecasting.
- `03_make_figures.py` — generation of the main and supplementary figures from the archived model outputs.

## Citation

If you use this code, please cite:

> *A strong 2026/27 El Niño elevates 2027 record-warmth risk*. Manuscript under review.

The full bibliographic citation and persistent repository identifier will be added upon publication and archival release.
