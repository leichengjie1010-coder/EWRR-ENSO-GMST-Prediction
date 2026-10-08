#!/usr/bin/env python3
"""Prepare predictor tables used by the ENSO--GMST forecast experiments.

The script reads the canonical local climate archive, applies the definitions
reported in Methods, and writes compact annual tables for the modelling step.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# Paths are intentionally supplied by the user or the execution environment.
# No machine-specific data address is embedded in the reproducibility code.
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT") or os.environ.get("CODEX_DATA_ROOT")
DEFAULT_OUTPUT = Path(os.environ.get("PREPARED_OUTPUT", "prepared"))
YEARS = np.arange(1982, 2027)
CLIMATOLOGY = (1991, 2020)

SST_FILENAMES = {
    "ersstv6": "ersstv6_monthly_sst_1981_2026_60S70N.nc",
    "ersstv5": "ersstv5_monthly_sst_1981_2026_60S70N.nc",
    "cobe2": "cobe2_sst_1981_2026_60S70N.nc",
    "hadisst": "hadisst_monthly_sst_1981_2026_60S70N.nc",
}
CURATED_FILENAMES = {
    "ersstv6": "enso_predictors_1982_2026_ersstv6.csv",
    "ersstv5": "enso_predictors_1982_2026_ersstv5.csv",
    "cobe2": "enso_predictors_1982_2026_cobe2_user_full.csv",
    "hadisst": "enso_predictors_1982_2026_hadisst.csv",
}
PREVIOUS_1981 = {
    "ersstv6": -0.12113380432128906,
    "ersstv5": -0.0798861161,
    "cobe2": -0.28676095604896545,
    "hadisst": -0.05189259722828865,
}
# These two values were present in an old prepared table.  They are retained
# only as a regression guard: a current run must either replace them from the
# supplied 2027 forecast-input JSON or fail before writing model inputs.
STALE_FORECAST_INPUTS_2027 = {
    "IPO_TPI_LAG1": -0.388260782,
    "GLOBAL_SST_ERSSTv6_LAG1": 0.446584880,
}

SST_BOXES = {
    "NINO12": (-10.0, 0.0, 270.0, 280.0),
    "NINO3": (-5.0, 5.0, 210.0, 270.0),
    "NINO34": (-5.0, 5.0, 190.0, 240.0),
    "NINO4": (-5.0, 5.0, 160.0, 210.0),
    "EMI_C": (-10.0, 10.0, 165.0, 220.0),
    "EMI_E": (-15.0, 5.0, 250.0, 290.0),
    "EMI_W": (-10.0, 20.0, 125.0, 145.0),
    "IOBM": (-20.0, 20.0, 40.0, 110.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(DEFAULT_DATA_ROOT) if DEFAULT_DATA_ROOT else None,
        help="Canonical climate-data root; may also be supplied through DATA_ROOT.",
    )
    parser.add_argument(
        "--base-frame", type=Path, required=True,
        help="Analysis-ready ENSO predictor table used as the base frame.",
    )
    parser.add_argument(
        "--gmst-frame", type=Path, required=True,
        help="Annual GMST predictor table before the ERSSTv6 Niño3.4 replacement.",
    )
    parser.add_argument(
        "--curated-dir", type=Path, default=None,
        help="Directory containing the four archived ENSO predictor tables.",
    )
    parser.add_argument(
        "--gmst-source", type=Path, default=None,
        help="Optional native GMST series used to calculate the reporting-baseline offset.",
    )
    parser.add_argument(
        "--forecast-inputs", type=Path, default=None,
        help=("JSON with the corrected 2027 annual_hybrid_tpi and "
              "annual_hybrid_global values; required when the prepared frame "
              "contains the retired placeholder values."),
    )
    parser.add_argument(
        "--reporting-offset", type=float, default=None,
        help="Optional audited 1850-1900 reporting-baseline offset in degrees C.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recompute-sst",
        action="store_true",
        help="recompute SST indices from NetCDF instead of using the archived analysis-ready tables",
    )
    return parser.parse_args()


def normalise_coordinates(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for old, new in (("latitude", "lat"), ("longitude", "lon")):
        if old in da.dims or old in da.coords:
            rename[old] = new
    if rename:
        da = da.rename(rename)
    if "lon" in da.coords:
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    if "lat" in da.coords:
        da = da.sortby("lat")
    return da


def find_variable(ds: xr.Dataset, preferred: tuple[str, ...]) -> xr.DataArray:
    for name in preferred:
        if name in ds.data_vars:
            return normalise_coordinates(ds[name])
    candidates = [name for name, value in ds.data_vars.items() if "time" in value.dims]
    if len(candidates) != 1:
        raise KeyError(f"Cannot identify variable in dataset; candidates={candidates}")
    return normalise_coordinates(ds[candidates[0]])


def subset(da: xr.DataArray, lat1: float, lat2: float, lon1: float, lon2: float) -> xr.DataArray:
    da = normalise_coordinates(da)
    return da.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2))


def area_mean(da: xr.DataArray, box: tuple[float, float, float, float]) -> xr.DataArray:
    field = subset(da, *box)
    return field.weighted(np.cos(np.deg2rad(field.lat))).mean(("lat", "lon"), skipna=True)


def monthly_anomaly(da: xr.DataArray) -> xr.DataArray:
    start, end = CLIMATOLOGY
    base = da.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    return da.groupby("time.month") - base.groupby("time.month").mean("time", skipna=True)


def annual_mean(series: xr.DataArray, months: tuple[int, ...], name: str) -> pd.Series:
    selected = series.where(series.time.dt.month.isin(months), drop=True)
    result = selected.groupby("time.year").mean("time", skipna=True).to_series()
    return result.reindex(YEARS).rename(name)


def d0jf(series: xr.DataArray, years: np.ndarray, name: str) -> pd.Series:
    values = series.to_series()
    result = pd.Series(index=years, dtype=float, name=name)
    for year in years:
        dates = [pd.Timestamp(year, 12, 1), pd.Timestamp(year + 1, 1, 1), pd.Timestamp(year + 1, 2, 1)]
        if all(date in values.index for date in dates):
            result.loc[year] = float(values.loc[dates].mean())
    return result


def zscore_reference(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    start, end = CLIMATOLOGY
    mask = (years >= start) & (years <= end) & np.isfinite(values)
    return (values - np.nanmean(values[mask])) / np.nanstd(values[mask], ddof=1)


def eof1(x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = (x_train @ x_train.T) / max(1, x_train.shape[0] - 1)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    u = eigenvector[:, -1]
    singular = np.sqrt(max(eigenvalue[-1], 0.0) * max(1, x_train.shape[0] - 1))
    loading = x_train.T @ u / max(singular, 1e-12)
    return x_all @ loading, loading


def sst_scalar_indices(sst: xr.DataArray) -> tuple[pd.DataFrame, xr.DataArray, float]:
    regional = {key: monthly_anomaly(area_mean(sst, box).compute()) for key, box in SST_BOXES.items()}
    emi = regional["EMI_C"] - 0.5 * regional["EMI_E"] - 0.5 * regional["EMI_W"]
    out: dict[str, pd.Series] = {}
    for key in ("NINO12", "NINO3", "NINO34", "NINO4"):
        values = regional[key]
        march = annual_mean(values, (3,), "march")
        may = annual_mean(values, (5,), "may")
        out[f"{key}_MAM"] = annual_mean(values, (3, 4, 5), f"{key}_MAM")
        out[f"{key}_TREND_MAM"] = (may - march).rename(f"{key}_TREND_MAM")
        out[f"{key}_D0JF"] = d0jf(values, YEARS, f"{key}_D0JF")
    out["EMI_TREND_MAM"] = (
        annual_mean(emi, (5,), "may") - annual_mean(emi, (3,), "march")
    ).rename("EMI_TREND_MAM")
    out["IOBM_MAM"] = annual_mean(regional["IOBM"], (3, 4, 5), "IOBM_MAM")
    previous = float(d0jf(regional["NINO34"], np.array([1981]), "n34").loc[1981])
    return pd.DataFrame(out, index=YEARS), regional["NINO34"], previous


def pmm_index(sst: xr.DataArray, nino34: xr.DataArray, atmosphere: Path) -> pd.Series:
    sst_field = subset(sst, -21, 32, 175, 265)
    with xr.open_dataset(atmosphere, engine="netcdf4", chunks={"time": 12}) as ds:
        u = subset(find_variable(ds[["u10"]], ("u10",)), -21, 32, 175, 265).interp(lat=sst_field.lat, lon=sst_field.lon)
        v = subset(find_variable(ds[["v10"]], ("v10",)), -21, 32, 175, 265).interp(lat=sst_field.lat, lon=sst_field.lon)
        xs = monthly_anomaly(sst_field).where(sst_field.time.dt.month.isin((3, 4, 5)), drop=True)
        xu, xv = monthly_anomaly(u), monthly_anomaly(v)
        common = np.intersect1d(xs.time.values, xu.time.values)
        xs, xu, xv = xs.sel(time=common).load(), xu.sel(time=common).load(), xv.sel(time=common).load()
    times = pd.DatetimeIndex(common)
    cti = nino34.sel(time=common).values.astype(float)
    train = (times.year >= 1982) & (times.year <= 2025)

    def prepare(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
        values = field.values.reshape(len(field.time), -1).astype(float)
        good = np.isfinite(values[train]).mean(axis=0) > 0.98
        values = values[:, good]
        mean = np.nanmean(values[train], axis=0)
        values = np.where(np.isfinite(values), values, mean)
        c0 = cti[train] - np.mean(cti[train])
        beta = (c0[:, None] * (values[train] - mean)).sum(axis=0) / np.sum(c0**2)
        residual = values - mean - (cti - np.mean(cti[train]))[:, None] * beta
        lat = np.repeat(field.lat.values[:, None], len(field.lon), axis=1).ravel()[good]
        residual *= np.sqrt(np.cos(np.deg2rad(lat)))[None, :]
        scale = np.sqrt(np.nanmean(np.nanvar(residual[train], axis=0, ddof=1)))
        return residual / scale, good

    sst_matrix, good = prepare(xs)
    u_matrix, _ = prepare(xu)
    v_matrix, _ = prepare(xv)
    wind_matrix = np.concatenate([u_matrix, v_matrix], axis=1) / np.sqrt(2.0)
    qs, rs = np.linalg.qr(sst_matrix[train].T, mode="reduced")
    qw, rw = np.linalg.qr(wind_matrix[train].T, mode="reduced")
    left, _, _ = np.linalg.svd((rs @ rw.T) / (train.sum() - 1), full_matrices=False)
    pattern = qs @ left[:, 0]
    pc = sst_matrix @ pattern
    flat_lat = np.repeat(xs.lat.values[:, None], len(xs.lon), axis=1).ravel()[good]
    flat_lon = np.repeat(xs.lon.values[None, :], len(xs.lat), axis=0).ravel()[good]
    northeast = (flat_lat >= 10) & (flat_lat <= 25) & (flat_lon >= 200) & (flat_lon <= 250)
    if np.nanmean(pattern[northeast] / np.sqrt(np.cos(np.deg2rad(flat_lat[northeast])))) < 0:
        pc *= -1
    pc = zscore_reference(pc, times.year)
    return pd.Series(pc, index=times).groupby(times.year).mean().reindex(YEARS).rename("PMM_MAM")


def pdo_index(sst: xr.DataArray) -> pd.Series:
    field = subset(sst, 20, 70, 110, 260)
    anomaly = monthly_anomaly(field).where(field.time.dt.month.isin((3, 4, 5)), drop=True)
    global_mean = monthly_anomaly(area_mean(sst, (-60, 60, 0, 358)).compute())
    anomaly = (anomaly - global_mean.sel(time=anomaly.time)).sel(time=slice("1982-03-01", "2026-05-31")).load()
    times = pd.DatetimeIndex(anomaly.time.values)
    values = anomaly.values.reshape(len(times), -1).astype(float)
    train = times.year <= 2025
    good = np.isfinite(values[train]).mean(axis=0) > 0.98
    values = values[:, good]
    mean = np.nanmean(values[train], axis=0)
    values = np.where(np.isfinite(values), values, mean) - mean
    lat = np.repeat(anomaly.lat.values[:, None], len(anomaly.lon), axis=1).ravel()[good]
    lon = np.repeat(anomaly.lon.values[None, :], len(anomaly.lat), axis=0).ravel()[good]
    weighted = values * np.sqrt(np.cos(np.deg2rad(lat)))[None, :]
    pc, loading = eof1(weighted[train], weighted)
    east = (lat >= 25) & (lat <= 60) & (lon >= 210) & (lon <= 260)
    if np.nanmean(loading[east] / np.sqrt(np.cos(np.deg2rad(lat[east])))) < 0:
        pc *= -1
    pc = zscore_reference(pc, times.year)
    return pd.Series(pc, index=times).groupby(times.year).mean().reindex(YEARS).rename("PDO_MAM")


def cold_memory(target: pd.Series, previous_1981: float) -> pd.Series:
    values = target.copy()
    values.loc[1981] = previous_1981
    duration = pd.Series(0.0, index=YEARS, name="PREV_LANINA_DURATION")
    for year in YEARS:
        previous, count = year - 1, 0
        while previous in values.index and pd.notna(values.loc[previous]) and values.loc[previous] <= -0.5:
            count += 1
            previous -= 1
        duration.loc[year] = count
    return duration


def build_sst_frame(name: str, path: Path, base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logging.info("Processing %s", name)
    with xr.open_dataset(path, engine="netcdf4", chunks={"time": 12}) as ds:
        sst = find_variable(ds, ("sst", "sea_surface_temperature", "tos"))
        if pd.Timestamp(sst.time.values[-1]) < pd.Timestamp("2026-05-01"):
            raise ValueError(f"{name} does not reach May 2026: {sst.time.values[-1]}")
        scalar, nino34, previous_1981 = sst_scalar_indices(sst)
        scalar["PDO_MAM"] = pdo_index(sst)

    frame = base.copy()
    for column in scalar:
        frame[column] = scalar[column]
    frame["ELNINO_D0JF"] = np.where(frame.NINO34_D0JF.notna(), frame.NINO34_D0JF >= 0.5, np.nan)
    frame["PREV_LANINA_DURATION"] = cold_memory(frame.NINO34_D0JF, previous_1981)
    metadata = {
        "dataset": name,
        "source": str(path),
        "climatology": "1991-2020 monthly",
        "years": "1982-2026",
        "previous_1981_D0JF": previous_1981,
    }
    return frame, metadata


def prepare_gmst_frame(
    output: Path, gmst_frame_path: Path, enso_frame: pd.DataFrame,
    previous_1981: float, gmst_source: Path | None = None,
    forecast_inputs: dict[str, float] | None = None,
    reporting_offset_override: float | None = None,
) -> dict:
    frame = pd.read_csv(gmst_frame_path).set_index("year").sort_index()
    required = [
        "GMST_LAG1", "NINO34_DJF_ENDING_YEAR", "IPO_TPI_LAG1",
        "GLOBAL_SST_ERSSTv6_LAG1", "ERF_WMGHG_LAG1", "DELTA_CMST2_GMST", "CMST2_GMST",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"GMST model frame is missing {missing}")
    # The GMST model must use the same ERSSTv6 Niño3.4 series as the ENSO
    # model.  Year y uses the previous winter's D(0)JF value.
    nino = enso_frame["NINO34_D0JF"].astype(float)
    if 1982 in frame.index:
        frame.loc[1982, "NINO34_DJF_ENDING_YEAR"] = float(previous_1981)
    for year in range(1983, 2027):
        if year in frame.index and year - 1 in nino.index:
            frame.loc[year, "NINO34_DJF_ENDING_YEAR"] = float(nino.loc[year - 1])
    stale = {
        key: float(frame.loc[2027, key])
        for key in STALE_FORECAST_INPUTS_2027
        if 2027 in frame.index and key in frame.columns
    }
    if forecast_inputs is not None:
        required_inputs = {"annual_hybrid_tpi", "annual_hybrid_global"}
        missing_inputs = required_inputs - set(forecast_inputs)
        if missing_inputs:
            raise KeyError(f"Forecast-input JSON is missing {sorted(missing_inputs)}.")
        if 2027 not in frame.index:
            raise KeyError("GMST model frame must contain year 2027 for forecast inputs.")
        corrected = {
            "IPO_TPI_LAG1": float(forecast_inputs["annual_hybrid_tpi"]),
            "GLOBAL_SST_ERSSTv6_LAG1": float(forecast_inputs["annual_hybrid_global"]),
        }
        if not all(np.isfinite(value) for value in corrected.values()):
            raise ValueError("2027 forecast inputs must be finite.")
        frame.loc[2027, list(corrected)] = list(corrected.values())
    current = {
        key: float(frame.loc[2027, key])
        for key in STALE_FORECAST_INPUTS_2027
        if 2027 in frame.index and key in frame.columns
    }
    for key, old_value in STALE_FORECAST_INPUTS_2027.items():
        if key in current and np.isclose(current[key], old_value, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Retired 2027 placeholder remains in {key}: {old_value}. "
                "Pass --forecast-inputs with the corrected hybrid values."
            )
    frame[required].reset_index().to_csv(
        output / "gmst_annual_model_frame.csv", index=False, float_format="%.17g"
    )

    reporting_offset = 0.0
    if gmst_source is not None:
        native = pd.read_csv(gmst_source).set_index("year").sort_index()
        reporting_offset = float(
            native.loc[1991:2020, "CMST2_GMST"].mean()
            - native.loc[1850:1900, "CMST2_GMST"].mean()
        )
    if reporting_offset_override is not None:
        if not np.isfinite(reporting_offset_override):
            raise ValueError("--reporting-offset must be finite.")
        reporting_offset = float(reporting_offset_override)
    return {
        "computational_baseline": "1991-2020",
        "reporting_baseline": "1850-1900",
        "reporting_offset_degC": reporting_offset,
        "gmst_frame": str(gmst_frame_path),
        "gmst_training_years": "1982-2025 (44 years)",
        "nino34_source_for_gmst": "ERSSTv6 D(0)JF series from the prepared ENSO frame",
        "forecast_method": "deterministic point forecasts; empirical Monte Carlo is used only for intervals and exceedance probabilities",
        "enso_error_pool": "15 completed El Nino target winters",
        "gmst_2027_uncertainty": "six independently resampled upstream-error pools; the ENSO pool contains 15 El Nino winters",
        "monte_carlo_draws": 200000,
        "forecast_inputs_2027": forecast_inputs,
        "retired_placeholder_values_seen_before_replacement": stale,
        "retired_placeholder_values_used": False,
    }


def validate_sources(
    data_root: Path | None, base_frame: Path, gmst_frame: Path,
    curated_dir: Path, require_sst: bool = False,
) -> dict[str, Path]:
    if data_root is None:
        raise ValueError("Set DATA_ROOT or pass --data-root before running the preparation step.")
    index = data_root / "DATA_INDEX.csv"
    if not index.exists():
        index = data_root / "DATA_INDEX.json"
    if not index.exists():
        raise FileNotFoundError(f"Canonical data index not found below {data_root}")
    paths = {name: data_root / "sst" / filename for name, filename in SST_FILENAMES.items()}
    paths.update({"annual_precursors": base_frame, "gmst_frame": gmst_frame})
    paths.update({f"curated_{name}": curated_dir / filename for name, filename in CURATED_FILENAMES.items()})
    # The current reproducibility run uses the archived, analysis-ready
    # predictor tables. NetCDF SST files are required only when the optional
    # --recompute-sst path is requested; this avoids failing on a stale alias
    # for an SST version that is not needed by the default workflow.
    if require_sst:
        candidates = [paths["annual_precursors"], paths["gmst_frame"],
                      *[paths[name] for name in SST_FILENAMES]]
    else:
        candidates = [paths["annual_precursors"], paths["gmst_frame"],
                      *[paths[f"curated_{name}"] for name in CURATED_FILENAMES]]
    absent = [str(path) for path in candidates if not path.exists()]
    if absent:
        raise FileNotFoundError("Missing canonical input files:\n" + "\n".join(absent))
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return path.name


def write_run_manifest(output: Path, input_files: list[Path | None], config: dict) -> None:
    files = [Path(__file__).resolve(), *(Path(path) for path in input_files if path is not None)]
    hashes = {
        portable_path(path): sha256_file(path)
        for path in files if path.exists() and path.is_file()
    }
    manifest = {
        "manifest_version": "1.0",
        "result_version": "CI15_prepared_inputs",
        "source_code": portable_path(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "output_directory": portable_path(output),
        "input_file_hashes": hashes,
        "configuration": config,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    curated_dir = args.curated_dir or args.base_frame.parent
    sources = validate_sources(
        args.data_root, args.base_frame, args.gmst_frame, curated_dir,
        require_sst=args.recompute_sst,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_frame).set_index("year").sort_index()
    metadata = {}
    forecast_inputs = None
    if args.forecast_inputs is not None:
        forecast_inputs = json.loads(args.forecast_inputs.read_text(encoding="utf-8"))
    for name in SST_FILENAMES:
        destination = args.output / f"enso_predictors_{name}.csv"
        if destination.exists() and not args.overwrite:
            logging.info("Keeping existing %s", destination.name)
            continue
        if args.recompute_sst:
            frame, info = build_sst_frame(name, sources[name], base)
        else:
            frame = pd.read_csv(sources[f"curated_{name}"]).set_index("year").sort_index()
            info = {
                "dataset": name,
                "source": str(sources[name]),
                "analysis_ready_table": str(sources[f"curated_{name}"]),
                "climatology": "1991-2020 monthly",
                "years": "1982-2026",
                "previous_1981_D0JF": PREVIOUS_1981[name],
            }
        frame.to_csv(destination, float_format="%.9g")
        metadata[name] = info
    ersstv6_frame = pd.read_csv(args.output / "enso_predictors_ersstv6.csv").set_index("year").sort_index()
    gmst_metadata = prepare_gmst_frame(
        args.output, args.gmst_frame, ersstv6_frame,
        PREVIOUS_1981["ersstv6"], args.gmst_source, forecast_inputs,
        args.reporting_offset,
    )
    metadata["_global"] = gmst_metadata
    (args.output / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_run_manifest(
        args.output,
        [args.base_frame, args.gmst_frame, args.gmst_source, args.forecast_inputs,
         *[sources[f"curated_{name}"] for name in CURATED_FILENAMES]],
        {
            "climatology": "1991-2020",
            "years": [1982, 2026],
            "forecast_inputs_2027": forecast_inputs,
            "reporting_offset_override": args.reporting_offset,
            "retired_placeholder_guard": STALE_FORECAST_INPUTS_2027,
        },
    )
    logging.info("Prepared tables written to %s", args.output)


if __name__ == "__main__":
    main()
