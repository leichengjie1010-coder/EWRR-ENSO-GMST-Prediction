#!/usr/bin/env python3
"""Fit the current nested-CV ENSO--GMST model.

The implementation follows the current manuscript configuration:

* 44 training years (1982--2025) for both targets;
* ENSO origin years are weighted for the Niño3.4 model;
* El Niño decay years (origin year + 1) are weighted for the GMST model;
* event-weight candidates are 1.00--5.95 in steps of 0.05;
* ridge candidates are 0 plus 99 logarithmically spaced values from 0.001 to 1;
* the fixed selection score is
  0.3(1-r_all) + 0.3(1-r_event) +
  0.2 RMSE_all/SD_all + 0.2 RMSE_event/SD_event;
* every target year selects its own weight and penalty inside nested CV;
* after minimizing the composite score, a selected candidate with CI >= 15
  keeps its weight and is moved upward along the fixed lambda grid until CI < 15;
* point forecasts are deterministic; Monte Carlo is used only for intervals
  and exceedance probabilities.

All input and output locations are supplied through arguments. No machine-
specific path is embedded in this file.
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


TRAIN_YEARS = np.arange(1982, 2026)
ENSO_EVENT_YEARS = np.array(
    [1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004,
     2006, 2009, 2014, 2015, 2018, 2019, 2023], dtype=int,
)
GMST_DECAY_YEARS = ENSO_EVENT_YEARS + 1
ENSO_FEATURES = [
    "WWB_INT", "SSH_RECHARGE_MAM", "PMM_MAM", "IOBM_MAM", "PDO_MAM",
    "NINO34_MAM", "SSH_TREND_MAM", "WWB_INT_MAY", "NINO12_TREND_MAM",
    "NINO3_TREND_MAM", "EMI_TREND_MAM", "D20_EW_GRAD_TREND_MAM",
    "PREV_LANINA_DURATION", "WWB_TIMING_MAM",
]
GMST_FEATURES = [
    "GMST_LAG1", "NINO34_DJF_ENDING_YEAR", "IPO_TPI_LAG1",
    "GLOBAL_SST_ERSSTv6_LAG1", "ERF_WMGHG_LAG1",
]

# 100 equally spaced values: 1.00, 1.05, ..., 5.95.
EVENT_WEIGHTS = np.round(1.0 + 0.05 * np.arange(100), 2)
# 100 values including zero: zero plus 99 logarithmically spaced positives.
RIDGE_LAMBDAS = np.r_[0.0, np.geomspace(0.001, 1.0, 99)]
GRID = np.array([(weight, penalty) for penalty in RIDGE_LAMBDAS for weight in EVENT_WEIGHTS])
SCORE_WEIGHTS = np.array([0.3, 0.3, 0.2, 0.2])
CONDITION_INDEX_LIMIT = 15.0
CONDITION_PENALTY_SCALE = 1.0e6
MC_DRAWS = 200_000
MC_SEED = 20260917
STALE_FORECAST_INPUTS_2027 = {
    "IPO_TPI_LAG1": -0.388260782,
    "GLOBAL_SST_ERSSTv6_LAG1": 0.446584880,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory written by 01_prepare_data.py.")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--joint-errors", type=Path, default=None,
                        help="Upstream-error table for TPI and global SST.")
    parser.add_argument("--erf-errors", type=Path, default=None,
                        help="Optional ERF residual table with an 'error' column.")
    parser.add_argument(
        "--enso-datasets", nargs="+",
        choices=("ersstv6", "ersstv5", "cobe2", "hadisst"),
        default=("ersstv6", "ersstv5", "cobe2", "hadisst"),
        help="ENSO SST datasets to evaluate; ERSSTv6 remains the GMST input.",
    )
    parser.add_argument(
        "--condition-index-limit", type=float, default=15.0,
        help="Upper bound for the regularized condition index (default: 15).",
    )
    parser.add_argument(
        "--disable-condition-index", action="store_true",
        help="Disable the condition-index feasibility rule for the unconditioned comparison run.",
    )
    parser.add_argument(
        "--gmst-forecast-inputs", type=Path, default=None,
        help=("Optional JSON containing annual_hybrid_tpi and "
              "annual_hybrid_global for the 2027 GMST forecast."),
    )
    parser.add_argument(
        "--gmst-sensitivity-manifest", type=Path, default=None,
        help=("Optional JSON mapping GMST dataset names to BADC-style annual "
              "CSV files. Use null for CMST2.0_main. Products are written "
              "below <output>/gmst_sensitivity."),
    )
    return parser.parse_args()


def metric_components(observed: np.ndarray, predicted: np.ndarray,
                      event_mask: np.ndarray) -> np.ndarray:
    """Return [1-r_all, 1-r_event, normalized-RMSE-all, normalized-RMSE-event]."""
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    parts = []
    for mask in (np.ones(len(observed), dtype=bool), event_mask):
        if mask.sum() < 3:
            raise ValueError("At least three observations are required for scoring.")
        obs = observed[mask]
        pred = predicted[mask]
        sd = obs.std(ddof=1)
        obs0 = obs - obs.mean()
        pred0 = pred - pred.mean(axis=0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denom = np.sqrt(np.sum(obs0**2) * np.sum(pred0**2, axis=0))
            corr = np.divide(
                obs0 @ pred0, denom,
                out=np.full(pred.shape[1], -1.0), where=denom > 1e-14,
            )
            rmse = np.sqrt(np.mean((pred - obs[:, None]) ** 2, axis=0)) / sd
        parts.append((1.0 - np.clip(corr, -1.0, 1.0), rmse))
    return np.column_stack([parts[0][0], parts[1][0], parts[0][1], parts[1][1]])


def summary_metrics(observed: np.ndarray, predicted: np.ndarray,
                    event_mask: np.ndarray) -> dict[str, float]:
    def one(mask: np.ndarray, label: str) -> dict[str, float]:
        obs, pred = observed[mask], predicted[mask]
        error = pred - obs
        return {
            f"{label}_n": int(mask.sum()),
            f"{label}_r": float(np.corrcoef(obs, pred)[0, 1]),
            f"{label}_RMSE": float(np.sqrt(np.mean(error**2))),
            f"{label}_MAE": float(np.mean(np.abs(error))),
        }

    return {
        **one(np.ones(len(observed), dtype=bool), "all"),
        **one(event_mask, "event"),
    }


def empirical_interval(samples: np.ndarray, coverage: float = 0.80) -> tuple[float, float]:
    alpha = (1.0 - coverage) / 2.0
    return tuple(np.quantile(samples, [alpha, 1.0 - alpha]).astype(float))


def empirical_samples(forecast: float, errors: np.ndarray, seed: int) -> np.ndarray:
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("The error pool must contain finite values.")
    rng = np.random.default_rng(seed)
    return float(forecast) - rng.choice(errors, size=MC_DRAWS, replace=True)


class NestedEngine:
    """Vectorized grid fitting and target-specific nested validation."""

    def __init__(self, name: str, kind: str, frame: pd.DataFrame,
                 features: list[str], target: str, offset: np.ndarray,
                 event_years: np.ndarray):
        self.name = name
        self.kind = kind
        self.frame = frame.copy()
        self.years = frame.index.to_numpy(int)
        self.features = features
        self.x = frame[features].to_numpy(float)
        self.y = frame[target].to_numpy(float)
        self.offset = np.asarray(offset, dtype=float)
        self.observed = self.y + self.offset
        self.event_mask = np.isin(self.years, event_years)
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError(f"{name}: non-finite training inputs or targets.")
        self._fit_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._component_cache: dict[tuple[int, ...], np.ndarray] = {}

    def drop_target(self, ids: np.ndarray, held: int) -> np.ndarray:
        keep = ids[ids != held]
        if self.kind == "enso":
            keep = keep[self.years[keep] != self.years[held] + 1]
        return keep

    def fit_grid(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = tuple(map(int, ids))
        if key in self._fit_cache:
            return self._fit_cache[key]
        x = self.x[ids]
        y = self.y[ids]
        mean = x.mean(axis=0)
        raw_scale = x.std(axis=0, ddof=1)
        scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
        z = (x - mean) / scale
        event = self.event_mask[ids]
        design = np.column_stack([np.ones(len(ids)), z])
        coefs = np.empty((len(GRID), z.shape[1] + 1), dtype=float)
        condition_indices = np.empty(len(GRID), dtype=float)
        for wi, weight in enumerate(EVENT_WEIGHTS):
            weights = np.where(event, weight, 1.0)
            root = np.sqrt(weights)
            wsum = weights.sum()
            z_mean = (weights[:, None] * z).sum(axis=0) / wsum
            y_mean = float((weights * y).sum() / wsum)
            z0 = z - z_mean
            y0 = y - y_mean
            covariance = (weights[:, None] * z0).T @ z0 / wsum
            cross = (weights * y0) @ z0 / wsum
            eigenvalue, eigenvector = np.linalg.eigh(covariance)
            positive_eigenvalue = np.maximum(eigenvalue, 0.0)
            for li, penalty in enumerate(RIDGE_LAMBDAS):
                grid_index = li * len(EVENT_WEIGHTS) + wi
                condition_indices[grid_index] = float(np.sqrt(
                    (positive_eigenvalue.max() + penalty)
                    / max(positive_eigenvalue.min() + penalty, 1.0e-15)
                ))
                if penalty == 0.0:
                    beta = np.linalg.lstsq(
                        design * root[:, None], y * root, rcond=None
                    )[0][1:]
                else:
                    beta = eigenvector @ (
                        (eigenvector.T @ cross) / (eigenvalue + penalty)
                    )
                intercept = y_mean - z_mean @ beta
                coefs[grid_index] = np.r_[intercept, beta]
        self._fit_cache[key] = (coefs, mean, scale, condition_indices)
        return self._fit_cache[key]

    def predict_grid(self, train_ids: np.ndarray, test_id: int) -> np.ndarray:
        coefs, mean, scale, _ = self.fit_grid(train_ids)
        z = (self.x[test_id] - mean) / scale
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return np.r_[1.0, z] @ coefs.T + self.offset[test_id]

    def inner_components(self, ids: np.ndarray) -> np.ndarray:
        key = tuple(map(int, ids))
        if key in self._component_cache:
            return self._component_cache[key]
        predictions = np.vstack([
            self.predict_grid(self.drop_target(ids, int(test)), int(test))
            for test in ids
        ])
        result = metric_components(self.observed[ids], predictions, self.event_mask[ids])
        self._component_cache[key] = result
        return result

    def select_grid(self, ids: np.ndarray) -> tuple[int, np.ndarray]:
        components = self.inner_components(ids)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            score = components @ SCORE_WEIGHTS
        # Singular unpenalized candidates can produce non-finite diagnostics
        # in small inner folds.  They are invalid candidates for J and must
        # never win the grid search.
        score = np.where(np.isfinite(score), score, np.inf)
        _, _, _, condition_indices = self.fit_grid(ids)
        selected = int(np.argmin(score))
        if not np.isfinite(CONDITION_INDEX_LIMIT):
            return selected, components

        # Apply the manuscript rule literally: retain the J-minimizing weight,
        # then increase lambda along the predefined grid until CI < limit.
        wi = selected % len(EVENT_WEIGHTS)
        li = selected // len(EVENT_WEIGHTS)
        if condition_indices[selected] >= CONDITION_INDEX_LIMIT:
            for candidate_li in range(li + 1, len(RIDGE_LAMBDAS)):
                candidate = candidate_li * len(EVENT_WEIGHTS) + wi
                if condition_indices[candidate] < CONDITION_INDEX_LIMIT:
                    selected = candidate
                    break
            else:
                # Defensive fallback for an insufficient lambda grid: choose
                # the smallest-CI candidate at the selected weight.
                same_weight = np.arange(len(RIDGE_LAMBDAS)) * len(EVENT_WEIGHTS) + wi
                selected = int(same_weight[np.argmin(condition_indices[same_weight])])
        return selected, components

    def nested_predictions(self, ids: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
        rows, predictions = [], []
        for test in ids:
            train = self.drop_target(ids, int(test))
            selected, components = self.select_grid(train)
            prediction = self.predict_grid(train, int(test))[selected]
            predictions.append(prediction)
            excluded = sorted(set(ids) - set(train))
            _, _, _, condition_indices = self.fit_grid(train)
            selected_condition_index = float(condition_indices[selected])
            rows.append({
                "target_year": int(self.years[test]),
                "outer_training_n": int(len(train)),
                "outer_excluded_years": ",".join(map(str, self.years[excluded])),
                "event_weight": float(GRID[selected, 0]),
                "lambda": float(GRID[selected, 1]),
                "condition_index": selected_condition_index,
                "condition_index_feasible": bool(selected_condition_index < CONDITION_INDEX_LIMIT),
                "inner_score": float(components[selected] @ SCORE_WEIGHTS),
                "r_all_component": float(components[selected, 0]),
                "r_event_component": float(components[selected, 1]),
                "RMSE_all_component": float(components[selected, 2]),
                "RMSE_event_component": float(components[selected, 3]),
            })
        return np.asarray(predictions), pd.DataFrame(rows)


def load_enso_frame(input_dir: Path, name: str, metadata: dict) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(input_dir / f"enso_predictors_{name}.csv").set_index("year").sort_index()
    frame = frame.loc[TRAIN_YEARS].copy()
    previous = frame["NINO34_D0JF"].shift(1)
    previous.loc[1982] = float(metadata[name]["previous_1981_D0JF"])
    frame["previous_nino34"] = previous
    frame["delta_nino34"] = frame["NINO34_D0JF"] - frame["previous_nino34"]
    return frame, float(metadata[name]["previous_1981_D0JF"])


def load_gmst_frame(input_dir: Path, enso_frame: pd.DataFrame,
                    previous_1981: float) -> pd.DataFrame:
    frame = pd.read_csv(input_dir / "gmst_annual_model_frame.csv").set_index("year").sort_index()
    frame.loc[1982, "NINO34_DJF_ENDING_YEAR"] = previous_1981
    for year in range(1983, 2028):
        if year - 1 in enso_frame.index and year in frame.index:
            frame.loc[year, "NINO34_DJF_ENDING_YEAR"] = enso_frame.loc[year - 1, "NINO34_D0JF"]
    train = frame.loc[TRAIN_YEARS, GMST_FEATURES + ["DELTA_CMST2_GMST"]]
    if train.isna().any().any():
        raise ValueError("GMST training frame contains missing values after ERSSTv6 Niño3.4 mapping.")
    return frame


def write_enso_outputs(engine: NestedEngine, nested: np.ndarray, tuning: pd.DataFrame,
                       full_model_index: int, output: Path, forecast_input: np.ndarray,
                       previous_forecast: float) -> tuple[dict, np.ndarray]:
    observed = engine.observed
    prediction = nested
    error = prediction - observed
    validation = pd.DataFrame({
        "year": engine.years,
        "observed": observed,
        "prediction": prediction,
        "observed_increment": engine.y,
        "prediction_increment": prediction - engine.offset,
        "error": error,
        "abs_error": np.abs(error),
        "is_elnino": engine.event_mask,
    }).merge(tuning, left_on="year", right_on="target_year", how="left").drop(columns="target_year")
    validation.to_csv(output / "enso_ersstv6_all_year_validation.csv", index=False, float_format="%.15g")
    validation.loc[validation.is_elnino].rename(columns={"year": "heldout_event"}).to_csv(
        output / "enso_ersstv6_validation.csv", index=False, float_format="%.15g"
    )
    coefs, mean, scale, condition_indices = engine.fit_grid(np.arange(len(engine.years)))
    coef = coefs[full_model_index]
    forecast = previous_forecast + float(
        coef[0] + ((forecast_input - mean) / scale) @ coef[1:]
    )
    event_errors = error[engine.event_mask]
    samples = empirical_samples(forecast, event_errors, MC_SEED)
    record = float(observed.max())
    interval = empirical_interval(samples)
    summary = {
        "dataset": "ersstv6",
        **summary_metrics(observed, prediction, engine.event_mask),
        "event_weight": float(GRID[full_model_index, 0]),
        "lambda": float(GRID[full_model_index, 1]),
        "condition_index": float(condition_indices[full_model_index]),
        "forecast_2026_D0JF": forecast,
        "historical_record": record,
        "record_probability": float(np.mean(samples > record)),
        "strong_probability": float(np.mean(samples > 2.0)),
        "PI80_lower": float(interval[0]),
        "PI80_upper": float(interval[1]),
        "error_pool_n": int(len(event_errors)),
        "selection_rule": (
            f"minimize J; if selected CI>={CONDITION_INDEX_LIMIT:g}, increase lambda along the "
            f"predefined grid at fixed w until CI<{CONDITION_INDEX_LIMIT:g}"
        ),
    }
    model = {
        "features": engine.features,
        "coefficient": coef.tolist(),
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "summary": summary,
        "grid": {
            "event_weights": EVENT_WEIGHTS.tolist(),
            "ridge_lambdas": RIDGE_LAMBDAS.tolist(),
            "score_weights": SCORE_WEIGHTS.tolist(),
        },
    }
    (output / "enso_ersstv6_model.json").write_text(
        json.dumps(model, indent=2), encoding="utf-8"
    )
    return summary, samples


def load_upstream_pools(path: Path | None, erf_path: Path | None,
                        gmst_errors: np.ndarray, enso_errors: np.ndarray) -> list[np.ndarray]:
    if path is None:
        raise ValueError("--joint-errors is required to reproduce the current six-pool GMST uncertainty calculation.")
    table = pd.read_csv(path)
    required = {"e_tpi_nmme", "e_gsst_nmme"}
    if not required.issubset(table.columns):
        raise KeyError(f"Joint-error table is missing {sorted(required - set(table.columns))}.")
    if erf_path is not None:
        erf = pd.read_csv(erf_path)["error"].to_numpy(float)
    elif "e_erf" in table.columns:
        erf = table["e_erf"].to_numpy(float)
    else:
        raise KeyError("Provide --erf-errors or an e_erf column in --joint-errors.")
    return [
        np.asarray(gmst_errors, float), np.asarray(gmst_errors, float),
        np.asarray(enso_errors, float),
        table.e_tpi_nmme.to_numpy(float), table.e_gsst_nmme.to_numpy(float),
        np.asarray(erf, float),
    ]


def validate_forecast_inputs(forecast_inputs: dict | None) -> dict:
    """Validate the corrected 2027 hybrid inputs before they reach the model."""
    if forecast_inputs is None:
        raise ValueError(
            "--gmst-forecast-inputs is required for the CI15 result; "
            "the retired prepared-table placeholders are not accepted."
        )
    required = {"annual_hybrid_tpi", "annual_hybrid_global"}
    missing = required - set(forecast_inputs)
    if missing:
        raise KeyError(f"GMST forecast-input JSON is missing {sorted(missing)}.")
    values = {
        "annual_hybrid_tpi": float(forecast_inputs["annual_hybrid_tpi"]),
        "annual_hybrid_global": float(forecast_inputs["annual_hybrid_global"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("2027 GMST forecast inputs must be finite.")
    for key, stale in STALE_FORECAST_INPUTS_2027.items():
        current = values["annual_hybrid_tpi"] if key == "IPO_TPI_LAG1" else values["annual_hybrid_global"]
        if np.isclose(current, stale, rtol=0.0, atol=1e-12):
            raise ValueError(f"Retired 2027 placeholder supplied for {key}: {stale}.")
    return values


def write_gmst_outputs(engine: NestedEngine, nested: np.ndarray, tuning: pd.DataFrame,
                       full_model_index: int, frame: pd.DataFrame, enso_forecast: float,
                       reporting_offset: float, output: Path,
                       pools: list[np.ndarray], enso_samples: np.ndarray,
                       forecast_inputs: dict | None = None) -> dict:
    coefs, mean, scale, condition_indices = engine.fit_grid(np.arange(len(engine.years)))
    coef = coefs[full_model_index]
    observed_increment = engine.y
    error = nested - observed_increment
    validation = pd.DataFrame({
        "year": engine.years,
        "observed": observed_increment,
        "prediction": nested,
        "observed_increment": observed_increment,
        "prediction_increment": nested,
        "error": error,
        "abs_error": np.abs(error),
        "is_decay_year": engine.event_mask,
    }).merge(tuning, left_on="year", right_on="target_year", how="left").drop(columns="target_year")
    reporting = float(reporting_offset)
    validation["observed_gmst"] = frame.loc[engine.years, "CMST2_GMST"].to_numpy(float) + reporting
    validation["hindcast_gmst"] = frame.loc[engine.years, "GMST_LAG1"].to_numpy(float) + nested + reporting
    validation.to_csv(output / "gmst_cmst2_validation.csv", index=False, float_format="%.15g")

    x2026 = frame.loc[2026, GMST_FEATURES].to_numpy(float).copy()
    x2026[0] = float(frame.loc[2025, "CMST2_GMST"])
    delta2026 = float(coef[0] + ((x2026 - mean) / scale) @ coef[1:])
    raw2026 = x2026[0] + delta2026
    x2027 = frame.loc[2027, GMST_FEATURES].to_numpy(float).copy()
    x2027[0] = raw2026
    x2027[1] = enso_forecast
    forecast_inputs = validate_forecast_inputs(forecast_inputs)
    # Use the same 2026 hybrid upstream forecasts used to define the 2027
    # target, rather than stale placeholder values in the prepared frame.
    x2027[2] = forecast_inputs["annual_hybrid_tpi"]
    x2027[3] = forecast_inputs["annual_hybrid_global"]
    delta2027 = float(coef[0] + ((x2027 - mean) / scale) @ coef[1:])
    raw2027 = raw2026 + delta2027

    rng = np.random.default_rng(MC_SEED)
    draws = np.column_stack([rng.choice(pool, MC_DRAWS, replace=True) for pool in pools])
    beta = coef[1:] / scale
    sensitivity = np.r_[-(1.0 + beta[0]), -1.0, -beta[1:]]
    gmst2026_samples = raw2026 + reporting - draws[:, 0]
    gmst2027_samples = raw2027 + reporting + draws @ sensitivity
    record = float(frame.loc[TRAIN_YEARS, "CMST2_GMST"].max() + reporting)
    interval = empirical_interval(gmst2027_samples)
    summary = {
        "dataset": "CMST2.0_main",
        **summary_metrics(observed_increment, nested, engine.event_mask),
        "event_weight": float(GRID[full_model_index, 0]),
        "lambda": float(GRID[full_model_index, 1]),
        "condition_index": float(condition_indices[full_model_index]),
        "forecast_2026_delta": delta2026,
        "forecast_2027_delta": delta2027,
        "forecast_2026_GMST": raw2026 + reporting,
        "forecast_2027_GMST": raw2027 + reporting,
        "forecast_2026_GMST_computational": raw2026,
        "forecast_2027_GMST_computational": raw2027,
        "record_threshold": record,
        "record_probability": float(np.mean(gmst2027_samples > record)),
        "record_probability_2026": float(np.mean(gmst2026_samples > record)),
        "PI80_lower": float(interval[0]),
        "PI80_upper": float(interval[1]),
        "error_pool_n": int(len(engine.years)),
        "decay_years": GMST_DECAY_YEARS.tolist(),
        "forecast_inputs_2027": {
            "GMST_LAG1": float(x2027[0]),
            "NINO34_DJF_ENDING_YEAR": float(x2027[1]),
            "IPO_TPI_LAG1": float(x2027[2]),
            "GLOBAL_SST_ERSSTv6_LAG1": float(x2027[3]),
            "ERF_WMGHG_LAG1": float(x2027[4]),
        },
        "selection_rule": (
            f"minimize J; if selected CI>={CONDITION_INDEX_LIMIT:g}, increase lambda along the "
            f"predefined grid at fixed w until CI<{CONDITION_INDEX_LIMIT:g}"
        ),
    }
    model = {
        "features": engine.features,
        "coefficient": coef.tolist(),
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "summary": summary,
        "grid": {
            "event_weights": EVENT_WEIGHTS.tolist(),
            "ridge_lambdas": RIDGE_LAMBDAS.tolist(),
            "score_weights": SCORE_WEIGHTS.tolist(),
        },
    }
    (output / "gmst_cmst2_model.json").write_text(
        json.dumps(model, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "monte_carlo_samples.npz",
        # Keep the ENSO and GMST draws together so the figure script can
        # reproduce every probability panel from the same model run.
        enso_2026=enso_samples,
        gmst_2026=gmst2026_samples, gmst_2027=gmst2027_samples,
    )
    return summary


def read_badc_series(path: Path) -> pd.Series:
    """Read the annual ``year,data`` section used by the GMST archives."""
    lines = path.read_text(errors="replace").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "data")
    except StopIteration as exc:
        raise ValueError(f"BADC file has no data section: {path}") from exc
    table = pd.read_csv(path, skiprows=start + 1)
    if not {"year", "data"}.issubset(table.columns):
        raise KeyError(f"BADC file must contain year and data columns: {path}")
    table["year"] = pd.to_numeric(table["year"], errors="coerce")
    table["data"] = pd.to_numeric(table["data"], errors="coerce")
    table = table.dropna(subset=["year", "data"])
    return pd.Series(table["data"].to_numpy(float), index=table["year"].astype(int)).sort_index()


def run_gmst_sensitivity(
    manifest_path: Path, output: Path, base_frame: pd.DataFrame,
    metadata: dict, enso_forecast: float, enso_samples: np.ndarray,
    event_errors: np.ndarray, joint_errors: Path, erf_errors: Path | None,
    forecast_inputs: dict,
) -> None:
    """Fit all GMST datasets listed in a portable, user-supplied manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", manifest)
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("GMST sensitivity manifest must map dataset names to files.")
    destination = output / "gmst_sensitivity"
    destination.mkdir(parents=True, exist_ok=True)
    rows, contributions, audits = [], [], []
    for name, raw_path in datasets.items():
        frame = base_frame.copy()
        if raw_path is not None:
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest_path.parent / path
            report = read_badc_series(path)
            offset = float(report.loc[1991:2020].mean())
            target = report - offset
            frame["CMST2_GMST"] = target.reindex(frame.index)
            frame["GMST_LAG1"] = target.reindex(frame.index - 1).to_numpy()
            frame["DELTA_CMST2_GMST"] = target.diff().reindex(frame.index)
        else:
            offset = float(metadata.get("_global", {}).get("reporting_offset_degC", 0.0))
        if not np.isfinite(frame.loc[TRAIN_YEARS, GMST_FEATURES + ["DELTA_CMST2_GMST"]]).all().all():
            raise ValueError(f"GMST sensitivity dataset {name} contains missing training values.")
        train = frame.loc[TRAIN_YEARS].copy()
        engine = NestedEngine(name, "gmst", train, GMST_FEATURES,
                              "DELTA_CMST2_GMST", np.zeros(len(train)), GMST_DECAY_YEARS)
        ids = np.arange(len(train))
        predictions, tuning = engine.nested_predictions(ids)
        selected, _ = engine.select_grid(ids)
        if not tuning.condition_index_feasible.all():
            raise ValueError(f"GMST sensitivity dataset {name} violates CI15.")
        pools = load_upstream_pools(joint_errors, erf_errors,
                                    predictions - engine.y, event_errors)
        dataset_output = destination / str(name)
        dataset_output.mkdir(parents=True, exist_ok=True)
        tuning.to_csv(dataset_output / "gmst_target_hyperparameters.csv", index=False)
        summary = write_gmst_outputs(
            engine, predictions, tuning, selected, frame, enso_forecast,
            offset, dataset_output, pools, enso_samples, forecast_inputs,
        )
        summary["dataset"] = str(name)
        samples = np.load(dataset_output / "monte_carlo_samples.npz")
        summary["PI80_2026_lower"], summary["PI80_2026_upper"] = np.quantile(
            samples["gmst_2026"], [0.1, 0.9]
        )
        summary["P_2027_gt_1p5"] = float(np.mean(samples["gmst_2027"] > 1.5))
        summary["P_2027_gt_2026"] = float(np.mean(samples["gmst_2027"] > samples["gmst_2026"]))
        model_path = dataset_output / "gmst_cmst2_model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["summary"] = summary
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        beta = np.asarray(model["coefficient"], dtype=float)
        mean = np.asarray(model["mean"], dtype=float)
        scale = np.asarray(model["std"], dtype=float)
        x2027 = np.array([summary["forecast_inputs_2027"][key] for key in GMST_FEATURES])
        values = np.r_[beta[0], beta[1:] * (x2027 - mean) / scale]
        np.testing.assert_allclose(values.sum(), summary["forecast_2027_delta"], atol=1e-12)
        contributions.extend(
            {"dataset": str(name), "factor": factor, "contribution": float(value)}
            for factor, value in zip(["Intercept"] + GMST_FEATURES, values)
        )
        audits.append({
            "dataset": str(name), "n": len(train),
            "event_n": int(engine.event_mask.sum()),
            "all_selected_CI_lt_15": True,
            "max_outer_CI": float(tuning.condition_index.max()),
            "error_pool_sizes": [len(pool) for pool in pools],
            "reporting_offset": offset,
        })
        rows.append(summary)
        pd.DataFrame(rows).to_csv(
            destination / "gmst_dataset_sensitivity_summary.csv",
            index=False, float_format="%.15g",
        )
    pd.DataFrame(contributions).to_csv(
        destination / "contributions.csv", index=False, float_format="%.15g"
    )
    (destination / "audit.json").write_text(
        json.dumps(audits, indent=2), encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Keep manifests relocatable by recording paths relative to the cwd."""
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return path.name


def write_run_manifest(output: Path, input_dir: Path, input_files: list[Path],
                       config: dict, forecast_inputs: Path | None) -> None:
    files = [Path(__file__).resolve(), *(Path(path) for path in input_files if path is not None)]
    hashes = {
        portable_path(path): sha256_file(path)
        for path in files if path.exists() and path.is_file()
    }
    manifest = {
        "manifest_version": "1.0",
        "result_version": "CI15_lambda0-1",
        "source_code": portable_path(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "input_directory": portable_path(input_dir),
        "output_directory": portable_path(output),
        "input_file_hashes": hashes,
        "forecast_inputs_file": portable_path(forecast_inputs) if forecast_inputs else None,
        "configuration": config,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    global CONDITION_INDEX_LIMIT, CONDITION_PENALTY_SCALE
    if args.disable_condition_index:
        CONDITION_INDEX_LIMIT = float("inf")
        CONDITION_PENALTY_SCALE = 0.0
    else:
        CONDITION_INDEX_LIMIT = float(args.condition_index_limit)
        CONDITION_PENALTY_SCALE = 1.0e6
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.input / "processing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    config = {
        "training_years": [1982, 2025],
        "n_training_years": 44,
        "enso_event_years": ENSO_EVENT_YEARS.tolist(),
        "gmst_decay_years": GMST_DECAY_YEARS.tolist(),
        "event_weight_grid": EVENT_WEIGHTS.tolist(),
        "ridge_lambda_grid": RIDGE_LAMBDAS.tolist(),
        "score_weights": SCORE_WEIGHTS.tolist(),
        "condition_index_limit": (None if not np.isfinite(CONDITION_INDEX_LIMIT) else CONDITION_INDEX_LIMIT),
        "condition_index_rule": (
            "disabled; candidates are ranked by J without a condition-index penalty"
            if not np.isfinite(CONDITION_INDEX_LIMIT)
            else f"minimize J; if selected CI >= {CONDITION_INDEX_LIMIT:g}, increase lambda along the predefined grid at fixed w until CI < {CONDITION_INDEX_LIMIT:g}"
        ),
        "condition_penalty_scale": (
            None if np.isfinite(CONDITION_INDEX_LIMIT) else CONDITION_PENALTY_SCALE
        ),
        "monte_carlo_draws": MC_DRAWS,
        "point_forecasts_use_monte_carlo_mean": False,
    }
    (args.output / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_summaries = []
    ersstv6_engine = None
    ersstv6_samples = None
    for name in args.enso_datasets:
        frame, previous = load_enso_frame(args.input, name, metadata)
        engine = NestedEngine(
            name, "enso", frame, ENSO_FEATURES, "delta_nino34",
            frame.previous_nino34.to_numpy(float), ENSO_EVENT_YEARS,
        )
        nested, tuning = engine.nested_predictions(np.arange(len(frame)))
        selected, _ = engine.select_grid(np.arange(len(frame)))
        tuning.to_csv(args.output / f"enso_{name}_target_hyperparameters.csv", index=False)
        raw_input = pd.read_csv(args.input / f"enso_predictors_{name}.csv").set_index("year")
        forecast_input = raw_input.loc[2026, ENSO_FEATURES].to_numpy(float)
        previous_forecast = float(raw_input.loc[2026, "previous_nino34"]) if "previous_nino34" in raw_input else float(raw_input.loc[2025, "NINO34_D0JF"])
        if name == "ersstv6":
            summary, samples = write_enso_outputs(
                engine, nested, tuning, selected, args.output,
                forecast_input, previous_forecast,
            )
            ersstv6_engine, ersstv6_samples = engine, samples
        else:
            obs, pred = engine.observed, nested
            coefs, mean, scale, condition_indices = engine.fit_grid(np.arange(len(engine.years)))
            coef = coefs[selected]
            forecast = previous_forecast + float(
                coef[0] + ((forecast_input - mean) / scale) @ coef[1:]
            )
            event_errors = (pred - obs)[engine.event_mask]
            samples = empirical_samples(forecast, event_errors, MC_SEED)
            record = float(obs.max())
            interval = empirical_interval(samples)
            summary = {
                "dataset": name, **summary_metrics(obs, pred, engine.event_mask),
                "event_weight": float(GRID[selected, 0]),
                "lambda": float(GRID[selected, 1]),
                "condition_index": float(condition_indices[selected]),
                "forecast_2026_D0JF": forecast,
                "historical_record": record,
                "record_probability": float(np.mean(samples > record)),
                "strong_probability": float(np.mean(samples > 2.0)),
                "PI80_lower": float(interval[0]),
                "PI80_upper": float(interval[1]),
                "error_pool_n": int(len(event_errors)),
                "selection_rule": (
                    f"minimize J; if selected CI>={CONDITION_INDEX_LIMIT:g}, increase lambda along the "
                    f"predefined grid at fixed w until CI<{CONDITION_INDEX_LIMIT:g}"
                ),
            }
            pd.DataFrame({
                "year": engine.years, "observed": obs, "prediction": pred,
                "error": pred - obs, "is_elnino": engine.event_mask,
            }).to_csv(args.output / f"enso_{name}_validation.csv", index=False, float_format="%.15g")
            (args.output / f"enso_{name}_model.json").write_text(
                json.dumps({"features": ENSO_FEATURES, "summary": summary}, indent=2),
                encoding="utf-8",
            )
        all_summaries.append(summary)

    if ersstv6_engine is None or ersstv6_samples is None:
        raise RuntimeError("ERSSTv6 ENSO model was not produced.")
    pd.DataFrame(all_summaries).to_csv(
        args.output / "enso_sensitivity_summary.csv", index=False, float_format="%.15g"
    )

    enso_frame, previous = load_enso_frame(args.input, "ersstv6", metadata)
    gmst_frame = load_gmst_frame(args.input, enso_frame, previous)
    forecast_inputs = None
    if args.gmst_forecast_inputs is not None:
        forecast_inputs = json.loads(args.gmst_forecast_inputs.read_text(encoding="utf-8"))
    forecast_inputs = validate_forecast_inputs(forecast_inputs)
    gmst_train = gmst_frame.loc[TRAIN_YEARS].copy()
    gmst_engine = NestedEngine(
        "GMST", "gmst", gmst_train, GMST_FEATURES, "DELTA_CMST2_GMST",
        np.zeros(len(gmst_train)), GMST_DECAY_YEARS,
    )
    gmst_nested, gmst_tuning = gmst_engine.nested_predictions(np.arange(len(gmst_train)))
    gmst_selected, _ = gmst_engine.select_grid(np.arange(len(gmst_train)))
    gmst_tuning.to_csv(args.output / "gmst_target_hyperparameters.csv", index=False)
    gmst_errors = gmst_nested - gmst_engine.y
    enso_validation = pd.read_csv(args.output / "enso_ersstv6_validation.csv")
    event_errors = enso_validation["error"].to_numpy(float)
    event_errors = event_errors[np.isin(enso_validation["heldout_event"], ENSO_EVENT_YEARS)]
    pools = load_upstream_pools(args.joint_errors, args.erf_errors, gmst_errors, event_errors)
    reporting_offset = float(metadata.get("_global", {}).get("reporting_offset_degC", 0.0))
    enso_forecast = float(json.loads(
        (args.output / "enso_ersstv6_model.json").read_text()
    )["summary"]["forecast_2026_D0JF"])
    gmst_summary = write_gmst_outputs(
        gmst_engine, gmst_nested, gmst_tuning, gmst_selected, gmst_frame,
        enso_forecast, reporting_offset, args.output, pools, ersstv6_samples,
        forecast_inputs,
    )
    pd.DataFrame([gmst_summary]).to_csv(
        args.output / "gmst_main_summary.csv", index=False, float_format="%.15g"
    )
    logging.info("ERSSTv6 ENSO forecast: %.4f°C", enso_forecast)
    logging.info("2027 GMST forecast: %.4f°C", gmst_summary["forecast_2027_GMST"])
    if args.gmst_sensitivity_manifest is not None:
        run_gmst_sensitivity(
            args.gmst_sensitivity_manifest, args.output, gmst_frame, metadata,
            enso_forecast, ersstv6_samples, event_errors, args.joint_errors,
            args.erf_errors, forecast_inputs,
        )
    write_run_manifest(
        args.output, args.input,
        [metadata_path, args.joint_errors, args.erf_errors, args.gmst_forecast_inputs,
         args.gmst_sensitivity_manifest, *args.input.glob("*.csv")],
        config, args.gmst_forecast_inputs,
    )
    logging.info("Model outputs written to %s", args.output)


if __name__ == "__main__":
    main()
