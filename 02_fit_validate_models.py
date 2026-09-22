#!/usr/bin/env python3
"""Fit, validate and forecast the ENSO and GMST models used in the paper.

The ENSO experiment predicts the year-to-year change in D(0)JF Nino3.4 with
event-weighted ridge regression. Hyperparameters are selected inside each
leave-one-event-out fold. The GMST experiment uses the retained five-factor
event-weighted regression. The script writes machine-readable predictions,
summaries and fitted coefficients.

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ENSO_PROJECT_ROOT", "/Users/leichengjie/Desktop/2026ENSO"))
DEFAULT_INPUT = PROJECT_ROOT / "数据" / "code_reproduction_1991_2020" / "prepared"
DEFAULT_OUTPUT = PROJECT_ROOT / "数据" / "code_reproduction_1991_2020" / "models"
DEFAULT_JOINT_ERRORS = PROJECT_ROOT / "结果" / "nmme_current_six_june_skill_20260912" / "selected_joint_upstream_errors.csv"
MC_DRAWS = 200_000
MC_SEED = 20260922

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
ENSO_TRAIN_YEARS = np.arange(1982, 2026)
ENSO_EVENT_YEARS = np.array([1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2014, 2015, 2018, 2019, 2023])
# Text S7: 19 equally spaced candidate case-weight multipliers from 1 to 10.
RIDGE_WEIGHTS = tuple(np.linspace(1.0, 10.0, 19))
RIDGE_LAMBDAS = np.r_[0.0, np.geomspace(1e-3, 1e2, 18)]
GMST_WEIGHTS = RIDGE_WEIGHTS
GMST_RIDGE_LAMBDAS = RIDGE_LAMBDAS.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--joint-errors", type=Path, default=DEFAULT_JOINT_ERRORS,
                        help="Six-component historical upstream-error table for 2027 GMST.")
    return parser.parse_args()


def standardize(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0, ddof=1)
        std = np.where(std > 1e-12, std, 1.0)
    return (x - mean) / std, mean, std


def fit_weighted_linear(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, mean, std = standardize(x)
    design = np.column_stack([np.ones(len(y)), z])
    if ridge_lambda > 0:
        penalty = np.diag(np.r_[0.0, np.repeat(ridge_lambda, z.shape[1])])
        coef = np.linalg.solve(
            (design.T * weights) @ design / weights.sum() + penalty,
            (design.T * weights) @ y / weights.sum(),
        )
    else:
        root = np.sqrt(weights)
        coef = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return coef, mean, std


def apply_model(fit: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> float:
    coef, mean, std = fit
    return float(coef[0] + ((np.asarray(x, float) - mean) / std) @ coef[1:])


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def student_t_pvalue(r: float, n: int) -> float:
    try:
        from scipy.stats import t
        statistic = abs(r) * math.sqrt((n - 2) / max(1e-15, 1 - r**2))
        return float(2 * t.sf(statistic, df=n - 2))
    except ImportError:
        return float("nan")


def empirical_samples(forecast: float, errors: np.ndarray, draws: int = MC_DRAWS,
                      seed: int = MC_SEED) -> np.ndarray:
    """Draw empirical Monte Carlo realizations using error = hindcast - observation."""
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("The historical error pool must contain finite values.")
    rng = np.random.default_rng(seed)
    return float(forecast) - rng.choice(errors, size=int(draws), replace=True)


def empirical_probability(samples: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(samples) > float(threshold)))


def empirical_interval(samples: np.ndarray, coverage: float = 0.80) -> tuple[float, float]:
    alpha = 1.0 - coverage
    return tuple(map(float, np.quantile(samples, [alpha / 2, 1 - alpha / 2])))


def remove_event_and_successor(years: np.ndarray, held: int) -> np.ndarray:
    return years[(years != held) & (years != held + 1)]


def make_enso_increment_table(frame: pd.DataFrame, previous_1981: float) -> pd.DataFrame:
    df = frame.copy()
    previous = df.NINO34_D0JF.shift(1)
    previous.loc[1982] = previous_1981
    df["PREVIOUS_NINO34_D0JF"] = previous
    df["DELTA_NINO34_D0JF"] = df.NINO34_D0JF - previous
    return df


def fit_enso(df: pd.DataFrame, years: np.ndarray, weight: float, ridge_lambda: float):
    x = df.loc[years, ENSO_FEATURES].to_numpy(float)
    y = df.loc[years, "DELTA_NINO34_D0JF"].to_numpy(float)
    w = np.where(np.isin(years, ENSO_EVENT_YEARS), weight, 1.0)
    return fit_weighted_linear(x, y, w, ridge_lambda)


def tune_enso(df: pd.DataFrame, years: np.ndarray) -> tuple[float, float]:
    events = ENSO_EVENT_YEARS[np.isin(ENSO_EVENT_YEARS, years)]
    best: tuple[float, float, float] | None = None
    for weight in RIDGE_WEIGHTS:
        for ridge_lambda in RIDGE_LAMBDAS:
            errors = []
            for held in events:
                train = remove_event_and_successor(years, int(held))
                fit = fit_enso(df, train, weight, float(ridge_lambda))
                delta = apply_model(fit, df.loc[held, ENSO_FEATURES].to_numpy(float))
                prediction = float(df.loc[held, "PREVIOUS_NINO34_D0JF"] + delta)
                errors.append(abs(prediction - df.loc[held, "NINO34_D0JF"]))
            candidate = (float(np.mean(errors)), float(ridge_lambda), float(weight))
            if best is None or candidate < best:
                best = candidate
                selected = (float(weight), float(ridge_lambda))
    return selected


def run_enso_dataset(name: str, path: Path, previous_1981: float, output: Path):
    frame = pd.read_csv(path).set_index("year").sort_index()
    required = set(ENSO_FEATURES + ["NINO34_D0JF"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{name}: missing predictors {missing}")
    df = make_enso_increment_table(frame, previous_1981)
    rows = []
    for held in ENSO_EVENT_YEARS:
        outer_train = remove_event_and_successor(ENSO_TRAIN_YEARS, int(held))
        weight, ridge_lambda = tune_enso(df, outer_train)
        fit = fit_enso(df, outer_train, weight, ridge_lambda)
        increment = apply_model(fit, df.loc[held, ENSO_FEATURES].to_numpy(float))
        prediction = float(df.loc[held, "PREVIOUS_NINO34_D0JF"] + increment)
        observed = float(df.loc[held, "NINO34_D0JF"])
        rows.append({
            "dataset": name, "heldout_event": int(held), "observed": observed,
            "prediction": prediction, "error": prediction - observed,
            "event_weight": weight, "lambda": ridge_lambda,
        })
    predictions = pd.DataFrame(rows)
    skill = metric_summary(predictions.observed.to_numpy(), predictions.prediction.to_numpy())
    skill["p_value"] = student_t_pvalue(skill["correlation"], len(predictions))
    final_weight, final_lambda = tune_enso(df, ENSO_TRAIN_YEARS)
    fit = fit_enso(df, ENSO_TRAIN_YEARS, final_weight, final_lambda)
    coef, mean, std = fit
    x2026 = df.loc[2026, ENSO_FEATURES].to_numpy(float)
    z2026 = (x2026 - mean) / std
    delta2026 = float(coef[0] + z2026 @ coef[1:])
    forecast = float(df.loc[2026, "PREVIOUS_NINO34_D0JF"] + delta2026)
    residual = predictions.error.to_numpy(float)
    observed_events = predictions.observed.to_numpy(float)
    strong_residual = residual[observed_events >= 1.5]
    # Final uncertainty and probabilities use the 15 completed El Niño error pool.
    mc_samples = empirical_samples(forecast, residual)
    mc_conditional = empirical_samples(forecast, strong_residual, seed=MC_SEED + 1)
    historical_pi = empirical_interval(mc_samples)
    conditional_pi = empirical_interval(mc_conditional)
    historical_record = float(predictions.observed.max())
    summary = {
        "dataset": name, **skill, "event_weight": final_weight, "lambda": final_lambda,
        "forecast_delta_2026": delta2026, "forecast_2026_D0JF": forecast,
        "historical_record": historical_record,
        "record_probability": empirical_probability(mc_samples, historical_record),
        "strong_probability": empirical_probability(mc_samples, 2.0),
        "PI80_historical_lower": historical_pi[0], "PI80_historical_upper": historical_pi[1],
        "PI80_conditional_lower": conditional_pi[0], "PI80_conditional_upper": conditional_pi[1],
    }
    model = {
        "features": ENSO_FEATURES, "summary": summary,
        "intercept_delta": float(coef[0]),
        "coefficients_standardized": dict(zip(ENSO_FEATURES, map(float, coef[1:]))),
        "training_mean": dict(zip(ENSO_FEATURES, map(float, mean))),
        "training_std": dict(zip(ENSO_FEATURES, map(float, std))),
        "x2026": dict(zip(ENSO_FEATURES, map(float, x2026))),
        "contribution_2026": dict(zip(ENSO_FEATURES, map(float, coef[1:] * z2026))),
    }
    predictions.to_csv(output / f"enso_{name}_validation.csv", index=False, float_format="%.9g")
    if name == "ersstv6":
        np.savez_compressed(output / "monte_carlo_samples.npz", enso_2026=mc_samples)
    (output / f"enso_{name}_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    return summary, predictions


def validate_enso_all_years(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held in ENSO_TRAIN_YEARS:
        train = remove_event_and_successor(ENSO_TRAIN_YEARS, int(held))
        weight, ridge_lambda = tune_enso(df, train)
        fit = fit_enso(df, train, weight, ridge_lambda)
        increment = apply_model(fit, df.loc[held, ENSO_FEATURES].to_numpy(float))
        prediction = float(df.loc[held, "PREVIOUS_NINO34_D0JF"] + increment)
        observed = float(df.loc[held, "NINO34_D0JF"])
        rows.append({
            "year": int(held),
            "observed": observed,
            "prediction": prediction,
            "error": prediction - observed,
            "is_elnino": bool(observed >= 0.5),
            "event_weight": weight,
            "lambda": ridge_lambda,
            "validation_protocol": "leave_target_and_successor_out",
        })
    return pd.DataFrame(rows)


def fit_gmst(
    frame: pd.DataFrame,
    years: np.ndarray,
    event_years: np.ndarray,
    event_weight: float,
    ridge_lambda: float,
):
    x = frame.loc[years, GMST_FEATURES].to_numpy(float)
    y = frame.loc[years, "DELTA_CMST2_GMST"].to_numpy(float)
    weights = np.where(np.isin(years, event_years), event_weight, 1.0)
    return fit_weighted_linear(x, y, weights, ridge_lambda)


def warm_jump_event_years(frame: pd.DataFrame, years: np.ndarray) -> tuple[float, np.ndarray]:
    """Define warm-jump years using only the supplied training split."""
    threshold = float(frame.loc[years, "DELTA_CMST2_GMST"].quantile(0.75))
    events = years[frame.loc[years, "DELTA_CMST2_GMST"].to_numpy(float) >= threshold]
    return threshold, events


def tune_gmst(frame: pd.DataFrame, years: np.ndarray) -> tuple[float, float]:
    best: tuple[float, float, float] | None = None
    selected: tuple[float, float] | None = None
    for event_weight in GMST_WEIGHTS:
        for ridge_lambda in GMST_RIDGE_LAMBDAS:
            errors = []
            for held in years:
                train = years[years != held]
                # Threshold and warm-jump membership are recalculated in the
                # inner training split; the held year cannot affect either.
                _, inner_events = warm_jump_event_years(frame, train)
                fit = fit_gmst(frame, train, inner_events, event_weight, float(ridge_lambda))
                prediction = apply_model(
                    fit, frame.loc[held, GMST_FEATURES].to_numpy(float)
                )
                observed = float(frame.loc[held, "DELTA_CMST2_GMST"])
                errors.append(abs(prediction - observed))
            candidate = (
                float(np.mean(errors)),
                float(ridge_lambda),
                float(event_weight),
            )
            if best is None or candidate < best:
                best = candidate
                selected = (float(event_weight), float(ridge_lambda))
    if selected is None:
        raise RuntimeError("GMST hyperparameter search returned no candidate")
    return selected


def run_gmst_main(
    path: Path,
    enso_forecast: float,
    output: Path,
    computational_baseline: str = "1990-2020",
    reporting_baseline: str = "1850-1900",
    reporting_offset: float = 0.0,
    joint_errors_path: Path | None = None,
):
    frame = pd.read_csv(path).set_index("year").sort_index()
    train_years = np.arange(1981, 2026)
    train_years = np.array([year for year in train_years if np.isfinite(frame.loc[year, GMST_FEATURES + ["DELTA_CMST2_GMST"]].to_numpy(float)).all()])
    threshold, event_years = warm_jump_event_years(frame, train_years)
    final_weight, final_lambda = tune_gmst(frame, train_years)
    rows = []
    for held in train_years:
        outer_train = train_years[train_years != held]
        outer_threshold, outer_events = warm_jump_event_years(frame, outer_train)
        outer_weight, outer_lambda = tune_gmst(frame, outer_train)
        fit = fit_gmst(
            frame,
            outer_train, outer_events, outer_weight, outer_lambda,
        )
        prediction = apply_model(fit, frame.loc[held, GMST_FEATURES].to_numpy(float))
        observed = float(frame.loc[held, "DELTA_CMST2_GMST"])
        rows.append({"year": int(held), "prediction": prediction, "observed": observed,
                     "error": prediction - observed, "event_weight": outer_weight,
                     "lambda": outer_lambda, "warm_jump_threshold": outer_threshold,
                     "validation_protocol": "nested_leave_one_year_out"})
    validation = pd.DataFrame(rows)
    skill = metric_summary(validation.observed.to_numpy(), validation.prediction.to_numpy())
    skill["p_value"] = student_t_pvalue(skill["correlation"], len(validation))
    skill["direction_accuracy"] = float(np.mean(np.sign(validation.prediction) == np.sign(validation.observed)))

    fit = fit_gmst(
        frame, train_years, event_years, final_weight, final_lambda
    )
    coef, mean, std = fit
    delta_2026 = apply_model(fit, frame.loc[2026, GMST_FEATURES].to_numpy(float))
    gmst_2026_computational = float(frame.loc[2026, "GMST_LAG1"] + delta_2026)
    forecast_frame = frame.copy()
    forecast_frame.loc[2027, "GMST_LAG1"] = gmst_2026_computational
    forecast_frame.loc[2027, "NINO34_DJF_ENDING_YEAR"] = enso_forecast
    delta_2027 = apply_model(fit, forecast_frame.loc[2027, GMST_FEATURES].to_numpy(float))
    gmst_2027_computational = gmst_2026_computational + delta_2027
    gmst_2026 = gmst_2026_computational + reporting_offset
    gmst_2027 = gmst_2027_computational + reporting_offset
    residual = validation.error.to_numpy(float)
    record_computational = float(frame.loc[train_years, "CMST2_GMST"].max())
    record = record_computational + reporting_offset
    # 2026 GMST uses its complete 45-year residual pool.
    mc_2026 = empirical_samples(gmst_2026, residual, seed=MC_SEED + 2)

    # 2027 uses the six independently sampled upstream-error pools. The
    # optional table is produced by the NMME/hybrid hindcast audit; a clear
    # fallback is retained for users who run the three scripts alone.
    joint_samples = None
    if joint_errors_path is not None and joint_errors_path.exists():
        errors = pd.read_csv(joint_errors_path)
        expected = ["e_gmst_step1", "e_gmst_step2", "e_enso", "e_tpi_nmme", "e_gsst_nmme", "e_erf"]
        if set(expected).issubset(errors.columns):
            rng = np.random.default_rng(MC_SEED + 3)
            idx = rng.integers(0, len(errors), size=MC_DRAWS)
            e = {name: errors[name].to_numpy(float)[idx] for name in expected}
            x_base = forecast_frame.loc[2027, GMST_FEATURES].to_numpy(float)
            # Each sampled error is hindcast minus observation and is
            # therefore subtracted from the forecast input.
            x_draws = np.repeat(x_base[None, :], MC_DRAWS, axis=0)
            x_draws[:, 0] = x_base[0] - e["e_gmst_step1"]
            x_draws[:, 1] = x_base[1] - e["e_enso"]
            x_draws[:, 2] = x_base[2] - e["e_tpi_nmme"]
            x_draws[:, 3] = x_base[3] - e["e_gsst_nmme"]
            x_draws[:, 4] = x_base[4] - e["e_erf"]
            z_draws = (x_draws - mean) / std
            delta_draws = coef[0] + z_draws @ coef[1:]
            gmst2026_draws = gmst_2026_computational - e["e_gmst_step1"]
            joint_samples = gmst2026_draws + delta_draws - e["e_gmst_step2"] + reporting_offset
    if joint_samples is None:
        # Fallback: independently sample the final increment residual only.
        joint_samples = empirical_samples(gmst_2027, residual, seed=MC_SEED + 3)
    pi_2027 = empirical_interval(joint_samples)
    z2027 = (forecast_frame.loc[2027, GMST_FEATURES].to_numpy(float) - mean) / std
    summary = {
        **skill, "warm_jump_threshold": threshold, "event_weight": final_weight,
        "lambda": final_lambda,
        "computational_baseline": computational_baseline,
        "reporting_baseline": reporting_baseline,
        "reporting_offset_degC": reporting_offset,
        "forecast_2026_delta": delta_2026, "forecast_2026_GMST": gmst_2026,
        "forecast_2027_delta": delta_2027, "forecast_2027_GMST": gmst_2027,
        "forecast_2026_GMST_computational": gmst_2026_computational,
        "forecast_2027_GMST_computational": gmst_2027_computational,
        "record_threshold_computational": record_computational,
        "record_threshold": record,
        "record_probability": empirical_probability(joint_samples, record),
        "record_probability_2026": empirical_probability(mc_2026, record),
        "PI80_historical_lower": pi_2027[0], "PI80_historical_upper": pi_2027[1],
        "PI80_conditional_lower": pi_2027[0], "PI80_conditional_upper": pi_2027[1],
    }
    model = {
        "features": GMST_FEATURES, "summary": summary,
        "intercept_delta": float(coef[0]),
        "coefficients_standardized": dict(zip(GMST_FEATURES, map(float, coef[1:]))),
        "training_mean": dict(zip(GMST_FEATURES, map(float, mean))),
        "training_std": dict(zip(GMST_FEATURES, map(float, std))),
        "contribution_2027": dict(zip(GMST_FEATURES, map(float, coef[1:] * z2027))),
    }
    validation.to_csv(output / "gmst_cmst2_validation.csv", index=False, float_format="%.9g")
    np.savez_compressed(output / "monte_carlo_samples.npz", enso_2026=np.load(output / "monte_carlo_samples.npz")["enso_2026"] if (output / "monte_carlo_samples.npz").exists() else np.empty(0), gmst_2026=mc_2026, gmst_2027=joint_samples)
    (output / "gmst_cmst2_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    return summary, validation


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.input / "processing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    summaries, validations = [], []
    for name in ("ersstv6", "ersstv5", "cobe2", "hadisst"):
        path = args.input / f"enso_predictors_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run 01_prepare_data.py first: {path}")
        previous = float(metadata.get(name, {}).get("previous_1981_D0JF", pd.read_csv(path).NINO34_D0JF.iloc[0]))
        logging.info("ENSO leave-one-event-out validation: %s", name)
        summary, validation = run_enso_dataset(name, path, previous, args.output)
        summaries.append(summary)
        validations.append(validation)
        if name == "ersstv6":
            frame = pd.read_csv(path).set_index("year").sort_index()
            all_year = validate_enso_all_years(make_enso_increment_table(frame, previous))
            all_year.to_csv(
                args.output / "enso_ersstv6_all_year_validation.csv",
                index=False,
                float_format="%.9g",
            )
    pd.DataFrame(summaries).to_csv(args.output / "enso_sensitivity_summary.csv", index=False, float_format="%.9g")
    pd.concat(validations, ignore_index=True).to_csv(args.output / "enso_all_validation.csv", index=False, float_format="%.9g")

    main_enso = next(item for item in summaries if item["dataset"] == "ersstv6")
    gmst_summary, gmst_validation = run_gmst_main(
        args.input / "gmst_annual_model_frame.csv",
        main_enso["forecast_2026_D0JF"],
        args.output,
        computational_baseline=metadata.get("_global", {}).get("computational_baseline", "1990-2020"),
        reporting_baseline=metadata.get("_global", {}).get("reporting_baseline", "1850-1900"),
        reporting_offset=float(metadata.get("_global", {}).get("reporting_offset_degC", 0.0)),
        joint_errors_path=args.joint_errors,
    )
    pd.DataFrame([gmst_summary]).to_csv(args.output / "gmst_main_summary.csv", index=False, float_format="%.9g")
    logging.info("ENSO forecast: %.3f degC", main_enso["forecast_2026_D0JF"])
    logging.info("2027 GMST forecast: %.3f degC", gmst_summary["forecast_2027_GMST"])
    logging.info("Model outputs written to %s", args.output)


if __name__ == "__main__":
    main()
