#!/usr/bin/env python3
"""Generate the manuscript and supplementary figures from model output.

The plotting code is deliberately separated from calculation.  It reads only
CSV/JSON products written by ``02_fit_validate_models.py`` plus the archived
diagnostic and sensitivity tables used in the Supplementary Information.

"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(os.environ.get("ENSO_PROJECT_ROOT", "/Users/leichengjie/Desktop/2026ENSO"))
DEFAULT_INPUT = PROJECT_ROOT / "数据" / "code_reproduction" / "models"
DEFAULT_PREPARED = PROJECT_ROOT / "数据" / "code_reproduction" / "prepared"
DEFAULT_OUTPUT = PROJECT_ROOT / "图件" / "code_reproduction"

EVENT_YEARS = np.array([1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2014, 2015, 2018, 2019, 2023])
STRONG_YEARS = {1982, 1991, 1997, 2015, 2023}
WARM_JUMP_YEARS = {1981, 1983, 1987, 1988, 1990, 1995, 1997, 1998, 2002, 2009, 2010, 2014, 2015, 2016, 2023}

BLACK = "#111827"
GRAY = "#C4C5C5"
DARK_GRAY = "#50514A"
ORANGE = "#ED963A"
LIGHT_ORANGE = "#F6B36D"
RED = "#DA5427"
CYAN = "#5DC8F5"
BLUE = "#77B5E8"
PALE_BLUE = "#DBEAFE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.8, "font.weight": "semibold",
        "axes.labelsize": 12, "axes.labelweight": "bold", "axes.linewidth": 1.0,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 180, "savefig.dpi": 450, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.020, dy: float = 0.020) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, ha="center", va="center",
             color="white", fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="circle,pad=0.33", fc=BLACK, ec="none"), zorder=50)


def metric_box(ax: plt.Axes, lines: list[str]) -> None:
    ax.text(0.045, 0.95, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=8.8, color=BLACK, linespacing=1.28,
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.04", fc="white", ec="#D8DEE8", alpha=0.88))


def save_figure(fig: plt.Figure, base: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_enso_scatter(ax: plt.Axes, validation: pd.DataFrame, summary: dict) -> None:
    low, high = -2.0, 3.0
    x = np.linspace(low, high, 200)
    ax.fill_between(x, x - 0.5, x + 0.5, color=PALE_BLUE, alpha=0.75)
    ax.plot(x, x, color=BLACK, lw=1.3)
    ax.plot(x, x - 0.5, color="#7FB9FF", lw=0.9, ls="--")
    ax.plot(x, x + 0.5, color="#7FB9FF", lw=0.9, ls="--")
    values = validation.observed.to_numpy(float)
    colors = plt.cm.OrRd(0.30 + 0.65 * (values - values.min()) / (values.max() - values.min()))
    ax.scatter(validation.observed, validation.prediction, s=42, c=colors, edgecolors="none", zorder=5)
    for year in (2002, 2009):
        row = validation.loc[validation.heldout_event.eq(year)].iloc[0]
        ax.text(row.observed, row.prediction + 0.16, str(year), color=ORANGE,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xlim(low, high); ax.set_ylim(low, high); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed Niño3.4 D(0)JF index (°C)")
    ax.set_ylabel("Hindcast Niño3.4 D(0)JF index (°C)")
    p = summary.get("p_value", np.nan)
    metric_box(ax, [f"r = {summary['correlation']:.3f}", f"p = {p:.2e}",
                    f"MAE = {summary['MAE']:.3f}°C", f"RMSE = {summary['RMSE']:.3f}°C"])


def draw_enso_bars(ax: plt.Axes, validation: pd.DataFrame, model: dict) -> None:
    years = validation.heldout_event.astype(int).tolist()
    x = np.arange(len(years))
    w = 0.36
    ax.bar(x - w / 2, validation.observed, width=w, color=GRAY, edgecolor="none", label="Observed")
    ax.bar(x + w / 2, validation.prediction, width=w, color=LIGHT_ORANGE, edgecolor="none", label="Hindcast")
    forecast = model["summary"]["forecast_2026_D0JF"]
    ax.bar(len(years), forecast, width=w, color=RED, edgecolor="none", label="Forecast")
    ax.text(len(years), forecast + 0.10, f"{forecast:.2f}°C", color=RED, ha="center", fontweight="bold")
    ax.axhline(1.5, color="#F39C34", lw=1.1, ls=(0, (5, 4)))
    ax.axhline(0.5, color="#9AA9BC", lw=0.9, ls=(0, (3, 4)), alpha=0.8)
    ax.set_xticks(np.arange(len(years) + 1)); ax.set_xticklabels([*map(str, years), "2026"], rotation=45, ha="right")
    ax.set_ylabel("Niño3.4 D(0)JF index (°C)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#D8DEE8")


def figure_enso(input_dir: Path, output: Path) -> None:
    validation = pd.read_csv(input_dir / "enso_ersstv6_validation.csv")
    model = read_json(input_dir / "enso_ersstv6_model.json")
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.9), gridspec_kw={"width_ratios": [1.05, 1.58], "wspace": 0.22})
    draw_enso_scatter(axes[0], validation, model["summary"])
    draw_enso_bars(axes[1], validation, model)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.17)
    panel_label(fig, axes[0], "a", dy=0.026); panel_label(fig, axes[1], "b", dy=0.026)
    save_figure(fig, output / "figure_ENSO_two_panel_ab")


def gmst_series(prepared: Path, validation: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(prepared / "gmst_annual_model_frame.csv").set_index("year")
    result = validation.set_index("year").copy()
    result["observed_gmst"] = frame.loc[result.index, "CMST2_GMST"]
    result["hindcast_gmst"] = frame.loc[result.index, "GMST_LAG1"] + result.prediction
    return result.reset_index()


def draw_gmst_scatter(ax: plt.Axes, validation: pd.DataFrame, summary: dict) -> None:
    low, high = -0.2, 0.42
    x = np.linspace(low, high, 200)
    ax.fill_between(x, x - 0.1, x + 0.1, color=PALE_BLUE, alpha=0.8)
    ax.plot(x, x, color=BLACK, lw=1.25)
    ax.plot(x, x - 0.1, color="#7FB9FF", lw=0.9, ls="--")
    ax.plot(x, x + 0.1, color="#7FB9FF", lw=0.9, ls="--")
    colors = [ORANGE if int(y) in WARM_JUMP_YEARS else CYAN for y in validation.year]
    ax.scatter(validation.observed, validation.prediction, c=colors, s=30, edgecolors="none", zorder=4)
    for year in (1981, 1992, 2023):
        row = validation.loc[validation.year.eq(year)].iloc[0]
        ax.text(row.observed, row.prediction + 0.018, str(year), color=colors[list(validation.year).index(year)],
                ha="center", fontsize=8.2, fontweight="bold")
    ax.set_xlim(low, high); ax.set_ylim(low, high); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed ΔGMST (°C)"); ax.set_ylabel("Predicted ΔGMST (°C)")
    metric_box(ax, [f"r = {summary['correlation']:.3f}", f"p = {summary['p_value']:.2e}",
                    f"MAE = {summary['MAE']:.3f}°C", f"RMSE = {summary['RMSE']:.3f}°C"])
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="Warm-jump years"),
                       Line2D([], [], marker="o", ls="", color=CYAN, label="Other years")],
              loc="lower right", frameon=True, facecolor="white", edgecolor="#D8DEE8")


def draw_gmst_time_series(ax: plt.Axes, series: pd.DataFrame, model: dict) -> None:
    years = series.year.to_numpy(int)
    event = set(EVENT_YEARS.tolist())
    post = {year + 1 for year in event}
    colors = [LIGHT_ORANGE if y in event else RED if y in post else GRAY for y in years]
    ax.bar(years, series.hindcast_gmst, width=0.67, color=colors, edgecolor="none", alpha=0.88)
    ax.plot(years, series.observed_gmst, color="black", lw=1.45, label="Observed CMST2.0")
    s = model["summary"]
    ax.plot([2025, 2026, 2027], [series.loc[series.year.eq(2025), "observed_gmst"].iloc[0],
                                 s["forecast_2026_GMST"], s["forecast_2027_GMST"]],
            color="#A61E2D", lw=1.6, ls="--")
    ax.scatter([2026, 2027], [s["forecast_2026_GMST"], s["forecast_2027_GMST"]],
               marker="*", s=95, color="#A61E2D", zorder=8)
    ax.text(2026.05, s["forecast_2026_GMST"] - 0.06, "2026", color="#A61E2D", fontweight="bold")
    ax.text(2027, s["forecast_2027_GMST"] + 0.05, "2027", color="#A61E2D", ha="center", fontweight="bold")
    ax.axhline(s["record_threshold"], color="#F39C34", lw=1.1, ls=(0, (5, 4)))
    ax.text(2007, s["record_threshold"] + 0.03, "2024 GMST: 1.194°C", color="#A61E2D", fontweight="bold")
    ax.axhline(0, color=BLACK, lw=0.9, ls=(0, (5, 4)))
    ax.text(0.98, 0.02, "Base period 1961–1990", transform=ax.transAxes, ha="right", fontweight="bold")
    ax.set_xlim(1980.2, 2027.8); ax.set_ylim(-0.1, 1.5)
    ax.set_ylabel("Annual mean GMST anomaly (°C)")
    ax.legend(handles=[Line2D([], [], color="black", label="Observed CMST2.0"),
                       Line2D([], [], marker="*", ls="", color="#A61E2D", label="Forecast"),
                       Patch(color=LIGHT_ORANGE, label="El Niño-year"), Patch(color=RED, label="Post-El Niño"),
                       Patch(color=GRAY, label="Other years")], ncol=3, loc="upper left",
              frameon=True, facecolor="white", edgecolor="#D8DEE8")


def draw_gmst_probability(ax: plt.Axes, validation: pd.DataFrame, model: dict) -> None:
    residual = validation.error.to_numpy(float)
    forecast = model["summary"]["forecast_2027_GMST"]
    possible = forecast - residual
    ax.hist(possible, bins=10, color=GRAY, edgecolor="white", alpha=0.95)
    mean, std = possible.mean(), possible.std(ddof=1)
    x = np.linspace(min(possible) - 0.04, max(possible) + 0.04, 300)
    density = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))
    bin_width = (possible.max() - possible.min()) / 10
    ax.plot(x, density * len(possible) * bin_width, color=BLACK, lw=1.8)
    record = model["summary"]["record_threshold"]
    ax.axvline(record, color=ORANGE, lw=1.4, ls=(0, (5, 4)))
    ax.text(record + 0.006, ax.get_ylim()[1] * 0.78, "Record threshold", rotation=90,
            color=DARK_GRAY, fontweight="bold")
    ax.text(0.06, 0.92, f"Gaussian P = {100 * model['summary']['record_probability']:.1f}%",
            transform=ax.transAxes, ha="left", va="top", color=DARK_GRAY, fontweight="bold")
    ax.scatter(possible, np.zeros_like(possible), color=GRAY, s=12, edgecolors="none", zorder=4)
    ax.set_xlabel("Possible 2027 GMST from all-year residuals (°C)"); ax.set_ylabel("Count")


def draw_gmst_contributions(ax: plt.Axes, model: dict) -> None:
    labels = {"GMST_LAG1": "Lagged GMST", "GLOBAL_SST_ERSSTv6_LAG1": "Global SST",
              "ERF_WMGHG_LAG1": "GHG forcing", "NINO34_DJF_ENDING_YEAR": "Niño3.4",
              "IPO_TPI_LAG1": "TPI"}
    order = ["GMST_LAG1", "GLOBAL_SST_ERSSTv6_LAG1", "ERF_WMGHG_LAG1", "NINO34_DJF_ENDING_YEAR", "IPO_TPI_LAG1"]
    values = [model["contribution_2027"][key] for key in order] + [model["intercept_delta"]]
    names = [labels[key] for key in order] + ["Intercept"]
    colors = [CYAN if v < 0 else ORANGE for v in values[:-1]] + [GRAY]
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, height=0.62, edgecolor="none")
    ax.axvline(0, color=BLACK, lw=1)
    for yi, value, color in zip(y, values, colors):
        if value < 0:
            # Keep the value inside the negative bar so it cannot collide with
            # the predictor label on the left-hand side of the axis.
            x_text, align, text_color = value + 0.025, "left", "white"
        else:
            x_text, align, text_color = value + 0.025, "left", color
        ax.text(x_text, yi, f"{value:+.3f}", ha=align, va="center",
                color=text_color, fontweight="bold", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Contribution to 2027 ΔGMST (°C)")


def figure_gmst(input_dir: Path, prepared: Path, output: Path) -> None:
    validation = pd.read_csv(input_dir / "gmst_cmst2_validation.csv")
    model = read_json(input_dir / "gmst_cmst2_model.json")
    series = gmst_series(prepared, validation)
    fig = plt.figure(figsize=(14.8, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.58], hspace=0.34, wspace=0.22,
                          left=0.07, right=0.985, top=0.95, bottom=0.09)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    draw_gmst_scatter(axes[0], validation, model["summary"])
    draw_gmst_time_series(axes[1], series, model)
    draw_gmst_probability(axes[2], validation, model)
    draw_gmst_contributions(axes[3], model)
    for ax, label in zip(axes, "abcd"):
        panel_label(fig, ax, label, dy=0.022)
    save_figure(fig, output / "figure_GMST_four_panel_abcd")


def supplementary_diagnostics(input_dir: Path, output: Path) -> None:
    validation = pd.read_csv(input_dir / "enso_ersstv6_validation.csv")
    seasonal_path = PROJECT_ROOT / "数据" / "diagnostics" / "ersstv6_2002_2009_WWB_D20_SSH_seasonal_standardized.csv"
    seasonal = pd.read_csv(seasonal_path)
    fig = plt.figure(figsize=(13.6, 8.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 0.95], hspace=0.55, wspace=0.24,
                          left=0.07, right=0.985, top=0.95, bottom=0.09)
    ax = fig.add_subplot(gs[0, :])
    years = validation.heldout_event.astype(int).to_numpy(); err = validation.error.to_numpy(float)
    ax.axhspan(-0.5, 0.5, color="#ECFDF5", alpha=0.8)
    ax.axhline(0, color=BLACK, lw=1.0)
    colors = [RED if y in STRONG_YEARS else "#FDBA74" for y in years]
    ax.bar(np.arange(len(years)), err, color=colors, edgecolor="none", width=0.68)
    ax.set_xticks(np.arange(len(years))); ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylabel("Error (°C)"); ax.set_xlabel("El Niño event year")
    axes = [ax]
    seasons = ["MAM", "JJA", "SON", "DJF"]
    styles = {"Other El Niño mean": (BLACK, "o"), "2002": (ORANGE, "s"), "2009": ("#C91424", "D")}
    for j, variable in enumerate(("WWB", "D20", "SSH")):
        subax = fig.add_subplot(gs[1, j]); axes.append(subax)
        for case, (color, marker) in styles.items():
            sub = seasonal[(seasonal.variable == variable) & (seasonal.case == case)]
            values = [sub.loc[sub.season.eq(s), "standardized_value"].iloc[0] for s in seasons]
            subax.plot(seasons, values, color=color, marker=marker, lw=1.8, ms=4.6, label=case)
        subax.axhline(0, color="#7A7A7A", lw=0.8)
        subax.set_ylabel("Standardized value")
        subax.text(0.03, 0.92, variable, transform=subax.transAxes, fontweight="bold")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.25, 1.30), ncol=3,
                   frameon=True, facecolor="white", edgecolor="#D8DEE8")
    for subax, label in zip(axes, "abcd"):
        panel_label(fig, subax, label, dy=0.022)
    save_figure(fig, output / "supplement_ENSO_error_2002_2009_diagnostics")


def supplementary_sensitivity(input_dir: Path, output: Path) -> None:
    summary = pd.read_csv(input_dir / "enso_sensitivity_summary.csv")
    all_validation = pd.read_csv(input_dir / "enso_all_validation.csv")
    colors = {"ersstv6": BLACK, "ersstv5": "#8A8D00", "cobe2": RED, "hadisst": CYAN}
    labels = {"ersstv6": "ERSSTv6", "ersstv5": "ERSSTv5", "cobe2": "COBE2", "hadisst": "HadISST"}
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 10.6), gridspec_kw={"hspace": 0.47})
    ax = axes[0]
    years = EVENT_YEARS.tolist() + [2026]; xpos = {year: i for i, year in enumerate(years)}
    for name in labels:
        sub = all_validation[all_validation.dataset.eq(name)].sort_values("heldout_event")
        x = [xpos[int(y)] for y in sub.heldout_event]
        ax.plot(x, sub.prediction, color=colors[name], marker="o", ms=4, lw=1.9, label=f"{labels[name]} hindcast")
        f = summary.loc[summary.dataset.eq(name), "forecast_2026_D0JF"].iloc[0]
        ax.plot([xpos[2023], xpos[2026]], [sub.loc[sub.heldout_event.eq(2023), "prediction"].iloc[0], f],
                color=colors[name], ls="--", lw=1.9)
        ax.scatter(xpos[2026], f, marker="*", s=90, color=colors[name], zorder=6)
    ax.axhline(1.5, color="#F39C34", ls=(0, (5, 4)), lw=1.1)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_ylabel("Niño3.4 D(0)JF index (°C)"); ax.set_xlabel("El Niño event year")
    ax.legend(ncol=2, loc="upper left", frameon=True, facecolor="white", edgecolor="#D8DEE8")

    gmst_summary_path = PROJECT_ROOT / "数据" / "global_temp_predictors" / "models" / "gmst_dataset_sensitivity" / "gmst_dataset_sensitivity_summary.csv"
    gmst_loyo_path = PROJECT_ROOT / "数据" / "global_temp_predictors" / "models" / "gmst_dataset_sensitivity" / "gmst_dataset_sensitivity_loyo_predictions.csv"
    ax = axes[1]
    if gmst_summary_path.exists() and gmst_loyo_path.exists():
        gsumm = pd.read_csv(gmst_summary_path).set_index("dataset")
        gval = pd.read_csv(gmst_loyo_path)
        palette = plt.cm.tab10(np.linspace(0, 0.8, len(gsumm)))
        for color, name in zip(palette, gsumm.index):
            sub = gval[gval.dataset.eq(name)].sort_values("year")
            if "hindcast_gmst" not in sub:
                continue
            ax.plot(sub.year, sub.hindcast_gmst, color=color, lw=1.6, alpha=0.82, label=name.replace("_main", ""))
            row = gsumm.loc[name]
            ax.plot([2025, 2026, 2027], [sub.loc[sub.year.eq(2025), "hindcast_gmst"].iloc[0],
                                         row.forecast_2026_GMST, row.forecast_2027_GMST],
                    color=color, ls="--", lw=1.6)
            ax.scatter([2026, 2027], [row.forecast_2026_GMST, row.forecast_2027_GMST], color=color, s=25)
        ax.legend(ncol=3, loc="upper left", frameon=True, facecolor="white", edgecolor="#D8DEE8")
    ax.set_ylabel("Annual mean GMST anomaly (°C)"); ax.set_xlabel("Year")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.07)
    panel_label(fig, axes[0], "a", dy=0.028); panel_label(fig, axes[1], "b", dy=0.028)
    save_figure(fig, output / "supplementary_figure2_sensitivity_timeseries")


def main() -> None:
    args = parse_args(); setup_style(); args.output.mkdir(parents=True, exist_ok=True)
    figure_enso(args.input, args.output)
    figure_gmst(args.input, args.prepared, args.output)
    supplementary_diagnostics(args.input, args.output)
    supplementary_sensitivity(args.input, args.output)
    print(f"Figures written to {args.output}")


if __name__ == "__main__":
    main()
