#!/usr/bin/env python3
"""Reproduce the final ENSO-GMST figure from the fitted model products."""

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
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.path import Path as MarkerPath
from scipy import stats


PROJECT_ROOT = Path(os.environ.get("ENSO_PROJECT_ROOT", "/Users/leichengjie/Desktop/2026ENSO"))
DEFAULT_INPUT = PROJECT_ROOT / "数据" / "code_reproduction_1991_2020" / "models"
DEFAULT_PREPARED = PROJECT_ROOT / "数据" / "code_reproduction_1991_2020" / "prepared"
DEFAULT_OUTPUT = PROJECT_ROOT / "图件" / "code_reproduction_1991_2020"

EVENT_YEARS = np.array([
    1982, 1986, 1987, 1991, 1994, 1997, 2002, 2004,
    2006, 2009, 2014, 2015, 2018, 2019, 2023,
])
BLACK = "#111827"
OBS_BLUE = "#2F779F"
PALE_BLUE = "#DAE8F6"
ORANGE = "#ED7640"
LIGHT_ORANGE = "#F6B36D"
POST_ORANGE = "#DF6035"
RED = "#FF0000"
GRAY = "#D1D5DB"
PEACOCK = "#2F779F"
FORECAST_SHADE = "#FCE8E8"
CHECK_MARKER = MarkerPath(
    [(-0.72, -0.02), (-0.22, -0.55), (0.76, 0.62)],
    [MarkerPath.MOVETO, MarkerPath.LINETO, MarkerPath.LINETO],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12.0,
        "font.weight": "semibold",
        "axes.labelsize": 13.3,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "legend.fontsize": 11.0,
        "figure.dpi": 180,
        "savefig.dpi": 450,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dy: float = 0.014) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 - 0.018, box.y1 + dy, label,
        ha="center", va="center", color="white", fontsize=18.4,
        fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.34", fc=BLACK, ec="none"),
        zorder=60,
    )


def sci_p(p: float) -> str:
    if p >= 0.001:
        return f"p = {p:.3f}"
    exponent = int(math.floor(math.log10(p)))
    mantissa = p / 10**exponent
    return f"p = {mantissa:.2f}×10$^{{{exponent}}}$"


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    result = {
        "r": float(np.corrcoef(observed, predicted)[0, 1]),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }
    result["p"] = float(stats.pearsonr(observed, predicted).pvalue)
    return result


def metric_box(ax: plt.Axes, result: dict[str, float], loc=(0.04, 0.95)) -> None:
    text = (
        f"R = {result['r']:.3f}\n{sci_p(result['p'])}\n"
        f"MAE = {result['MAE']:.3f}°C\nRMSE = {result['RMSE']:.3f}°C"
    )
    ax.text(
        *loc, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=12.8, color=BLACK, linespacing=1.25,
        bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="#D8DEE8", alpha=0.92),
        zorder=20,
    )


def event_styles(event_validation: pd.DataFrame) -> dict[int, tuple[float, str]]:
    ordered = event_validation.sort_values("observed", ascending=False).reset_index(drop=True)
    values = event_validation["observed"].to_numpy(float)
    scale = (values - values.min()) / max(1e-12, values.max() - values.min())
    sizes = 115 + scale**1.15 * 370
    size_map = dict(zip(event_validation.heldout_event.astype(int), sizes))
    dark = np.array(mpl.colors.to_rgb("#8F1D1D"))
    light = np.array(mpl.colors.to_rgb(LIGHT_ORANGE))
    styles = {}
    for rank, row in ordered.iterrows():
        fraction = rank / max(1, len(ordered) - 1)
        color = mpl.colors.to_hex(dark * (1 - fraction) + light * fraction)
        year = int(row.heldout_event)
        styles[year] = (float(size_map[year]), color)
    return styles


def gmst_series(
    prepared: Path, validation: pd.DataFrame, model: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(prepared / "gmst_annual_model_frame.csv").set_index("year")
    result = validation.set_index("year").copy()
    reporting_offset = float(model["summary"].get("reporting_offset_degC", 0.0))
    result["observed_gmst"] = frame.loc[result.index, "CMST2_GMST"] + reporting_offset
    result["hindcast_gmst"] = frame.loc[result.index, "GMST_LAG1"] + result.prediction + reporting_offset
    return result.reset_index(), frame


def post_elnino_records(series: pd.DataFrame, include_2027: float | None = None) -> list[tuple[int, float]]:
    values = {
        int(row.year): float(row.hindcast_gmst)
        for row in series.itertuples()
        if int(row.year) in set(EVENT_YEARS + 1)
    }
    if include_2027 is not None:
        values[2027] = float(include_2027)
    records = []
    running = -np.inf
    for year in sorted(values):
        if values[year] > running:
            records.append((year, values[year]))
            running = values[year]
    return records


def panel_a(ax: plt.Axes, all_year: pd.DataFrame, forecast: float) -> None:
    years = all_year.year.to_numpy(int)
    ax.axvspan(2025.5, 2027.0, color=FORECAST_SHADE, zorder=0)
    observed_bars = ax.bar(
        years, all_year.observed, width=0.72, color=OBS_BLUE,
        edgecolor="none", alpha=0.97, zorder=2,
    )
    hindcast_line, = ax.plot(
        years, all_year.prediction, color=ORANGE, marker="o", ms=6.4,
        markerfacecolor=ORANGE, markeredgecolor=ORANGE, lw=2.6,
        zorder=4,
    )
    forecast_bar = ax.bar([2026], [forecast], width=0.72, color=RED,
                          edgecolor="none", zorder=5)
    ax.axhline(0, color="#9CA3AF", lw=0.85, zorder=1)
    ax.set_xlim(1980.7, 2027.1)
    ax.set_ylim(-2.0, 3.0)
    ticks = [1981, 1990, 2000, 2010, 2020, 2026]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        ["1981/1982", "1990/1991", "2000/2001", "2010/2011", "2020/2021", "2026/2027"],
        rotation=36, ha="right",
    )
    ax.get_xticklabels()[-1].set_color(RED)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel("Niño3.4 D(0)JF (°C)")
    result = metrics(all_year.observed.to_numpy(float), all_year.prediction.to_numpy(float))
    header = (
        "Leave-one-event-out validation      "
        f"R = {result['r']:.3f}      {sci_p(result['p'])}      "
        f"MAE = {result['MAE']:.3f}°C      RMSE = {result['RMSE']:.3f}°C"
    )
    ax.text(0.015, 1.06, header, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14.1, fontweight="bold", clip_on=False)
    history_legend = ax.legend(
        [observed_bars[0], hindcast_line], ["Observed", "Hindcast"],
        loc="lower left", bbox_to_anchor=(0.0, -0.005), frameon=False,
        ncol=2, handlelength=2.0, columnspacing=1.8, fontsize=13.0,
    )
    ax.add_artist(history_legend)
    forecast_legend = ax.legend(
        [forecast_bar[0]], ["2026 Forecast"],
        loc="lower right", bbox_to_anchor=(0.995, -0.005), frameon=False,
        handlelength=2.0, fontsize=13.0,
    )
    for text in [*history_legend.get_texts(), *forecast_legend.get_texts()]:
        text.set_fontweight("bold")
    bubble = ax.text(
        2026, forecast + 0.48, f"{forecast:.3f}°C",
        color=RED, fontsize=13.7, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="circle,pad=0.75", fc="#FFE7E7", ec="#FFAAAA"),
        clip_on=False, zorder=15,
    )
    record = ax.text(
        2026, forecast + 1.08, "Record-breaking", color=RED,
        fontsize=13.5, fontweight="bold", ha="center", va="bottom",
        clip_on=False, zorder=15,
    )
    ax._forecast_bubble = bubble
    ax._forecast_record = record


def overlay_event_status(
    ax: plt.Axes, x: float, y: float, size: float,
    positive: bool, record: bool,
) -> None:
    if positive:
        ax.scatter([x], [y], s=size * 1.08, facecolors="none", edgecolors=BLACK,
                   linewidths=1.7, zorder=7)
    if record:
        ax.scatter([x], [y], s=size * 0.30, marker=CHECK_MARKER,
                   facecolors="none", edgecolors="white", linewidths=2.1, zorder=8)
    elif not positive:
        ax.scatter([x], [y], s=size * 0.22, marker="x", color="white",
                   linewidths=2.1, zorder=8)


def panel_b(
    ax: plt.Axes, key_ax: plt.Axes, all_year: pd.DataFrame,
    events: pd.DataFrame, gmst: pd.DataFrame,
) -> None:
    low, high = -2.0, 3.0
    grid = np.linspace(low, high, 300)
    ax.fill_between(grid, grid - 0.5, grid + 0.5, color=PALE_BLUE, zorder=0)
    ax.plot(grid, grid, color=BLACK, lw=1.3)
    ax.plot(grid, grid - 0.5, color="#8ABBE4", lw=0.85, ls="--")
    ax.plot(grid, grid + 0.5, color="#8ABBE4", lw=0.85, ls="--")
    event_set = set(events.heldout_event.astype(int))
    other = ~all_year.year.astype(int).isin(event_set)
    ax.scatter(all_year.loc[other, "observed"], all_year.loc[other, "prediction"],
               s=58, color=GRAY, alpha=0.82, edgecolors="none", zorder=2)
    styles = event_styles(events)
    record_years = {year for year, _ in post_elnino_records(gmst)}
    gmst_by_year = gmst.set_index("year")
    for row in events.itertuples():
        year = int(row.heldout_event)
        size, color = styles[year]
        ax.scatter([row.observed], [row.prediction], s=size, color=color,
                   edgecolors="none", zorder=5)
        post = year + 1
        positive = post in gmst_by_year.index and float(gmst_by_year.loc[post, "observed"]) > 0
        overlay_event_status(ax, row.observed, row.prediction, size, positive, post in record_years)
    for year, offset in ((2002, (10, -18)), (2009, (10, 7))):
        row = events.loc[events.heldout_event.eq(year)].iloc[0]
        ax.annotate(str(year), (row.observed, row.prediction), xytext=offset,
                    textcoords="offset points", color=styles[year][1],
                    fontsize=11.0, fontweight="bold")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_box_aspect(0.72)
    ax.set_xlabel("Observed Niño3.4 D(0)JF index (°C)")
    ax.set_ylabel("Hindcast Niño3.4 D(0)JF (°C)")
    metric_box(ax, metrics(events.observed.to_numpy(float), events.prediction.to_numpy(float)))

    key_ax.set_axis_off()
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.text(0.05, 1.015, "El Niño", ha="left", va="top",
                fontsize=15.0, color=BLACK, fontweight="bold")
    ordered = events.sort_values("observed", ascending=False)
    y_positions = np.linspace(0.91, 0.05, len(ordered))
    for y, row in zip(y_positions, ordered.itertuples()):
        year = int(row.heldout_event)
        size, color = styles[year]
        key_ax.scatter([0.10], [y], s=size * 0.57, color=color, edgecolors="none")
        post = year + 1
        positive = post in gmst_by_year.index and float(gmst_by_year.loc[post, "observed"]) > 0
        overlay_event_status(key_ax, 0.10, y, size * 0.57, positive, post in record_years)
        key_ax.text(0.24, y, f"{year}/{year + 1}", ha="left", va="center",
                    fontsize=11.5, color=color, fontweight="bold")


def empirical_panel(
    ax: plt.Axes, possible: np.ndarray, threshold: float,
    title: str, xlabel: str, probability: float, bar_fraction: float = 0.78,
) -> None:
    counts, edges = np.histogram(possible, bins=10)
    width = edges[1] - edges[0]
    ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=width * bar_fraction,
           color=PEACOCK, edgecolor="none", alpha=0.93, zorder=2)
    grid = np.linspace(min(possible) - width, max(possible) + width, 500)
    # The black curve is a Gaussian-kernel density estimate of the empirical
    # Monte Carlo sample; no parametric normality assumption is made.
    if len(np.unique(possible)) > 1:
        density = stats.gaussian_kde(possible)(grid)
        curve = density / density.max() * (max(counts) + 0.8)
    else:
        curve = np.zeros_like(grid)
        curve[np.argmin(np.abs(grid - possible[0]))] = max(counts) + 0.8
    ax.plot(grid, curve, color=BLACK, lw=1.8, zorder=4)
    rng = np.random.default_rng(2027)
    ax.scatter(possible, -0.20 + rng.uniform(-0.025, 0.025, len(possible)),
               s=20, color=RED, edgecolors="none", zorder=5)
    ax.axvline(threshold, color=RED, lw=1.55, ls=(0, (5, 4)))
    ax.text(threshold - 0.012, 0.93, f"Record P:\n{probability * 100:.1f}%",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=13.0, color=RED, fontweight="bold")
    ax.set_title(title, loc="left", x=0.04, fontsize=13.2, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=RED, fontsize=13.1)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", colors=RED)
    ax.spines["bottom"].set_color(BLACK)
    ax.set_ylim(-0.45, max(counts) + 2.0)


def panel_c(ax: plt.Axes, possible: np.ndarray, forecast: float, record: float,
            probability: float, strong_probability: float) -> None:
    empirical_panel(
        ax, possible, record,
        f"15 El Niño-year error resampling (n = {len(EVENT_YEARS)})",
        "Possible 2026/2027 Niño3.4 (°C)", probability, bar_fraction=0.55,
    )
    ax.axvline(2.0, color=RED, lw=1.45, ls=(0, (5, 4)))
    ax.text(2.0 - 0.04, 0.93, f"Super-strong P:\n{strong_probability * 100:.1f}%",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=13.0, color=RED, fontweight="bold")
    ymax = ax.get_ylim()[1]
    ax.text(2.0, ymax * 0.27, "2.0°C", color=RED, rotation=90,
            ha="left", va="center", fontsize=10.5, fontweight="bold")
    ax.text(record, ymax * 0.27, "Record", color=RED, rotation=90,
            ha="left", va="center", fontsize=10.5, fontweight="bold")


def gmst_record_styles(series: pd.DataFrame, years: list[int]) -> dict[int, tuple[float, str]]:
    values = np.array([series.loc[series.year.eq(year), "hindcast_gmst"].iloc[0] for year in years])
    order = np.argsort(values)[::-1]
    dark = np.array(mpl.colors.to_rgb("#B52A2A"))
    light = np.array(mpl.colors.to_rgb("#F06464"))
    styles = {}
    for rank, index in enumerate(order):
        fraction = rank / max(1, len(order) - 1)
        styles[years[index]] = (200 - 65 * fraction, mpl.colors.to_hex(dark * (1 - fraction) + light * fraction))
    return styles


def panel_d(ax: plt.Axes, key_ax: plt.Axes, series: pd.DataFrame, summary: dict) -> None:
    observed = series.observed.to_numpy(float)
    predicted = series.prediction.to_numpy(float)
    low, high = -0.18, 0.42
    grid = np.linspace(low, high, 300)
    ax.fill_between(grid, grid - 0.1, grid + 0.1, color=PALE_BLUE, zorder=0)
    ax.plot(grid, grid, color=BLACK, lw=1.3)
    record_years = [year for year, _ in post_elnino_records(series)]
    record_set = set(record_years)
    warm = set(series.loc[series.observed >= summary["warm_jump_threshold"], "year"].astype(int))
    other = ~series.year.astype(int).isin(record_set | warm)
    ax.scatter(series.loc[other, "observed"], series.loc[other, "prediction"],
               s=62, color=PEACOCK, alpha=0.72, edgecolors="none", zorder=2)
    warm_only = series.year.astype(int).isin(warm - record_set)
    ax.scatter(series.loc[warm_only, "observed"], series.loc[warm_only, "prediction"],
               s=92, color=ORANGE, edgecolors="none", zorder=4)
    styles = gmst_record_styles(series, record_years)
    for year in record_years:
        row = series.loc[series.year.eq(year)].iloc[0]
        size, color = styles[year]
        ax.scatter([row.observed], [row.prediction], s=size, color=color,
                   edgecolors="none", zorder=5)
    for year in (1981, 1992, 2023):
        row = series.loc[series.year.eq(year)]
        if not row.empty:
            ax.annotate(str(year), (row.observed.iloc[0], row.prediction.iloc[0]),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=9.8, fontweight="bold",
                        color=ORANGE if year == 2023 else PEACOCK)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_box_aspect(0.72)
    ax.set_xlabel("Observed ΔGMST (°C)")
    ax.set_ylabel("Hindcast ΔGMST (°C)")
    metric_box(ax, metrics(observed, predicted))

    key_ax.set_axis_off()
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.text(0.05, 1.015, "Record", ha="left", va="top",
                fontsize=15.0, fontweight="bold", color=BLACK)
    y_positions = np.linspace(0.91, 0.35, len(record_years))
    for y, year in zip(y_positions, reversed(record_years)):
        size, color = styles[year]
        key_ax.scatter([0.10], [y], s=size * 0.70, color=color, edgecolors="none")
        key_ax.text(0.24, y, str(year), ha="left", va="center",
                    fontsize=11.5, color=color, fontweight="bold")
    key_ax.text(0.05, 0.27, "Warm jump", fontsize=13.3, fontweight="bold", color=BLACK)
    key_ax.scatter([0.10], [0.20], s=75, color=ORANGE, edgecolors="none")
    key_ax.text(0.05, 0.12, "Other", fontsize=13.3, fontweight="bold", color=BLACK)
    key_ax.scatter([0.10], [0.05], s=62, color=PEACOCK, edgecolors="none")


def panel_e(
    ax: plt.Axes, series: pd.DataFrame, frame: pd.DataFrame, model: dict,
) -> None:
    summary = model["summary"]
    years = series.year.to_numpy(int)
    event = set(EVENT_YEARS)
    post = {year + 1 for year in event}
    colors = []
    for year in years:
        if year == 1992:
            colors.append("#4A84BD")
        elif year in event:
            colors.append(LIGHT_ORANGE)
        elif year in post:
            colors.append(POST_ORANGE)
        else:
            colors.append("#C7C9C9")
    ax.bar(years, series.hindcast_gmst, width=0.62, color=colors,
           edgecolor="none", alpha=0.92, zorder=2)
    ax.plot(years, series.observed_gmst, color="black", marker="o", ms=3.8,
            lw=2.0, label="Observed GMST", zorder=5)
    f2026 = float(summary["forecast_2026_GMST"])
    f2027 = float(summary["forecast_2027_GMST"])
    ax.plot([2025, 2026, 2027], [float(series.loc[series.year.eq(2025), "observed_gmst"].iloc[0]), f2026, f2027],
            color=RED, lw=1.8, ls="--", zorder=6)
    ax.scatter([2026], [f2026], marker="*", s=330, color="#3575D3",
               edgecolors="white", linewidths=0.7, zorder=8)
    ax.scatter([2027], [f2027], marker="*", s=330, color=RED,
               edgecolors="white", linewidths=0.7, zorder=8)
    ax.axvspan(2025.5, 2030, ymin=0, ymax=1.70 / 1.8,
               color="#F5E7DB", zorder=0)
    threshold = 1.5
    ax.plot([2005, 2030], [threshold, threshold], color=RED,
            lw=2.1, ls=(0, (5, 4)), zorder=4)
    ax.text(2005.3, threshold + 0.025, "1.5°C threshold", color=RED,
            fontsize=12.3, fontweight="bold")
    records = post_elnino_records(series, f2027)
    for (year0, value0), (year1, value1) in zip(records[:-1], records[1:]):
        level = max(value0, value1) + 0.035
        if year1 == 2027:
            level = value1 - 0.015
        end = year1 - 0.35 if year1 == 2027 else year1
        line = dict(color=POST_ORANGE, lw=1.75, ls=(0, (3, 3)), zorder=6)
        ax.plot([year0, year0], [value0 + 0.012, level + 0.018], **line)
        ax.plot([year1, year1], [value1 + 0.012, level + 0.018], **line)
        ax.annotate("", xy=(end, level), xytext=(year0, level),
                    arrowprops=dict(arrowstyle="-|>", color=POST_ORANGE,
                                    lw=1.75, linestyle=(0, (3, 3)), mutation_scale=11))
        ax.text((year0 + year1) / 2, level + 0.023, f"+{value1 - value0:.2f}°C",
                ha="center", va="bottom", color=POST_ORANGE,
                fontsize=9.4, fontweight="bold")
    ax.set_xlim(1980.3, 2030)
    ax.set_ylim(0, 1.8)
    ax.set_xticks([1981, 1990, 2000, 2010, 2020, 2025, 2030])
    ax.set_xticklabels([1981, 1990, 2000, 2010, 2020, 2025, 2030], rotation=35, ha="right")
    ax.set_ylabel("Annual mean GMST anomaly (°C)\n(Base period 1850–1900)")

    bubble_y = [(2022.8, 1.855, float(frame.loc[2024, "CMST2_GMST"]) + float(summary.get("reporting_offset_degC", 0.0)),
                 POST_ORANGE, "2024"),
                (2028.1, 2.065, f2027, RED, "2027")]
    bubbles = []
    for x, y, value, color, year in bubble_y:
        text = ax.text(x, y, f"{value:.3f}°C", color=color, fontsize=13.0,
                       fontweight="bold", ha="center", va="center", clip_on=False,
                       bbox=dict(boxstyle="circle,pad=0.60", fc=mpl.colors.to_rgba(color, 0.13),
                                 ec=mpl.colors.to_rgba(color, 0.38)), zorder=12)
        ax.annotate(year, (x, y), xytext=(0, 32), textcoords="offset points",
                    ha="center", va="bottom", color=color, fontsize=12.5,
                    fontweight="bold", clip_on=False)
        bubbles.append(text)
    start = (2024.45, bubble_y[0][1] - 0.018)
    end = (2026.45, bubble_y[1][1] - 0.018)
    path = MarkerPath([start, ((start[0] + end[0]) / 2, start[1] - 0.10), end],
                      [MarkerPath.MOVETO, MarkerPath.CURVE3, MarkerPath.CURVE3])
    arrow = FancyArrowPatch(path=path, transform=ax.transData, arrowstyle="-|>",
                            color=RED, linewidth=2.2, mutation_scale=14,
                            clip_on=False, zorder=11)
    ax.add_patch(arrow)
    ax._bubble_2027 = bubbles[1]
    handles = [
        Patch(color=LIGHT_ORANGE, label="El Niño-year"),
        Patch(color=POST_ORANGE, label="Post-El Niño"),
        Patch(color="#4A84BD", label="Volcanic eruption"),
        Line2D([], [], color="black", marker="o", ms=3.8, lw=2, label="Observed GMST"),
        Line2D([], [], marker="*", ls="", ms=13, color="#3575D3", label="2026 Forecast"),
        Line2D([], [], marker="*", ls="", ms=13, color=RED, label="2027 Forecast"),
    ]
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                       ncol=6, frameon=False, fontsize=11.7, columnspacing=1.3)
    for text, color in zip(legend.get_texts()[-2:], ("#3575D3", RED)):
        text.set_color(color)
        text.set_fontweight("bold")


def panel_f(ax: plt.Axes, possible: np.ndarray, model: dict) -> None:
    forecast = float(model["summary"]["forecast_2027_GMST"])
    threshold = float(model["summary"]["record_threshold"])
    empirical_panel(
        ax, possible, threshold,
        "Joint upstream-error Monte Carlo (n = 200,000)",
        "Possible 2027 GMST (°C)", float(model["summary"]["record_probability"]),
    )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])


def save_figure(fig: plt.Figure, base: Path, extra=()) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            base.with_suffix(f".{suffix}"),
            dpi=450 if suffix == "png" else None,
            bbox_inches="tight", bbox_extra_artists=list(extra), facecolor="white",
        )


def make_convergence_figure(output: Path, enso_samples: np.ndarray,
                            gmst_samples: np.ndarray, enso_record: float,
                            gmst_record: float) -> None:
    """Plot the 50-seed stability check used for Supplementary Fig. S5."""
    draws = np.array([100, 300, 1000, 3000, 10000, 30000, 100000, 200000])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, samples, threshold, title in zip(
        axes, (enso_samples, gmst_samples), (enso_record, gmst_record),
        ("ENSO record probability", "2027 GMST record probability"),
    ):
        rng_master = np.random.default_rng(20260922)
        curves = []
        for _ in range(50):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
            idx = rng.integers(0, len(samples), size=int(draws[-1]))
            sampled = samples[idx]
            curves.append([np.mean(sampled[:n] > threshold) for n in draws])
        curves = np.asarray(curves)
        primary_rng = np.random.default_rng(20260922)
        primary = primary_rng.integers(0, len(samples), size=int(draws[-1]))
        primary_curve = np.array([np.mean(samples[primary[:n]] > threshold) for n in draws])
        ax.fill_between(draws, np.quantile(curves, .1, axis=0),
                        np.quantile(curves, .9, axis=0), color="#F6B36D", alpha=.30)
        ax.plot(draws, primary_curve, color=ORANGE, lw=1.8)
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cumulative Monte Carlo draws")
        ax.set_ylabel("Record probability")
        ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    save_figure(fig, output / "Figure_S5_MC_convergence")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    args.output.mkdir(parents=True, exist_ok=True)
    all_year = pd.read_csv(args.input / "enso_ersstv6_all_year_validation.csv")
    events = pd.read_csv(args.input / "enso_ersstv6_validation.csv")
    enso_model = read_json(args.input / "enso_ersstv6_model.json")
    gmst_validation = pd.read_csv(args.input / "gmst_cmst2_validation.csv")
    gmst_model = read_json(args.input / "gmst_cmst2_model.json")
    samples_path = args.input / "monte_carlo_samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError("Run 02_fit_validate_models.py first to create empirical MC samples.")
    mc = np.load(samples_path)
    series, gmst_frame = gmst_series(args.prepared, gmst_validation, gmst_model)

    fig = plt.figure(figsize=(22.2, 27.0), facecolor="white")
    outer = fig.add_gridspec(
        4, 1, height_ratios=[0.68, 0.92, 0.92, 0.98],
        left=0.075, right=0.895, top=0.945, bottom=0.105, hspace=0.30,
    )
    ax_a = fig.add_subplot(outer[0])
    row_b = outer[1].subgridspec(1, 2, width_ratios=[1.58, 0.62], wspace=0.20)
    group_b = row_b[0].subgridspec(1, 2, width_ratios=[1.22, 0.22], wspace=0.033)
    ax_b, key_b = fig.add_subplot(group_b[0]), fig.add_subplot(group_b[1])
    ax_c = fig.add_subplot(row_b[1])
    row_d = outer[2].subgridspec(1, 2, width_ratios=[1.58, 0.62], wspace=0.20)
    group_d = row_d[0].subgridspec(1, 2, width_ratios=[1.22, 0.22], wspace=0.033)
    ax_d, key_d = fig.add_subplot(group_d[0]), fig.add_subplot(group_d[1])
    ax_f = fig.add_subplot(row_d[1])
    ax_e = fig.add_subplot(outer[3])

    forecast = float(enso_model["summary"]["forecast_2026_D0JF"])
    panel_a(ax_a, all_year, forecast)
    panel_b(ax_b, key_b, all_year, events, series)
    panel_c(
        ax_c, mc["enso_2026"], forecast,
        float(enso_model["summary"]["historical_record"]),
        float(enso_model["summary"]["record_probability"]),
        float(enso_model["summary"]["strong_probability"]),
    )
    panel_d(ax_d, key_d, series, gmst_model["summary"])
    panel_f(ax_f, mc["gmst_2027"], gmst_model)
    panel_e(ax_e, series, gmst_frame, gmst_model)

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_f, ax_e):
        ax.tick_params(width=1.0, length=3.7)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
    fig.canvas.draw()
    for probability_ax in (ax_c, ax_f):
        pos = probability_ax.get_position()
        probability_ax.set_position([pos.x1 - pos.width * 0.84,
                                     pos.y0 + pos.height * 0.25,
                                     pos.width * 0.84, pos.height * 0.64])
    epos = ax_e.get_position()
    ax_e.set_position([ax_a.get_position().x0, epos.y0 - 0.01,
                       ax_a.get_position().width, epos.height * 0.91])

    for ax, label in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"),
                      (ax_d, "d"), (ax_f, "f"), (ax_e, "e")):
        panel_label(fig, ax, label)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    c_start = tuple(fig.transFigure.inverted().transform(ax_c.transData.transform((4.0, 10.0))))
    a_end = tuple(fig.transFigure.inverted().transform(ax_a.transData.transform((2026.0, 0.0))))
    a_path = MarkerPath(
        [c_start, (min(0.99, ax_c.get_position().x1 + 0.075), c_start[1] + 0.035),
         (min(0.99, ax_a.get_position().x1 + 0.070), a_end[1] - 0.020), a_end],
        [MarkerPath.MOVETO, MarkerPath.CURVE4, MarkerPath.CURVE4, MarkerPath.CURVE4],
    )
    arrow_c = FancyArrowPatch(path=a_path, transform=fig.transFigure, arrowstyle="-|>",
                              mutation_scale=18, linewidth=3.2, color="#3575D3",
                              clip_on=False, zorder=40)
    fig.add_artist(arrow_c)
    f_end = tuple(fig.transFigure.inverted().transform(ax_f.transData.transform((1.86, 7.5))))
    e_box = ax_e._bubble_2027.get_bbox_patch().get_window_extent(renderer).transformed(fig.transFigure.inverted())
    e_start = (e_box.x1 + 0.002, e_box.y0 + 0.55 * e_box.height)
    f_path = MarkerPath(
        [e_start, (min(0.99, e_start[0] + 0.055), e_start[1] + 0.07),
         (min(0.99, ax_f.get_position().x1 + 0.05), f_end[1] + 0.05), f_end],
        [MarkerPath.MOVETO, MarkerPath.CURVE4, MarkerPath.CURVE4, MarkerPath.CURVE4],
    )
    arrow_f = FancyArrowPatch(path=f_path, transform=fig.transFigure, arrowstyle="-|>",
                              mutation_scale=18, linewidth=3.2, color="#3575D3",
                              clip_on=False, zorder=40)
    fig.add_artist(arrow_f)

    extra = [*fig.texts, ax_a._forecast_bubble, ax_a._forecast_record,
             ax_e._bubble_2027, arrow_c, arrow_f]
    base = args.output / "combined_ENSO_GMST_six_panel"
    save_figure(fig, base, extra)
    plt.close(fig)

    fig_f, standalone = plt.subplots(figsize=(6.2, 4.8))
    panel_f(standalone, mc["gmst_2027"], gmst_model)
    fig_f.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.86)
    save_figure(fig_f, args.output / "GMST_probability_panel_f")
    plt.close(fig_f)
    make_convergence_figure(
        args.output, mc["enso_2026"], mc["gmst_2027"],
        float(enso_model["summary"]["historical_record"]),
        float(gmst_model["summary"]["record_threshold"]),
    )
    print(f"Figures written to {args.output}")


if __name__ == "__main__":
    main()
