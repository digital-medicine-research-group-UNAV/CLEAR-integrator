#!/usr/bin/env python
"""Generate the CLEAR CovGap manuscript figure from available result folders."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _plotting_common import (
    OUTDIR,
    VALID_COVGAP_THRESHOLD,
    configure_matplotlib,
    load_metric_table,
    method_keys,
    method_labels,
    METHOD_COLORS_COVGAP,
    METHOD_LABELS,
    print_warnings,
    save_all,
    write_loaded_experiments,
)


BASENAME = "figure2_covgap"


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.6, zorder=0)


def heatmap_text_color(value: float, norm: mpl.colors.Normalize, cmap: mpl.colors.Colormap) -> str:
    rgba = cmap(norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.45 else "#1F1F1F"


def plot_covgap_heatmap(ax: mpl.axes.Axes, df: pd.DataFrame) -> mpl.cm.ScalarMappable:
    tasks = list(df["Task"].cat.categories)
    methods = method_labels(df)
    cov_matrix = (
        df.pivot_table(index="Task", columns="Method", values="CovGap", aggfunc="mean", observed=True)
        .reindex(index=tasks, columns=methods)
    )

    finite_values = cov_matrix.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    vmax = max(float(np.nanmax(finite_values)), VALID_COVGAP_THRESHOLD * 2)
    vmin = min(float(np.nanmin(finite_values)), -VALID_COVGAP_THRESHOLD)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=VALID_COVGAP_THRESHOLD, vmax=vmax)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")

    image = ax.imshow(cov_matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", rotation_mode="anchor", fontsize=12)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=12)
    ax.set_xlabel("Method", fontweight="bold", labelpad=8)
    ax.set_ylabel("Integration challenge", fontweight="bold", labelpad=10)
    ax.set_title("Coverage gap across benchmark tasks", loc="left", pad=8, fontweight="bold")
    panel_label(ax, "A")

    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            value = cov_matrix.iat[i, j]
            if pd.isna(value):
                label = "NA"
                color = "#8A8A8A"
                weight = "normal"
            else:
                label = f"{value:.3f}"
                color = heatmap_text_color(float(value), norm, cmap)
                weight = "bold" if value <= VALID_COVGAP_THRESHOLD else "normal"
            ax.text(j, i, label, ha="center", va="center", fontsize=10.0, color=color, fontweight=weight)

    return image


def plot_pass_rate(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    keys = method_keys(df)
    labels = [METHOD_LABELS[key] for key in keys]
    total_tasks = df["Task"].nunique()
    pass_counts = df.groupby("MethodKey", observed=True)["Valid_CovGap"].sum().reindex(keys).fillna(0)

    y = np.arange(len(keys))
    colors = [METHOD_COLORS_COVGAP[key] for key in keys]
    ax.barh(y, pass_counts.to_numpy(), height=0.62, color=colors, edgecolor="#222222", linewidth=0.45, zorder=3)
    ax.axvline(total_tasks, color="#222222", linestyle=(0, (3, 3)), linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, total_tasks + 0.75)
    ax.set_xlabel(f"Valid tasks (CovGap <= {VALID_COVGAP_THRESHOLD:.2f})", fontweight="bold", labelpad=6)
    ax.set_title("Operational validity pass rate", loc="left", pad=8, fontweight="bold")
    panel_label(ax, "B")
    clean_axis(ax)

    for yi, value in zip(y, pass_counts):
        ax.text(value + 0.12, yi, f"{int(value)}/{total_tasks}", ha="left", va="center", fontsize=10, color="#222222")


def plot_shortfall_distribution(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    keys = method_keys(df)
    labels = [METHOD_LABELS[key] for key in keys]
    data = [df.loc[df["MethodKey"] == key, "Coverage_shortfall"].to_numpy(dtype=float) for key in keys]
    positions = np.arange(1, len(keys) + 1)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 0.8},
        capprops={"color": "#444444", "linewidth": 0.8},
    )
    for patch, key in zip(box["boxes"], keys):
        patch.set_facecolor(mpl.colors.to_rgba(METHOD_COLORS_COVGAP[key], 0.24))
        patch.set_edgecolor(METHOD_COLORS_COVGAP[key])
        patch.set_linewidth(0.9)

    rng = np.random.default_rng(7)
    for pos, key, y_values in zip(positions, keys, data):
        jitter = rng.uniform(-0.13, 0.13, size=len(y_values))
        ax.scatter(
            np.full_like(y_values, pos, dtype=float) + jitter,
            y_values,
            s=15,
            color=METHOD_COLORS_COVGAP[key],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )

    ax.axhline(VALID_COVGAP_THRESHOLD, color="#222222", linestyle=(0, (3, 3)), linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor", fontsize=12)
    ax.set_ylabel("Coverage shortfall, max(CovGap, 0)", fontweight="bold", labelpad=8)
    ax.set_title("Shortfall distribution", loc="left", pad=8, fontweight="bold")
    panel_label(ax, "C")
    clean_axis(ax)
    ax.text(
        0.98,
        0.94,
        f"Dashed line: {VALID_COVGAP_THRESHOLD:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color="#333333",
    )


def make_figure(df: pd.DataFrame) -> list:
    configure_matplotlib("covgap")
    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.52, 1.0], height_ratios=[0.95, 1.05])

    ax_heatmap = fig.add_subplot(grid[:, 0])
    image = plot_covgap_heatmap(ax_heatmap, df)
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.025)
    colorbar.set_label("Coverage gap (target - empirical)", fontweight="bold", labelpad=8)
    colorbar.ax.tick_params(labelsize=9.0, width=0.7, length=3.0)

    ax_pass = fig.add_subplot(grid[0, 1])
    plot_pass_rate(ax_pass, df)

    ax_shortfall = fig.add_subplot(grid[1, 1])
    plot_shortfall_distribution(ax_shortfall, df)

    return save_all(fig, BASENAME, tight=True)


def main() -> None:
    df, loaded, warnings = load_metric_table(require_covgap=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / f"{BASENAME}_source_data.csv", index=False)
    write_loaded_experiments(loaded, OUTDIR / f"{BASENAME}_loaded_experiments.csv")
    figure_paths = make_figure(df)
    plt.close("all")

    print_warnings(warnings)
    print(f"Loaded {len(loaded)} available CLEAR experiment result folders.")
    print("Generated files:")
    for path in figure_paths:
        print(f"- {path}")
    print(f"- {OUTDIR / f'{BASENAME}_source_data.csv'}")
    print(f"- {OUTDIR / f'{BASENAME}_loaded_experiments.csv'}")


if __name__ == "__main__":
    main()
