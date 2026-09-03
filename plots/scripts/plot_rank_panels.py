#!/usr/bin/env python
"""Generate the CLEAR compact four-panel ranking figure."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from _plotting_common import (
    METHOD_COLORS_RANK,
    OUTDIR,
    VALID_COVGAP_THRESHOLD,
    available_metrics,
    configure_matplotlib,
    load_metric_table,
    print_warnings,
    save_all,
    write_loaded_experiments,
)


BASENAME = "figure3_rank_panels_abcd"
MB_UMAP_PATH = OUTDIR / "umap_panel_MB-assay.png"
BIOLOGICAL_METRICS = ["ASW-L", "GC", "NMI", "ARI", "IsLa"]
INTEGRATION_METRIC_CANDIDATES = ["iLISI", "cLISI", "kBET", "ASW-B", "PCR"]
LOWER_IS_BETTER = {"kBET"}


def add_metric_ranks(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
        ascending = metric in LOWER_IS_BETTER
        out[f"{metric}_rank"] = out.groupby("Task", observed=True)[metric].rank(
            ascending=ascending,
            method="average",
        )
    return out


def add_summary_ranks(df: pd.DataFrame, bio_metrics: list[str], integration_metrics: list[str]) -> pd.DataFrame:
    out = add_metric_ranks(df, bio_metrics + integration_metrics)
    out["mean_biological_conservation_rank"] = out[[f"{m}_rank" for m in bio_metrics]].mean(axis=1)
    out["mean_integration_by_task_rank"] = out[[f"{m}_rank" for m in integration_metrics]].mean(axis=1)

    if "CovGap" in out.columns:
        out["CovGap"] = pd.to_numeric(out["CovGap"], errors="coerce")
        out["Valid_CovGap"] = out["CovGap"] <= VALID_COVGAP_THRESHOLD
    else:
        out["CovGap"] = np.nan
        out["Valid_CovGap"] = False

    return out


def luminance_text_color(value: float, norm: Normalize, cmap) -> str:
    rgba = cmap(norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.47 else "black"


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )


def set_method_headers(ax: mpl.axes.Axes, methods: list[str]) -> None:
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=28, ha="left", fontweight="bold", fontsize=6.9, rotation_mode="anchor")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, pad=1.8)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "CLEAR":
            tick.set_color("#B2182B")


def rank_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    title: str,
    panel: str,
    show_y_labels: bool = True,
    show_colorbar: bool = True,
) -> None:
    methods = list(matrix.columns)
    tasks = list(matrix.index)
    n_methods = len(methods)
    cmap = mpl.colormaps["viridis_r"]
    norm = Normalize(vmin=1, vmax=n_methods)

    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks if show_y_labels else [])
    set_method_headers(ax, methods)

    ax.set_title(title, fontsize=9.2, fontweight="bold", pad=8)
    panel_label(ax, panel, x=-0.13, y=1.28)

    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.58)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", length=3.2 if show_y_labels else 0.0, width=0.65)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iat[row_idx, col_idx]
            if pd.isna(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=luminance_text_color(float(value), norm, cmap),
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#A8A8A8")

    if show_colorbar:
        cbar = plt.colorbar(image, ax=ax, orientation="horizontal", fraction=0.075, pad=0.052)
        cbar.set_ticks(np.arange(1, n_methods + 1))
        cbar.set_label("Rank (lower is better)", fontsize=6.8, fontweight="bold", labelpad=1.5)
        cbar.outline.set_linewidth(0.65)
        cbar.ax.tick_params(labelsize=7.0, length=2.2, width=0.55)


def matrix_for(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tasks = list(df["Task"].cat.categories)
    methods = list(df["Method"].cat.categories)
    return (
        df.pivot_table(index="Task", columns="Method", values=value_col, aggfunc="mean", observed=True)
        .reindex(index=tasks, columns=methods)
    )


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("Method", observed=True)
        .agg(
            mean_biological_conservation_rank=("mean_biological_conservation_rank", "mean"),
            mean_integration_by_task_rank=("mean_integration_by_task_rank", "mean"),
            valid_tasks=("Valid_CovGap", "sum"),
            total_tasks=("Task", "nunique"),
        )
        .reset_index()
    )
    summary["Method"] = summary["Method"].astype(str)
    return summary


def plot_overall_position(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    offsets = {
        "scANVI": (-7, -7, "right", "top"),
        "CLEAR": (7, 7, "left", "bottom"),
        "scVI": (12, 18, "left", "bottom"),
        "Harmony": (-18, -18, "right", "top"),
        "Combat": (7, 10, "left", "bottom"),
        "Scanorama": (7, -5, "left", "top"),
    }

    for _, row in summary.iterrows():
        method = row["Method"]
        x = row["mean_integration_by_task_rank"]
        y = row["mean_biological_conservation_rank"]
        color = METHOD_COLORS_RANK.get(method, "#666666")

        ax.scatter(x, y, s=34, color=color, edgecolor="white", linewidth=0.65, zorder=3)

        dx, dy, ha, va = offsets.get(method, (7, 5, "left", "bottom"))
        ax.annotate(
            f"{method}\n({x:.2f}, {y:.2f})\n{int(row['valid_tasks'])}/{int(row['total_tasks'])} valid",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            color=color,
            fontsize=5.9,
            fontweight="bold",
            ha=ha,
            va=va,
            linespacing=1.02,
        )

    max_rank = max(
        float(summary["mean_integration_by_task_rank"].max()),
        float(summary["mean_biological_conservation_rank"].max()),
    )
    upper = min(6.20, max(5.65, np.ceil(max_rank * 2) / 2 + 0.65))
    ax.set_xlim(0.8, upper)
    ax.set_ylim(upper, 0.8)
    ax.set_xlabel("Mean variance removal rank", fontsize=8.8, labelpad=4)
    ax.set_ylabel("Mean biological conservation rank", fontsize=8.8, labelpad=4)
    ax.set_title("Overall benchmark position", fontsize=9.2, pad=6, fontweight="bold")
    ax.grid(color="#D8D8D8", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.2)
    panel_label(ax, "C", x=-0.15, y=1.04)


def trim_white_border(image: np.ndarray, threshold: float = 0.985) -> np.ndarray:
    rgb = image[..., :3]
    mask = np.any(rgb < threshold, axis=2)
    if not np.any(mask):
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 8
    row0 = max(int(rows[0]) - pad, 0)
    row1 = min(int(rows[-1]) + pad + 1, image.shape[0])
    col0 = max(int(cols[0]) - pad, 0)
    col1 = min(int(cols[-1]) + pad + 1, image.shape[1])
    return image[row0:row1, col0:col1]


def plot_mb_umap_panel(ax: mpl.axes.Axes, image_path=MB_UMAP_PATH) -> None:
    if not image_path.exists():
        raise FileNotFoundError(
            f"MB UMAP panel not found: {image_path}. "
            "Run CLEAR/plots/scripts/make_mb_umap_panel.py before plot_rank_panels.py."
        )

    image = trim_white_border(mpimg.imread(image_path))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("MB integration UMAPs", fontsize=9.2, pad=6, fontweight="bold")
    panel_label(ax, "D", x=-0.055, y=1.04)


def make_figure(df: pd.DataFrame) -> list:
    configure_matplotlib("rank")
    fig = plt.figure(figsize=(8.27, 7.65), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        left=0.130,
        right=0.965,
        bottom=0.060,
        top=0.885,
        height_ratios=[1.05, 0.92],
        hspace=0.38,
    )
    top_grid = grid[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.24)
    bottom_grid = grid[1].subgridspec(1, 2, width_ratios=[0.34, 0.66], wspace=0.13)

    ax_a = fig.add_subplot(top_grid[0, 0])
    rank_heatmap(ax_a, matrix_for(df, "mean_biological_conservation_rank"), "Biological conservation by task", "A")

    ax_b = fig.add_subplot(top_grid[0, 1])
    rank_heatmap(
        ax_b,
        matrix_for(df, "mean_integration_by_task_rank"),
        "Variance removal by task",
        "B",
        show_y_labels=False,
    )

    ax_c = fig.add_subplot(bottom_grid[0, 0])
    plot_overall_position(ax_c, method_summary(df))

    ax_d = fig.add_subplot(bottom_grid[0, 1])
    plot_mb_umap_panel(ax_d)

    return save_all(fig, BASENAME)


def write_source_outputs(
    df: pd.DataFrame,
    loaded,
    bio_metrics: list[str],
    integration_metrics: list[str],
) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / f"{BASENAME}_source_data.csv", index=False)
    method_summary(df).to_csv(OUTDIR / f"{BASENAME}_method_summary.csv", index=False)
    write_loaded_experiments(loaded, OUTDIR / f"{BASENAME}_loaded_experiments.csv")
    pd.DataFrame(
        {
            "biological_metrics": pd.Series(bio_metrics),
            "integration_metrics": pd.Series(integration_metrics),
        }
    ).to_csv(OUTDIR / f"{BASENAME}_metric_sets.csv", index=False)


def main() -> None:
    raw, loaded, warnings = load_metric_table()
    bio_metrics = available_metrics(raw, BIOLOGICAL_METRICS)
    integration_metrics = available_metrics(raw, INTEGRATION_METRIC_CANDIDATES)

    if not bio_metrics:
        raise ValueError("No biological metrics were found.")
    if not integration_metrics:
        raise ValueError("No integration metrics were found.")

    df = add_summary_ranks(raw, bio_metrics, integration_metrics)
    write_source_outputs(df, loaded, bio_metrics, integration_metrics)
    figure_paths = make_figure(df)
    plt.close("all")

    print_warnings(warnings)
    print(f"Loaded {len(loaded)} available CLEAR experiment result folders.")
    print(f"Biological metrics: {', '.join(bio_metrics)}")
    print(f"Integration metrics: {', '.join(integration_metrics)}")
    print("Generated files:")
    for path in figure_paths:
        print(f"- {path}")
    print(f"- {OUTDIR / f'{BASENAME}_source_data.csv'}")
    print(f"- {OUTDIR / f'{BASENAME}_method_summary.csv'}")
    print(f"- {OUTDIR / f'{BASENAME}_loaded_experiments.csv'}")
    print(f"- {OUTDIR / f'{BASENAME}_metric_sets.csv'}")


if __name__ == "__main__":
    main()
