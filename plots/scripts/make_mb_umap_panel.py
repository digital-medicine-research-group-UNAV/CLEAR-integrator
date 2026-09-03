#!/usr/bin/env python
"""Compose the MB integration UMAP panel (Panel D) used in the CLEAR ranking figure.

The panel is rendered directly from the processed AnnData so every method UMAP is
drawn cleanly at a uniform size with a single shared legend per encoding (batch and
cell type), instead of tiling pre-rendered PNGs that each carried their own tiny,
repeated legend.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _plotting_common import METHOD_LABELS, METHOD_ORDER, OUTDIR, RESULTS_DIR, configure_matplotlib


MB_RESULT_DIR = RESULTS_DIR / "MB" / "experimento-batch-coverage"
ADATA_PATH = MB_RESULT_DIR / "experimento-batch-coverage_rep0_seed42_adata_processed_batch_embeddings.h5ad"
OUTPUT_PATH = OUTDIR / "umap_panel_MB-assay.png"

CLEAR_RED = "#B2182B"
CELL_RATIO = 1.45         # width : height of every UMAP cell (keeps them uniform)
POINT_SIZE = 3.2
CELL_FACECOLOR = "#F4F4F7"
# "cols":  2 rows (Batch / Cell type) x 6 method columns  -> canonical, but a wide strip.
# "stacked": 3 methods per band, Batch over Cell type     -> fills the panel; larger UMAPs.
LAYOUT = os.environ.get("MB_PANEL_LAYOUT", "stacked")


def find_adata() -> Path:
    if ADATA_PATH.exists():
        return ADATA_PATH
    candidates = sorted(MB_RESULT_DIR.glob("*_adata_processed_batch_embeddings.h5ad"))
    if not candidates:
        raise FileNotFoundError(
            f"MB processed AnnData not found under {MB_RESULT_DIR}. Expected a file matching "
            "*_adata_processed_batch_embeddings.h5ad with X_umap_<method> embeddings."
        )
    return candidates[0]


def cell_limits(xy: np.ndarray, ratio: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Robust, equal-scale axis limits padded to the target width:height ratio."""
    xlo, xhi = np.percentile(xy[:, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(xy[:, 1], [0.5, 99.5])
    cx, cy = (xlo + xhi) / 2.0, (ylo + yhi) / 2.0
    w, h = (xhi - xlo) * 1.08, (yhi - ylo) * 1.08
    if w / h < ratio:
        w = h * ratio
    else:
        h = w / ratio
    return (cx - w / 2.0, cx + w / 2.0), (cy - h / 2.0, cy + h / 2.0)


def draw_umap(ax: mpl.axes.Axes, xy: np.ndarray, codes: np.ndarray, colors: np.ndarray, order: np.ndarray) -> None:
    ax.scatter(
        xy[order, 0], xy[order, 1],
        c=colors[codes[order]], s=POINT_SIZE, linewidths=0.0, alpha=0.9, rasterized=True,
    )
    xlim, ylim = cell_limits(xy, CELL_RATIO)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1.0 / CELL_RATIO)  # uniform cell shape and equal on-screen scale
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(CELL_FACECOLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#BBBBBB")


def legend_handles(colors: np.ndarray, n: int, markersize: float = 8.0) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=markersize,
               markerfacecolor=colors[i], markeredgecolor="none")
        for i in range(n)
    ]


class PanelData:
    def __init__(self, adata) -> None:
        batch = adata.obs["batch"]
        ctype = adata.obs["cell_type"]
        self.batch_codes = batch.cat.codes.to_numpy()
        self.ctype_codes = ctype.cat.codes.to_numpy()
        self.batch_cats = list(batch.cat.categories)
        self.ctype_cats = list(ctype.cat.categories)
        self.batch_colors = np.asarray(list(adata.uns["batch_colors"]))
        self.ctype_colors = np.asarray(list(adata.uns["cell_type_colors"]))
        self.order = np.random.default_rng(0).permutation(adata.n_obs)  # fair overplotting
        self.methods = [m for m in METHOD_ORDER if f"X_umap_{m}" in adata.obsm]
        self.xy = {m: adata.obsm[f"X_umap_{m}"] for m in self.methods}

    def method_title(self, ax: mpl.axes.Axes, method: str) -> None:
        label = METHOD_LABELS[method]
        ax.set_title(
            label, fontsize=11.5, fontweight="bold", pad=4,
            color=CLEAR_RED if label == "CLEAR" else "#1A1A1A",
        )


def add_legends(
    leg_ax: mpl.axes.Axes,
    data: PanelData,
    ct_anchor: float = 0.66,
    leg_x: float = 0.0,
    fontsize: float = 9.5,
    title_fontsize: float = 10.0,
    markersize: float = 8.0,
) -> None:
    leg_ax.axis("off")
    batch_legend = leg_ax.legend(
        legend_handles(data.batch_colors, len(data.batch_cats), markersize), data.batch_cats,
        title="Batch", loc="upper left", bbox_to_anchor=(leg_x, 1.0), frameon=False,
        handletextpad=0.4, labelspacing=0.45, fontsize=fontsize, borderaxespad=0.0,
    )
    batch_legend.get_title().set_fontweight("bold")
    batch_legend.get_title().set_fontsize(title_fontsize)
    leg_ax.add_artist(batch_legend)
    ctype_legend = leg_ax.legend(
        legend_handles(data.ctype_colors, len(data.ctype_cats), markersize), data.ctype_cats,
        title="Cell type", loc="upper left", bbox_to_anchor=(leg_x, ct_anchor), frameon=False,
        handletextpad=0.4, labelspacing=0.45, fontsize=fontsize, borderaxespad=0.0,
    )
    ctype_legend.get_title().set_fontweight("bold")
    ctype_legend.get_title().set_fontsize(title_fontsize)


def layout_cols(data: PanelData) -> mpl.figure.Figure:
    """2 rows (Batch / Cell type) x N method columns + shared legends (wide strip)."""
    n = len(data.methods)
    fig = plt.figure(figsize=(11.0, 3.9))
    gs = fig.add_gridspec(
        2, n + 1, width_ratios=[1.0] * n + [1.15],
        wspace=0.07, hspace=0.10, left=0.035, right=0.995, top=0.90, bottom=0.03,
    )
    for col, method in enumerate(data.methods):
        xy = data.xy[method]
        ax_b = fig.add_subplot(gs[0, col])
        draw_umap(ax_b, xy, data.batch_codes, data.batch_colors, data.order)
        data.method_title(ax_b, method)
        ax_c = fig.add_subplot(gs[1, col])
        draw_umap(ax_c, xy, data.ctype_codes, data.ctype_colors, data.order)
        if col == 0:
            ax_b.set_ylabel("Batch", fontsize=10.5, fontweight="bold", color="#333333", labelpad=4)
            ax_c.set_ylabel("Cell type", fontsize=10.5, fontweight="bold", color="#333333", labelpad=4)
    add_legends(fig.add_subplot(gs[:, n]), data, ct_anchor=0.66)
    return fig


def layout_stacked(data: PanelData, per_band: int = 3) -> mpl.figure.Figure:
    """Methods split into bands; within each band Batch sits over Cell type.

    Fills the panel's slot better than the wide strip, so every UMAP is larger.
    """
    methods = data.methods
    bands = [methods[i:i + per_band] for i in range(0, len(methods), per_band)]
    band_gap = 0.45  # extra vertical white space between the two method bands

    # Rows: [batch, cell] per band, with an empty spacer row between bands.
    height_ratios: list[float] = []
    for b in range(len(bands)):
        if b > 0:
            height_ratios.append(band_gap)
        height_ratios.extend([1.0, 1.0])
    nrows = len(height_ratios)
    ncols = per_band + 1  # + shared-legend column

    fig = plt.figure(figsize=(9.2, 5.9))
    gs = fig.add_gridspec(
        nrows, ncols, width_ratios=[1.0] * per_band + [0.58], height_ratios=height_ratios,
        wspace=0.07, hspace=0.12, left=0.045, right=0.995, top=0.94, bottom=0.02,
    )
    for b, band in enumerate(bands):
        row_b = 3 * b       # each earlier band = 2 rows + 1 spacer
        row_c = 3 * b + 1
        for col, method in enumerate(band):
            xy = data.xy[method]
            ax_b = fig.add_subplot(gs[row_b, col])
            draw_umap(ax_b, xy, data.batch_codes, data.batch_colors, data.order)
            data.method_title(ax_b, method)
            ax_c = fig.add_subplot(gs[row_c, col])
            draw_umap(ax_c, xy, data.ctype_codes, data.ctype_colors, data.order)
            if col == 0:
                ax_b.set_ylabel("Batch", fontsize=12.0, fontweight="bold", color="#333333", labelpad=4)
                ax_c.set_ylabel("Cell type", fontsize=12.0, fontweight="bold", color="#333333", labelpad=4)
    add_legends(
        fig.add_subplot(gs[:, per_band]), data, ct_anchor=0.62,
        leg_x=0.12, fontsize=12.5, title_fontsize=13.5, markersize=10.0,
    )
    return fig


def make_panel(adata_path: Path) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib("rank")
    mpl.rcParams.update({"savefig.dpi": 300})

    data = PanelData(ad.read_h5ad(adata_path))
    builder = {"cols": layout_cols, "stacked": layout_stacked}.get(LAYOUT)
    if builder is None:
        raise ValueError(f"Unknown MB panel layout: {LAYOUT!r} (use 'cols' or 'stacked').")
    fig = builder(data)

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


def main() -> None:
    output_path = make_panel(find_adata())
    print(f"Generated MB UMAP panel: {output_path}")


if __name__ == "__main__":
    main()
