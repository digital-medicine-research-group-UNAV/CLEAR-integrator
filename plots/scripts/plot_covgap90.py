#!/usr/bin/env python
"""Generate Figure 4: CovGap at 90% target coverage using the Figure 2 layout."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _plotting_common import (
    OUTDIR,
    load_metric_table,
    print_warnings,
    write_loaded_experiments,
)
from plot_covgap import make_figure


BASENAME = "figure4_covgap90"


def main() -> None:
    df, loaded, warnings = load_metric_table(require_covgap=True)
    if "TrainTargetCovGap" not in df.columns:
        raise ValueError("Loaded metric table is missing required metric: TrainTargetCovGap")

    df = df.copy()
    df["CovGap"] = df["TrainTargetCovGap"]
    df["Valid_CovGap"] = df["CovGap"] <= 0.01
    df["Coverage_shortfall"] = df["CovGap"].clip(lower=0)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / f"{BASENAME}_source_data.csv", index=False)
    write_loaded_experiments(loaded, OUTDIR / f"{BASENAME}_loaded_experiments.csv")

    import plot_covgap

    previous_basename = plot_covgap.BASENAME
    plot_covgap.BASENAME = BASENAME
    try:
        figure_paths = make_figure(df)
    finally:
        plot_covgap.BASENAME = previous_basename
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
