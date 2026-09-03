#!/usr/bin/env python
"""Run all CLEAR manuscript plotting scripts in dependency order."""

from __future__ import annotations

import make_mb_umap_panel
import plot_covgap
import plot_rank_panels


def main() -> None:
    make_mb_umap_panel.main()
    plot_covgap.main()
    plot_rank_panels.main()


if __name__ == "__main__":
    main()
