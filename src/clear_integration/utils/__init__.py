"""Utility functions for CLEAR experiments."""

from ._evaluation import evaluate_latent_logreg
from ._plotting import (
    plot_conftr_diagnostics,
    plot_training_curves,
    reset_umap_basis,
    save_latent_plot,
    save_latent_umap,
)
from ._runtime import _normalize_verbose, _vprint

__all__ = [
    "evaluate_latent_logreg",
    "plot_conftr_diagnostics",
    "plot_training_curves",
    "reset_umap_basis",
    "save_latent_plot",
    "save_latent_umap",
    "_normalize_verbose",
    "_vprint",
]
