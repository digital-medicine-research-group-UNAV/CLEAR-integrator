"""User-facing CLEAR integration model API."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from .._legacy import Integrator, VAEConfig
from ..module import ConditionalVAE


class CLEARIntegrationModel:
    """Thin public wrapper around the original CLEAR Integrator."""

    def __init__(
        self,
        adata: Any | None = None,
        *,
        batch_key: str | None = None,
        cell_type_col: str | None = None,
        layer: str = "lognorm_gaussian",
        reference_dictionary: dict[str, list[str]] | None = None,
        seed: int = 0,
        **integrator_kwargs: Any,
    ) -> None:
        self.adata = None
        self.batch_key = batch_key
        self.cell_type_col = cell_type_col
        self.layer = layer
        self.reference_dictionary = reference_dictionary
        self.seed = seed
        self.integrator_kwargs = dict(integrator_kwargs)
        self.integrator: Integrator | None = None
        self.module: ConditionalVAE | None = None
        self.counts_tensor: torch.Tensor | None = None
        self.cell_types: list[str] | None = None

        if adata is not None:
            self.setup_anndata(
                adata,
                batch_key=batch_key,
                cell_type_col=cell_type_col,
                layer=layer,
                reference_dictionary=reference_dictionary,
            )

    def setup_anndata(
        self,
        adata: Any,
        *,
        batch_key: str | None = None,
        cell_type_col: str | None = None,
        layer: str | None = None,
        reference_dictionary: dict[str, list[str]] | None = None,
    ) -> "CLEARIntegrationModel":
        batch_key = batch_key or self.batch_key
        cell_type_col = cell_type_col or self.cell_type_col
        layer = layer or self.layer
        reference_dictionary = reference_dictionary or self.reference_dictionary

        if batch_key is None:
            raise ValueError("batch_key must be provided")
        if cell_type_col is None:
            raise ValueError("cell_type_col must be provided")
        if reference_dictionary is None:
            raise ValueError("reference_dictionary must be provided")
        if batch_key not in adata.obs:
            raise KeyError(f"batch_key {batch_key!r} not found in adata.obs")
        if cell_type_col not in adata.obs:
            raise KeyError(f"cell_type_col {cell_type_col!r} not found in adata.obs")
        if layer != "X" and layer not in adata.layers:
            raise KeyError(f"layer {layer!r} not found in adata.layers")

        self.adata = adata
        self.batch_key = batch_key
        self.cell_type_col = cell_type_col
        self.layer = layer
        self.reference_dictionary = reference_dictionary
        self.cell_types = adata.obs[cell_type_col].astype(str).unique().tolist()
        return self

    def train(
        self,
        *,
        verbose: int | bool = 2,
        beta: float | None = None,
        epochs_CG_start: int | None = None,
        conf_T_init: float | None = None,
        conf_T_max_decay: float | None = None,
        lambda_size: float | None = None,
        gamma_tau_align: float | None = None,
    ) -> "CLEARIntegrationModel":
        if self.adata is None:
            raise RuntimeError("Call setup_anndata before train")

        matrix = self.adata.X if self.layer == "X" else self.adata.layers[self.layer]
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        else:
            matrix = np.asarray(matrix)
        matrix = matrix.astype(np.float32, copy=False)

        self.integrator = Integrator(seed_offset=self.seed, **self.integrator_kwargs)
        self.module, self.counts_tensor = self.integrator.train_cvae_on_counts(
            matrix,
            df_obs=self.adata.obs,
            batch_key=self.batch_key,
            cell_type_col=self.cell_type_col,
            cell_types=self.cell_types,
            ref_batch=self.reference_dictionary,
            verbose=verbose,
            beta=beta,
            epochs_CG_start=epochs_CG_start,
            conf_T_init=conf_T_init,
            conf_T_max_decay=conf_T_max_decay,
            lambda_size=lambda_size,
            gamma_tau_align=gamma_tau_align,
        )
        return self

    def get_latent_representation(self) -> np.ndarray:
        if self.module is None or self.integrator is None or self.counts_tensor is None:
            raise RuntimeError("Model has not been trained")

        self.module.eval()
        with torch.no_grad():
            mu_z, _ = self.module.encode(
                self.counts_tensor.to(self.integrator.device),
                self.integrator.b_tensor.to(self.integrator.device),
                self.integrator.ct_tensor.to(self.integrator.device),
            )
        return mu_z.cpu().numpy().astype(np.float32, copy=False)

    def save(self, path: str | Path) -> Path:
        if self.module is None:
            raise RuntimeError("Model has not been trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.module.state_dict(),
            "vae_config": asdict(self.module.cfg),
            "batch_key": self.batch_key,
            "cell_type_col": self.cell_type_col,
            "layer": self.layer,
            "reference_dictionary": self.reference_dictionary,
            "seed": self.seed,
            "integrator_kwargs": self.integrator_kwargs,
            "cell_types": self.cell_types,
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "CLEARIntegrationModel":
        payload = torch.load(Path(path), map_location=map_location)
        obj = cls(
            batch_key=payload.get("batch_key"),
            cell_type_col=payload.get("cell_type_col"),
            layer=payload.get("layer", "lognorm_gaussian"),
            reference_dictionary=payload.get("reference_dictionary"),
            seed=payload.get("seed", 0),
            **payload.get("integrator_kwargs", {}),
        )
        cfg = VAEConfig(**payload["vae_config"])
        obj.module = ConditionalVAE(cfg)
        obj.module.load_state_dict(payload["state_dict"])
        obj.module.to(map_location)
        obj.cell_types = payload.get("cell_types")
        return obj


__all__ = ["CLEARIntegrationModel", "Integrator"]
