#!/usr/bin/env python
"""
Compare multiple batch integration strategies (scANVI, ComBat, Scanorama, scVI, Harmony, Fountain)
on the MB_processed dataset and export the learned embeddings for downstream analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import logging
import random
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, List, Optional, Any
import shutil
import os
import itertools

from anndata.io import read_csv, read_loom, read_text, read_elem
from anndata.abc import CSRDataset, CSCDataset


# Keep JAX from preallocating most GPU memory when metrics run after PyTorch models.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
#os.environ.setdefault("JAX_PLATFORMS", "cpu")
#os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import numbers
from collections.abc import Mapping, Sequence


from clear_integration import Integrator, DataPreparation
from clear_integration.utils import (
    evaluate_latent_logreg,
    plot_training_curves,
    plot_conftr_diagnostics,
)

from scipy import sparse
from scvi import model as scvi_model
import scib_metrics as sm
from scib_metrics import pcr_comparison

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

TRAINING_LOGREG_TARGET_COVERAGE = 0.90

@dataclass
class IntegrationConfig:
    """Runtime configuration for the integration comparison."""
    
    description: str = "Batch integration experiment"
    data_path: Path = None #Path(__file__).resolve().parents[2] / "data" / "diabetic-kidney-disease_processed.h5ad"
    output_dir: Path = None#Path(__file__).resolve().parent / "embeddings"
    subset_max_cells: Optional[int] = 0
    methods: tuple[str, ...] = None #("scanvi", "combat", "scanorama", "scvi", "harmony", "conftr")
    seed: int = 123
    n_repetitions: int = 1
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    min_cells_per_type: int = 300
    n_top_genes: int = 4000
    cell_type_col: str = None #"cell_type"
    reference_dictionary: Dict[str, List[str]] = field(default_factory=dict)
    #reference_dictionary: Dict[str, List[str]] =field(default_factory=lambda:  {"healthy_6": ["control_1", "control_2", "control_3", "healthy_4", "healthy_5", 'diabetic_2', 'diabetic_3','diabetic_4', 'diabetic_5']})  #field(default_factory=lambda: {"10x 5' v1": ["10x 3' v3"]})
    batch_key: str = "batch" #"assay" #"batch"
    counts_layer: str = "counts"
    lognorm_layer: str = "lognorm_gaussian"
 
    ref_batch: str = field(init=False)
    batches: List[str] = field(init=False)

    #ref_batch: str = next(iter(reference_dictionary))
    #batches: list = reference_dictionary[ref_batch]
    #batches_filter: #Optional[Iterable[str]] = field(default_factory=lambda:batches) #["10X", "snATAC"]
    
    beta : float = None
    epochs_CG_start : int = None
    conf_T_init : float = None
    conf_T_max_decay : float = None
    lambda_size : float = None
    gamma_tau_align : float = None
    batch_size: int = None
    conftr_batch_size: Optional[int] = None
    metrics_batch_size: int = 256
    
    pca_components: int = 50
    scvi_latent_dim: int = 16
    scvi_max_epochs: int = 400
    scvi_batch_size: int = 256
    scvi_gene_likelihood: str = "zinb"
    scanvi_latent_dim: int = 16
    scanvi_max_epochs: int = 400
    scanvi_batch_size: int = 1064
    scanvi_unlabeled_fraction: float = 0.1
    scanvi_unlabeled_category: str = "Unknown"
    
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    umap_metric: str = "euclidean"
    

    # Evaluation metrics
    metrics_k: int = 30                 # vecinos kNN para métricas locales (LISI/kBET/ASW)
    metrics_compute: bool = True        # calcula métricas al final
    metrics_compute_scib: bool = True
    metrics_compute_logreg: bool = True
    metrics_compute_pcr: bool = True    # calcula PCR (requiere una representación "pre")
    metrics_isolated_labels: bool = True
    metrics_neighbors_backend: str = "jax"  # "jax" or "pynndescent"
    metrics_output_csv: str = "integration_metrics.csv"
    compute_umap: bool = True
    export_full_h5ad: bool = True
    preprocess_cache: bool = False
    preprocess_cache_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.preprocess_cache_dir is not None:
            self.preprocess_cache_dir = Path(self.preprocess_cache_dir).expanduser()
    #    if self.batches_filter is not None:
    #        self.batches_filter = tuple(sorted(self.batches_filter))
        self.methods = tuple(self.methods)
        self.ref_batch = next(iter(self.reference_dictionary))
        self.batches = self.reference_dictionary[self.ref_batch] + [self.ref_batch]

        self.hyperparams = dict(self.hyperparams)


    @property
    def accelerator(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu" 

    @property
    def lightning_devices(self) -> int | str:
        return 1 if self.accelerator in ("gpu", "cuda") else "auto"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def collect_runtime_environment() -> Dict[str, Any]:
    gpu_info = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        else:
            gpu_info = result.stderr.strip() or None
    except Exception as exc:
        gpu_info = f"unavailable: {exc}"

    jax_devices = None
    try:
        import jax

        jax_devices = [str(device) for device in jax.devices()]
    except Exception as exc:
        jax_devices = f"unavailable: {exc}"

    torch_device_name = None
    if torch.cuda.is_available():
        try:
            torch_device_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            torch_device_name = f"unavailable: {exc}"

    return {
        "captured_at": _utc_now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "scanpy": package_version("scanpy"),
            "anndata": package_version("anndata"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "scvi-tools": package_version("scvi-tools"),
            "scib-metrics": package_version("scib-metrics"),
            "jax": package_version("jax"),
            "scanorama": package_version("scanorama"),
            "harmonypy": package_version("harmonypy"),
        },
        "environment": {
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "torch_cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch_device_name,
        },
        "jax_devices": jax_devices,
        "nvidia_smi": gpu_info,
    }


def log_runtime_environment(runtime: Mapping[str, Any]) -> None:
    packages = runtime.get("packages", {})
    torch_cuda = runtime.get("torch_cuda", {})
    logging.info(
        "Runtime: python=%s torch=%s torch_cuda=%s scanpy=%s scvi-tools=%s scib-metrics=%s",
        runtime.get("python"),
        packages.get("torch"),
        packages.get("torch_cuda"),
        packages.get("scanpy"),
        packages.get("scvi-tools"),
        packages.get("scib-metrics"),
    )
    logging.info(
        "CUDA: available=%s device_count=%s device=%s | JAX devices=%s",
        torch_cuda.get("available"),
        torch_cuda.get("device_count"),
        torch_cuda.get("device_name"),
        runtime.get("jax_devices"),
    )


def update_run_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    write_json_atomic(manifest_path, manifest)


def method_output_prefix(run_name: Optional[str], name_outputs: bool) -> str:
    return f"{run_name}_" if name_outputs and run_name else ""


def method_embedding_path(config: IntegrationConfig, method: str, run_name: Optional[str], name_outputs: bool) -> Path:
    return config.output_dir / f"{method_output_prefix(run_name, name_outputs)}{method}_embedding.npy"


def method_status_path(config: IntegrationConfig, method: str, run_name: Optional[str], name_outputs: bool) -> Path:
    return config.output_dir / f"{method_output_prefix(run_name, name_outputs)}{method}_status.json"


def set_method_status(
    config: IntegrationConfig,
    method: str,
    run_name: Optional[str],
    name_outputs: bool,
    status: str,
    *,
    extra: Optional[Mapping[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
) -> None:
    payload = {
        "method": method,
        "run_name": run_name,
        "status": status,
        "updated_at": _utc_now(),
    }
    if extra:
        payload.update(dict(extra))
    write_json_atomic(method_status_path(config, method, run_name, name_outputs), payload)

    if manifest is not None and manifest_path is not None:
        run_key = run_name or "default"
        run_state = manifest.setdefault("runs", {}).setdefault(run_key, {"methods": {}})
        run_state.setdefault("methods", {})[method] = payload
        update_run_manifest(manifest_path, manifest)


def config_preprocess_signature(config: IntegrationConfig) -> Dict[str, Any]:
    data_stat = config.data_path.stat() if config.data_path.exists() else None
    return {
        "data_path": str(config.data_path.resolve()),
        "data_size": data_stat.st_size if data_stat else None,
        "data_mtime_ns": data_stat.st_mtime_ns if data_stat else None,
        "subset_max_cells": config.subset_max_cells,
        "seed": config.seed,
        "min_cells_per_type": config.min_cells_per_type,
        "n_top_genes": config.n_top_genes,
        "cell_type_col": config.cell_type_col,
        "reference_dictionary": config.reference_dictionary,
        "batch_key": config.batch_key,
        "counts_layer": config.counts_layer,
        "lognorm_layer": config.lognorm_layer,
    }


def preprocess_cache_path(config: IntegrationConfig) -> Path:
    cache_dir = config.preprocess_cache_dir or (config.output_dir / ".preprocess_cache")
    signature = config_preprocess_signature(config)
    digest = hashlib.sha256(json.dumps(_json_safe(signature), sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"prepared_{digest}.h5ad"


def load_and_prepare_data_cached(config: IntegrationConfig) -> sc.AnnData:
    if not config.preprocess_cache:
        return load_and_prepare_data(config)

    cache_path = preprocess_cache_path(config)
    if cache_path.exists():
        logging.info("Loading preprocessed AnnData cache from %s", cache_path)
        return sc.read_h5ad(cache_path)

    adata = load_and_prepare_data(config)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write(cache_path, compression="gzip")
        logging.info("Saved preprocessed AnnData cache to %s", cache_path)
    except Exception as exc:
        logging.warning("Could not save preprocessed cache to %s: %s", cache_path, exc)
    return adata


def _as_scalar(x) -> float:
    """
    Convierte x (float/int/ndarray/list/tuple/dict anidado) en un float escalar
    tomando la media (nanmean) de todos los valores numéricos contenidos.
    Si no hay ningún número, devuelve NaN.
    """
    def _iter_nums(v):
        # Escalar numérico (numpy o python)
        if isinstance(v, numbers.Number) or (isinstance(v, np.ndarray) and v.ndim == 0 and np.issubdtype(v.dtype, np.number)):
            yield float(v)
            return

        # ndarray
        if isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.number):
                for y in v.ravel():
                    yield float(y)
            else:
                # dtype no numérico → ignorar
                return
        # mapping (dict-like): recorrer valores
        elif isinstance(v, Mapping):
            for y in v.values():
                yield from _iter_nums(y)
        # secuencias (lista/tupla), pero no tratar str/bytes como secuencia numérica
        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            for y in v:
                yield from _iter_nums(y)
        # cualquier otro tipo: ignorar

    vals = list(_iter_nums(x))
    return float(np.nan) if len(vals) == 0 else float(np.nanmean(vals))


def _onehot_float32(vec) -> np.ndarray:
    """One-hot (float32) para covariables categóricas."""
    return pd.get_dummies(pd.Series(vec), drop_first=False, dtype=np.float32).to_numpy()

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def load_and_prepare_data(config: IntegrationConfig) -> sc.AnnData:
    logging.info("Loading AnnData from %s", config.data_path)

    adata = sc.read_h5ad(config.data_path)

   
    # This is for small quick runs
    if config.subset_max_cells != 0 and config.subset_max_cells < adata.n_obs:
        rng = np.random.default_rng(config.seed)
        subset_indices = np.sort(rng.choice(adata.n_obs, size=config.subset_max_cells, replace=False))
        adata = adata[subset_indices].copy()
        logging.info("Subsampled dataset to %d cells for quick experimentation", adata.n_obs)


    batch_values = adata.obs[config.batch_key].astype(str)
    adata.obs[config.batch_key] = batch_values.astype("category")
    cell_types = adata.obs[config.cell_type_col].astype(str)
    adata.obs[config.cell_type_col] = cell_types.astype("category")

    # Filter to specific batches if requested
    if config.batches is not None:
        mask = adata.obs[config.batch_key].isin(config.batches)
        adata = adata[mask].copy()
        adata.obs[config.batch_key] = adata.obs[config.batch_key].cat.remove_unused_categories()
        logging.info("Restricted to batches: %s (n=%d)",", ".join(config.batches),adata.n_obs)


    batch_codes = adata.obs[config.batch_key].cat.codes.to_numpy()
    sort_idx = np.argsort(batch_codes)
    adata = adata[sort_idx].copy()

    # Filter cell types with too few cells
    if config.min_cells_per_type > 0:
        type_counts = adata.obs[config.cell_type_col].value_counts()
        keep_types = type_counts[type_counts >= config.min_cells_per_type].index
        if len(keep_types) == 0:
            raise ValueError(
                f"No cell types have at least {config.min_cells_per_type} cells. ")
        
        adata = adata[adata.obs[config.cell_type_col].isin(keep_types)].copy()
        adata.obs[config.cell_type_col] = (
            adata.obs[config.cell_type_col].astype("category").cat.remove_unused_categories()
        )
        logging.info("Filtered to %d cell types with ≥%d cells each", len(keep_types),config.min_cells_per_type,)


    # Select highly variable genes
    if config.n_top_genes is not None:
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=config.n_top_genes,
                flavor="seurat_v3",
                batch_key=config.batch_key,
                subset=True,
                layer=config.counts_layer,
            )
        except Exception as e:
            print("Highly variable gene selection failed:", e)
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=config.n_top_genes,
                flavor="seurat",
                batch_key=config.batch_key,
                subset=True,
                layer="normalized",
            )
        logging.info("Selected top %d highly variable features", config.n_top_genes)
        

    counts_layer = adata.layers[config.counts_layer]
    if sparse.issparse(counts_layer):
        counts_matrix = counts_layer.copy().astype(np.float32)
    else:
        counts_matrix = np.asarray(counts_layer, dtype=np.float32)

    temp = sc.AnnData(
        X=counts_matrix.copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    sc.pp.normalize_total(temp, target_sum=1e4)
    sc.pp.log1p(temp)
    adata.layers[config.lognorm_layer] = temp.X.astype(np.float32, copy=True)
    adata.X = adata.layers[config.lognorm_layer].astype(np.float32, copy=True)

    #logging.info(    "Dataset ready: %d cells × %d features | batches=%s",    adata.n_obs,    adata.n_vars,    ", ".join(adata.obs[config.batch_key].cat.categories))

    return adata


def run_scvi(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:
    logging.info("Training scVI model")
    adata_model = adata.copy()
    scvi_model.SCVI.setup_anndata(
        adata_model,
        layer=config.counts_layer,
        batch_key=config.batch_key,
    )

    model = scvi_model.SCVI(
        adata_model,
        n_hidden = 256,
        n_latent=config.scvi_latent_dim,
        gene_likelihood=config.scvi_gene_likelihood,
    )
    model.train(
        max_epochs=config.scvi_max_epochs,
        accelerator=config.accelerator,
        devices=config.lightning_devices,
        batch_size=config.scvi_batch_size or config.batch_size,
        early_stopping=True,
        early_stopping_patience=10,
        gradient_clip_val = 1.0
    )

    latent = model.get_latent_representation()
    return latent.astype(np.float32)


def run_scanvi(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:
    logging.info("Training scANVI model")
    adata_model = adata.copy()
    labels_cat = adata_model.obs[config.cell_type_col].astype("category")
    categories = labels_cat.cat.categories.tolist()
    if config.scanvi_unlabeled_category not in categories:
        categories.append(config.scanvi_unlabeled_category)
    labels_array = labels_cat.astype(str).to_numpy(copy=True)
    if config.scanvi_unlabeled_fraction > 0:
        rng = np.random.default_rng(config.seed)
        unlabeled_count = max(1, int(np.floor(config.scanvi_unlabeled_fraction * adata_model.n_obs)))
        unlabeled_indices = rng.choice(adata_model.n_obs, size=unlabeled_count, replace=False)
        labels_array[unlabeled_indices] = config.scanvi_unlabeled_category
    adata_model.obs[config.cell_type_col] = pd.Categorical(labels_array, categories=categories)

    scvi_model.SCANVI.setup_anndata(
        adata_model,
        labels_key=config.cell_type_col,
        unlabeled_category=config.scanvi_unlabeled_category,
        layer=config.counts_layer,
        batch_key=config.batch_key,
    )

    

    train_kwargs = {
        "max_epochs": config.scanvi_max_epochs,
        "accelerator": config.accelerator,
        "devices": config.lightning_devices,
        "early_stopping": True,
        "early_stopping_patience": 20,
        "gradient_clip_val": 1.0, # Prevents NaN errors
    }

    try:
        # Attempt 1: Standard initialization and training
        scanvi_batch_size = config.scanvi_batch_size or config.batch_size
        logging.info(f"Attempting training with batch_size={scanvi_batch_size}")
        model = scvi_model.SCANVI(adata_model, n_hidden=256, n_latent=config.scanvi_latent_dim)
        model.train(batch_size=scanvi_batch_size, **train_kwargs)

    except Exception as e:

        if "Expected more than 1 value per channel" in str(e):
            logging.warning("Batch Size 1 error detected (internal train/val split caused a remainder of 1).")
            
            # change batch size significantly 
            new_batch_size = int((config.scanvi_batch_size or config.batch_size) * 0.95)
            logging.info(f"Re-initializing and retrying with batch_size={new_batch_size}")
            
            #  Re-Initialize the model. 
            model = scvi_model.SCANVI(adata_model, n_hidden=256, n_latent=config.scanvi_latent_dim)
            model.train(batch_size=new_batch_size, **train_kwargs)

        else:
            # If it's a different error, crash normally
            raise e
    

    latent = model.get_latent_representation()
    return latent.astype(np.float32)


def run_combat(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:
    logging.info("Running ComBat integration")
    adata_cb = adata.copy()
    adata_cb.X = adata.layers[config.lognorm_layer].astype(np.float32, copy=True)
    sc.pp.scale(adata_cb, zero_center=True, max_value=None)
    sc.pp.combat(adata_cb, key=config.batch_key)
    sc.tl.pca(adata_cb, n_comps=config.pca_components, svd_solver="arpack")
    return adata_cb.obsm["X_pca"][:, : config.pca_components].astype(np.float32)


def run_scanorama(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:
    logging.info("Running Scanorama integration")
    adata_scanorama = adata.copy()
    adata_scanorama.X = adata.layers[config.lognorm_layer].astype(np.float32, copy=True)
    sc.pp.pca(adata_scanorama, n_comps=config.pca_components, svd_solver="arpack")

    sce.pp.scanorama_integrate(
        adata_scanorama,
        key=config.batch_key,
        adjusted_basis="X_scanorama",
    )
    return adata_scanorama.obsm["X_scanorama"].astype(np.float32)


def run_harmony(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:
    logging.info("Running Harmony integration")
    adata_harmony = adata.copy()
    adata_harmony.X = adata.layers[config.lognorm_layer].astype(np.float32, copy=True)
    sc.pp.scale(adata_harmony, zero_center=True, max_value=None)
    sc.tl.pca(adata_harmony, n_comps=config.pca_components, svd_solver="arpack")
    sce.pp.harmony_integrate(
        adata_harmony,
        key=config.batch_key,
        adjusted_basis="X_pca_harmony",
    )
    return adata_harmony.obsm["X_pca_harmony"].astype(np.float32)

def run_conftr(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:

    logging.info("Running conftr integration")
    adata_conftr = adata.copy()
    model_matrix = adata.layers[config.lognorm_layer].astype(np.float32, copy=True)

    train_cvae_epochs_default = 350
    train_cvae_lr_default = 5e-4
    train_cvae_kl_anneal_epochs_default = 1
    train_cvae_batch_size_default = config.conftr_batch_size or config.batch_size #512
    calibration_fraction_default = 0.35  # Set >0 to hold out a calibration set (e.g., 0.4)
    build_data_val_fraction_default = 0.2

    integrador = Integrator(
        seed_offset=config.seed,
        epochs=train_cvae_epochs_default,
        lr=train_cvae_lr_default,
        kl_anneal_epochs=train_cvae_kl_anneal_epochs_default,
        batch_size=train_cvae_batch_size_default,
        calibration_fraction=calibration_fraction_default,
        val_fraction=build_data_val_fraction_default,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    cell_types = adata.obs[config.cell_type_col].astype(str).unique().tolist() #list of cell types
    
    #target_coverage = [0.90,0.80, 0.70]
    #alpha_list = [0.1, 0.2] #[1.0 - c for c in target_coverage]


    #print("batches:", config.batches)
    #print("batch_key:", config.batch_key)
    #print("len batches:", len(config.batches))

    model, counts_tensor = integrador.train_cvae_on_counts(
            model_matrix,
            df_obs=adata.obs,
            batch_key=config.batch_key,
            cell_type_col=config.cell_type_col,
            cell_types=cell_types,
            ref_batch=config.reference_dictionary,
            verbose=1,
            beta = config.beta,
            epochs_CG_start  = config.epochs_CG_start,
            conf_T_init  = config.conf_T_init,
            conf_T_max_decay = config.conf_T_max_decay,
            lambda_size  = config.lambda_size,
            gamma_tau_align = config.gamma_tau_align,
            #alpha_list=[0.1, 0.4, 0.7],
            #alpha_objective="mean_max",
            #coverage_gap_target=0.01,
            #alpha_max_weight = 0.1
        )  

    conftr_plots_dir = config.output_dir / "conftr_training_plots"
    plot_training_curves(model, plots_dir=conftr_plots_dir)
    plot_conftr_diagnostics(model, plots_dir=conftr_plots_dir)

    model.eval()
    with torch.no_grad():
        mu_z, logvar_z = model.encode(
            counts_tensor.to(integrador.device),
            integrador.b_tensor.to(integrador.device),
            integrador.ct_tensor.to(integrador.device),
        )
        latent_np = mu_z.cpu().numpy()
        recon_mu, _ = model.decode(mu_z, integrador.b_tensor.to(integrador.device), integrador.ct_tensor.to(integrador.device))
        recon_np = recon_mu.cpu().numpy()
        logvar_np = logvar_z.cpu().numpy()  

        adata_conftr.obsm["X_pca_conftr"] = latent_np
        
    return adata_conftr.obsm["X_pca_conftr"].astype(np.float32)
    

METHOD_REGISTRY: Mapping[str, callable] = {
    "scvi": run_scvi,
    "scanvi": run_scanvi,
    "combat": run_combat,
    "scanorama": run_scanorama,
    "harmony": run_harmony,
    "conftr": run_conftr
}


def run_methods(
    adata: sc.AnnData,
    config: IntegrationConfig,
    *,
    run_name: Optional[str],
    name_outputs: bool,
    resume: bool = False,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    for method in config.methods:
        method_key = method.lower()
        if method_key not in METHOD_REGISTRY:
            logging.warning("Skipping unknown method '%s'", method)
            continue

        npy_path = method_embedding_path(config, method_key, run_name, name_outputs)
        if resume and npy_path.exists():
            try:
                existing = np.load(npy_path, allow_pickle=False)
                if existing.shape[0] == adata.n_obs and np.issubdtype(existing.dtype, np.number):
                    embeddings[method_key] = existing.astype(np.float32, copy=False)
                    logging.info("Resumed %s from %s", method_key, npy_path)
                    set_method_status(
                        config,
                        method_key,
                        run_name,
                        name_outputs,
                        "done",
                        extra={"embedding_path": str(npy_path), "resumed": True, "shape": list(existing.shape)},
                        manifest=manifest,
                        manifest_path=manifest_path,
                    )
                    continue
                logging.warning(
                    "Ignoring checkpoint for %s because shape %s does not match n_obs=%d",
                    method_key,
                    existing.shape,
                    adata.n_obs,
                )
            except Exception as exc:
                logging.warning("Could not load checkpoint for %s at %s: %s", method_key, npy_path, exc)

        runner = METHOD_REGISTRY[method_key]
        set_method_status(
            config,
            method_key,
            run_name,
            name_outputs,
            "running",
            manifest=manifest,
            manifest_path=manifest_path,
        )
        try:
            embedding = runner(adata, config)
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.shape[0] != adata.n_obs:
                raise ValueError(
                    f"Embedding rows for {method_key} ({embedding.shape[0]}) do not match n_obs={adata.n_obs}"
                )
            if not np.all(np.isfinite(embedding)):
                raise ValueError(f"Embedding for {method_key} contains NaN/Inf")
            embeddings[method_key] = embedding
            np.save(npy_path, embedding, allow_pickle=False)
            set_method_status(
                config,
                method_key,
                run_name,
                name_outputs,
                "done",
                extra={"embedding_path": str(npy_path), "resumed": False, "shape": list(embedding.shape)},
                manifest=manifest,
                manifest_path=manifest_path,
            )
            logging.info(
                "Finished %s | embedding shape = %s | checkpoint=%s",
                method_key,
                embeddings[method_key].shape,
                npy_path,
            )
        except Exception as exc:
            logging.exception("Method %s failed; continuing with remaining methods.", method_key)
            set_method_status(
                config,
                method_key,
                run_name,
                name_outputs,
                "failed",
                extra={
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                manifest=manifest,
                manifest_path=manifest_path,
            )
    return embeddings


def compute_umap_embeddings(
    base_adata: sc.AnnData,
    embeddings: Mapping[str, np.ndarray],
    config: IntegrationConfig,
    new_name: str = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
    umap_results: Dict[str, np.ndarray] = {}
    figure_paths: Dict[str, str] = {}

    for method, matrix in embeddings.items():
        embedding_key = f"X_{method}"
        adata_copy = base_adata.copy()
        adata_copy.obsm[embedding_key] = matrix

        sc.pp.neighbors(
            adata_copy,
            use_rep=embedding_key,
            n_neighbors=config.umap_n_neighbors,
            metric=config.umap_metric,
        )
        sc.tl.umap(
            adata_copy,
            min_dist=config.umap_min_dist,
            spread=config.umap_spread,
            random_state=config.seed,
        )

        umap_coords = adata_copy.obsm["X_umap"].astype(np.float32, copy=True)
        umap_results[method] = umap_coords

        fig = sc.pl.umap(
            adata_copy,
            color=[config.batch_key, config.cell_type_col],
            return_fig=True,
            show=False,
        )
        # Ensure unique filenames across hyperparameter runs and methods
        # When new_name is provided (e.g., exp + hyperparam), include it in the filename
        if new_name:
            fig_path = config.output_dir / f"umap_{new_name}_{method}.png"
        else:
            fig_path = config.output_dir / f"umap_{method}.png"

        fig_to_save = None
        if fig is None:
            logging.warning("Could not generate UMAP figure for method '%s'.", method)
        elif hasattr(fig, "savefig"):
            fig_to_save = fig
        elif hasattr(fig, "figure"):
            fig_to_save = fig.figure

        if fig_to_save is not None:
            fig_to_save.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig_to_save)
            figure_paths[method] = str(fig_path)
            logging.info("Saved UMAP visualization for %s to %s", method, fig_path)
        else:
            logging.warning("Unsupported figure object returned for method '%s'.", method)

    return umap_results, figure_paths


def export_results(
    base_adata: sc.AnnData,
    embeddings: Mapping[str, np.ndarray],
    config: IntegrationConfig,
    umap_embeddings: Optional[Mapping[str, np.ndarray]] = None,
    umap_figures: Optional[Mapping[str, str]] = None,
    metrics: Optional[pd.DataFrame] = None,   # <--- NUEVO
    run_name: Optional[str] = None,
) -> Path:
    

    output_adata = base_adata.copy()
    for method, matrix in embeddings.items():
        key = f"X_{method}"
        output_adata.obsm[key] = matrix

        if umap_embeddings is not None and method in umap_embeddings:
            umap_key = f"X_umap_{method}"
            output_adata.obsm[umap_key] = umap_embeddings[method]

    output_adata.uns["integration_config"] = json.loads(json.dumps(_serialize_config(config)))

    if umap_figures:
        output_adata.uns["integration_umap_figures"] = dict(umap_figures)

    # --- NUEVO: métricas en .uns (JSON-izable) ---
    if metrics is not None:
        output_adata.uns["integration_metrics"] = _metrics_df_to_uns_dict(metrics)

    output_prefix = f"{run_name}_" if run_name else ""
    output_path = config.output_dir / f"{output_prefix}adata_processed_batch_embeddings.h5ad"
    output_adata.write(output_path, compression="gzip")
    logging.info("Saved combined embeddings to %s", output_path)

    for method, matrix in embeddings.items():
        npy_path = config.output_dir / f"{output_prefix}{method}_embedding.npy"
        np.save(npy_path, matrix, allow_pickle=False)
        logging.info("Saved %s embedding matrix to %s", method, npy_path)

    return output_path



def get_pre_integration_representation(adata: sc.AnnData, config: IntegrationConfig) -> np.ndarray:
    """
    Construye una representación PRE-integración (PCA sobre la capa log-normalizada)
    para comparar con los embeddings integrados mediante PCR.
    """
    ad = adata.copy()
    # Garantiza que trabajamos con la capa log-normalizada que ya creas en load_and_prepare_data
    ad.X = ad.layers[config.lognorm_layer].astype(np.float32, copy=True)
    sc.pp.pca(ad, n_comps=config.pca_components, svd_solver="arpack")
    return ad.obsm["X_pca"][:, : config.pca_components].astype(np.float32)

def compute_integration_metrics(
    base_adata: sc.AnnData,
    embeddings: Mapping[str, np.ndarray],
    config: IntegrationConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cal_indices: List[int],
    b_array: np.ndarray,
    ct_array: np.ndarray,
    ref_to_targets_idx: Dict[int, List[int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula métricas scIB/scib-metrics para cada embedding en 'embeddings'.
    Devuelve un DataFrame indexado por método y guarda un CSV con los resultados.
    """

    batches_codes = b_array.astype(int)
    labels_codes = ct_array.astype(int)

    batches = base_adata.obs[config.batch_key].astype(str).to_numpy()
    labels  = base_adata.obs[config.cell_type_col].astype(str).to_numpy()

    train_idx = np.asarray(train_loader.dataset.indices, dtype=int) if train_loader is not None else None
    val_idx = np.asarray(val_loader.dataset.indices, dtype=int) if val_loader is not None else None
    
    # PRE: representación previa a la integración (para PCR), si procede
    pre_rep = None
    if config.metrics_compute_pcr:
        pre_rep = get_pre_integration_representation(base_adata, config)

    import jax
    print(jax.devices())

    rows = []
    coverage_rows = []
    for method, X in embeddings.items():
        row = {"method": method}

        if config.metrics_compute_scib:
            if config.metrics_neighbors_backend == "pynndescent":
                nn = sm.nearest_neighbors.pynndescent(
                    X,
                    n_neighbors=config.metrics_k,
                    random_state=config.seed,
                )
            elif config.metrics_neighbors_backend == "jax":
                nn = sm.nearest_neighbors.jax_approx_min_k(
                    X,
                    n_neighbors=config.metrics_k,
                )
            else:
                raise ValueError(
                    f"Unknown metrics_neighbors_backend={config.metrics_neighbors_backend!r}; "
                    "expected 'jax' or 'pynndescent'."
                )

            cluster_scores = sm.nmi_ari_cluster_labels_leiden(
                nn,
                labels_codes,
                optimize_resolution=True,
                seed=config.seed,
            )

            row.update(
                {
                    "iLISI": _as_scalar(sm.ilisi_knn(nn, batches_codes)),
                    "cLISI": _as_scalar(sm.clisi_knn(nn, labels_codes)),
                    "kBET": _as_scalar(sm.kbet(nn, batches_codes)),
                    "ASW_label": _as_scalar(sm.silhouette_label(X, labels, rescale=True)),
                    "ASW_batch": _as_scalar(sm.silhouette_batch(X, labels, batch=batches, rescale=True)),
                    "graph_connectivity": _as_scalar(sm.graph_connectivity(nn, labels_codes)),
                    "nmi": _as_scalar(cluster_scores["nmi"]),
                    "ari": _as_scalar(cluster_scores["ari"]),
                }
            )
            if config.metrics_isolated_labels:
                row["isolated_labels"] = _as_scalar(sm.isolated_labels(X, labels, batch=batches, rescale=True))

        row["logreg_covgap"] = np.nan
        row["logreg_train_target_covgap"] = np.nan
        row["logreg_avg_size"] = np.nan
        if config.metrics_compute_logreg and cal_indices is not None and train_idx is not None and val_idx is not None:
            try:
                # Comprobación básica de rangos

                target_coverage = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
                alpha_list = [1.0 - c for c in target_coverage]
                logreg_eval = evaluate_latent_logreg(
                    np.asarray(X, dtype=np.float32),          # latent_epoch_np
                    train_idx ,
                    val_idx,
                    b_array,          
                    ct_array,          
                    ref_to_targets_idx,
                    cal_indices,
                    target_coverage,
                    gap_objective="max",
                    return_details=True,
                )

                #print(covgap)
                covgap = logreg_eval["covgap"]
                row["logreg_covgap"] = _as_scalar(covgap)
                method_coverage_rows = logreg_eval.get("coverage", [])
                train_target_gaps = []
                for coverage_row in method_coverage_rows:
                    coverage_row = dict(coverage_row)
                    coverage_row["method"] = method
                    coverage_rows.append(coverage_row)
                    if np.isclose(
                        coverage_row.get("target_coverage", np.nan),
                        TRAINING_LOGREG_TARGET_COVERAGE,
                    ):
                        train_target_gaps.append(coverage_row.get("coverage_gap", np.nan))
                if train_target_gaps:
                    row["logreg_train_target_covgap"] = _as_scalar(train_target_gaps)
                if method_coverage_rows:
                    row["logreg_avg_size"] = _as_scalar(
                        [r.get("avg_set_size", np.nan) for r in method_coverage_rows]
                    )



            except Exception as e:
                logging.warning("LogReg eval failed for %s: %s", method, e)
                row["logreg_covgap"] = np.nan
                row["logreg_train_target_covgap"] = np.nan
                row["logreg_avg_size"] = np.nan


        # 5) PCR (batch variance reduction PRE vs POST)
        row["PCR"] = np.nan  # valor por defecto si no aplica o falla
        if pre_rep is not None and config.metrics_compute_pcr:
            try:
               

                # Asegura tipos numéricos
                X_pre  = np.asarray(pre_rep, dtype=np.float32)
                X_post = np.asarray(X,        dtype=np.float32)

                # Vector 1D de códigos de batch (categoría -> int)
                batch_codes = pd.Categorical(batches).codes.astype(np.int32)

                # Filtra posibles -1 (NaN) en batch
                valid = batch_codes >= 0
                X_pre_sub  = X_pre[valid]
                X_post_sub = X_post[valid]
                cov_sub    = batch_codes[valid]

                # Si solo queda un batch, PCR no tiene sentido
                if pd.Series(cov_sub).nunique(dropna=True) > 1:
                    pcr_val = pcr_comparison(
                        X_pre_sub,
                        X_post_sub,
                        covariate=cov_sub,   # vector 1D
                        categorical=True,    # que scib-metrics haga el one-hot
                        scale=True,
                    )
                    # <- AQUÍ va exactamente lo que preguntabas:
                    row["PCR"] = _as_scalar(pcr_val)
            except Exception as e:
                logging.warning("PCR failed for %s: %s", method, e)
                row["PCR"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("method").sort_index()
    coverage_df = pd.DataFrame(coverage_rows)

    # Guarda CSV con resultados
    csv_path = Path(config.output_dir) / config.metrics_output_csv
    df.to_csv(csv_path, float_format="%.6f")
    logging.info("Saved metrics table to %s", csv_path)

    coverage_csv_name = config.metrics_output_csv.replace(
        "integration_metrics",
        "empirical_coverage",
        1,
    )
    if coverage_csv_name == config.metrics_output_csv:
        coverage_csv_name = f"empirical_coverage_{config.metrics_output_csv}"
    coverage_csv_path = Path(config.output_dir) / coverage_csv_name
    coverage_df.to_csv(coverage_csv_path, index=False, float_format="%.6f")
    logging.info("Saved empirical coverage table to %s", coverage_csv_path)

    return df, coverage_df

def _metrics_df_to_uns_dict(df: pd.DataFrame) -> dict:
    out = {}
    for method, row in df.iterrows():
        out[str(method)] = {str(k): _as_scalar(v) for k, v in row.items()}
    return out

def _serialize_config(config: IntegrationConfig) -> Dict[str, object]:
    cfg = asdict(config)
    cfg["data_path"] = str(config.data_path)
    cfg["output_dir"] = str(config.output_dir)
    if cfg.get("preprocess_cache_dir") is not None:
        cfg["preprocess_cache_dir"] = str(config.preprocess_cache_dir)
    cfg["methods"] = list(config.methods)
    if cfg.get("batches_filter") is not None:
        cfg["batches_filter"] = list(cfg["batches_filter"])
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch integration comparison experiments",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Ruta al archivo JSON de configuración de experimentos.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        default="all",
        help="Nombre del experimento a ejecutar (debe ser una clave en el JSON).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra explícitamente el directorio de salida del experimento antes de ejecutar.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reutiliza embeddings/checkpoints ya existentes cuando tengan la forma esperada.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el plan de ejecución sin cargar datos ni entrenar modelos.",
    )
    parser.add_argument(
        "--use-preprocess-cache",
        action="store_true",
        help="Activa cache de AnnData ya filtrado/HVG/log-normalizado.",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="No calcula UMAPs ni figuras UMAP.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="No calcula métricas de integración.",
    )
    parser.add_argument(
        "--skip-full-h5ad",
        action="store_true",
        help="No exporta el .h5ad combinado final.",
    )
    parser.add_argument(
        "--metrics-neighbors-backend",
        choices=("jax", "pynndescent"),
        default=None,
        help=(
            "Nearest-neighbor backend for scIB metrics. "
            "Default is the fast JAX backend; use 'pynndescent' for a CPU-based, lower-memory fallback."
        ),
    )

    return parser.parse_args()



def _resolve_config_path(value: Any, config_dir: Path) -> Any:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def json_args(config_dict, experiment_name, config_dir: Optional[Path] = None) -> IntegrationConfig:
    default_params = config_dict.get("default_params", {})
    experiment_params = config_dict.get(experiment_name)
    if experiment_params is None:
        available = [key for key in config_dict.keys() if key != "default_params"]
        raise KeyError(
            f"Experiment '{experiment_name}' not found in config. Available: {available}"
        )

    final_config = default_params.copy()
    final_config.update(experiment_params)
    if config_dir is not None:
        if "data_path" in final_config:
            final_config["data_path"] = _resolve_config_path(final_config["data_path"], config_dir)
        if "output_dir" in final_config:
            final_config["output_dir"] = _resolve_config_path(final_config["output_dir"], config_dir)
        if final_config.get("preprocess_cache_dir") is not None:
            final_config["preprocess_cache_dir"] = _resolve_config_path(
                final_config["preprocess_cache_dir"],
                config_dir,
            )

    return IntegrationConfig(**final_config)


def save_embeddings_to_adata(
    adata: sc.AnnData,
    embeddings: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    prefix: str = "X_",           # nombre en .obsm => X_<método>
    float_dtype = np.float32,     # reduce tamaño de archivo
    umaps: dict[str, np.ndarray] | None = None,  # opcional: UMAPs por método
) -> Path:
    """
    Inserta cada embedding en adata.obsm[f'{prefix}{method}'] y guarda un .h5ad.

    Parameters
    ----------
    adata : AnnData base (no se modifica in-place; se copia para guardar)
    embeddings : dict {method -> (n_cells, d)}
    out_path : ruta del .h5ad de salida
    prefix : prefijo para las claves en .obsm (por defecto "X_")
    float_dtype : dtype al guardar las matrices (p.ej. np.float32)
    umaps : opcional dict {method -> (n_cells, 2)} para guardar proyecciones UMAP
            como .obsm[f'X_umap_{method}']
    """
    out_path = Path(out_path)
    ad = adata.copy()

    saved = []
    for method, X in embeddings.items():
        X = np.asarray(X)
        if X.shape[0] != ad.n_obs:
            raise ValueError(
                f"[{method}] filas del embedding ({X.shape[0]}) != n_obs ({ad.n_obs})"
            )
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"[{method}] dtype no numérico: {X.dtype}")
        if np.any(~np.isfinite(X)):
            raise ValueError(f"[{method}] contiene NaN/Inf")

        key = f"{prefix}{method}"
        ad.obsm[key] = X.astype(float_dtype, copy=False)
        saved.append(key)

        # UMAP opcional
        if umaps is not None and method in umaps:
            U = np.asarray(umaps[method])
            if U.shape[0] != ad.n_obs or U.shape[1] != 2:
                raise ValueError(f"[{method}] UMAP debe ser (n_cells, 2); got {U.shape}")
            ad.obsm[f"X_umap_{method}"] = U.astype(float_dtype, copy=False)

    # Metadatos mínimos para saber qué se guardó
    ad.uns.setdefault("saved_embeddings", saved)

    # Guardar comprimido
    ad.write(out_path, compression="gzip")
    return out_path

def apply_single_value_hyperparams(config: IntegrationConfig) -> None:
    for key, value_list in config.hyperparams.items():
        if isinstance(value_list, list) and len(value_list) == 1:
            setattr(config, key, value_list[0])


def build_hyperparam_grid(config: IntegrationConfig) -> list[dict[str, Any]]:
    grid_params = {
        key: value
        for key, value in config.hyperparams.items()
        if isinstance(value, list) and len(value) > 1
    }
    if not grid_params:
        return [{}]

    grid_keys = list(grid_params.keys())
    grid_values = list(grid_params.values())
    return [
        dict(zip(grid_keys, combo_values))
        for combo_values in itertools.product(*grid_values)
    ]


def format_grid_name(exp_name: str, experiment_params: Mapping[str, Any]) -> str:
    if not experiment_params:
        return exp_name

    param_parts = []
    for key, value in experiment_params.items():
        param_parts.extend([key, str(value)])
    return f"{exp_name}_{'_'.join(param_parts)}"


def run_single_experiment(
    config: IntegrationConfig,
    run_name: str,
    *,
    name_outputs: bool,
    export_full_results: bool,
    resume: bool,
    manifest: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    output_name = run_name if name_outputs else None
    config.metrics_output_csv = (
        f"integration_metrics_{run_name}.csv" if name_outputs else "integration_metrics.csv"
    )

    set_global_seeds(config.seed)

    if manifest is not None and manifest_path is not None:
        run_state = manifest.setdefault("runs", {}).setdefault(run_name, {"methods": {}})
        run_state["status"] = "running"
        run_state["started_at"] = run_state.get("started_at") or _utc_now()
        update_run_manifest(manifest_path, manifest)

    base_adata = load_and_prepare_data_cached(config)

    logging.info("Accelerator: %s | Devices: %s", config.accelerator, config.lightning_devices)

    embeddings = run_methods(
        base_adata,
        config,
        run_name=run_name,
        name_outputs=name_outputs,
        resume=resume,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if not embeddings:
        logging.error("No methods completed for %s; skipping metrics, UMAP and export.", run_name)
        if manifest is not None and manifest_path is not None:
            manifest.setdefault("runs", {}).setdefault(run_name, {"methods": {}})["status"] = "failed"
            update_run_manifest(manifest_path, manifest)
        return None, None

    metrics_df = None
    coverage_df = None
    if config.metrics_compute:
        data_preparation = DataPreparation()
        data_preparation.data_loader_prep(
            obs_df=base_adata.obs,
            batch_key=config.batch_key,
            cell_type_col=config.cell_type_col,
            reference_dictionary=config.reference_dictionary,
        )

        train_loader = None
        val_loader = None
        cal_indices = None
        if config.metrics_compute_logreg:
            train_cvae_batch_size_default = config.metrics_batch_size
            calibration_fraction_default = 0.35
            build_data_val_fraction_default = 0.2

            print("\nBuilding data tensors for logreg metrics...")
            model_matrix = base_adata.layers[config.lognorm_layer]
            if sparse.issparse(model_matrix):
                model_matrix = model_matrix.toarray()
            else:
                model_matrix = np.asarray(model_matrix)
            model_matrix = model_matrix.astype(np.float32, copy=False)
            _, train_loader, val_loader, cal_indices = data_preparation.build_data_tensors(
                model_matrix,
                seed_offset=config.seed,
                batch_size=train_cvae_batch_size_default,
                calibration_fraction=calibration_fraction_default,
                val_fraction=build_data_val_fraction_default,
            )

        metrics_df, coverage_df = compute_integration_metrics(
            base_adata,
            embeddings,
            config,
            train_loader,
            val_loader,
            cal_indices,
            data_preparation.b_tensor.cpu().numpy(),
            data_preparation.ct_tensor.cpu().numpy(),
            data_preparation.ref_to_targets_idx,
        )
        logging.info("Metrics summary:\n%s", metrics_df.round(3).to_string())

    umap_embeddings = None
    umap_figures = None
    if config.compute_umap:
        umap_embeddings, umap_figures = compute_umap_embeddings(
            base_adata,
            embeddings,
            config,
            output_name,
        )

    if export_full_results:
        export_results(
            base_adata,
            embeddings,
            config,
            umap_embeddings,
            umap_figures,
            metrics=metrics_df,
            run_name=output_name,
        )

    if manifest is not None and manifest_path is not None:
        run_state = manifest.setdefault("runs", {}).setdefault(run_name, {"methods": {}})
        run_state["status"] = "done"
        run_state["finished_at"] = _utc_now()
        update_run_manifest(manifest_path, manifest)

    return metrics_df, coverage_df


def add_metrics_metadata(
    metrics_df: pd.DataFrame,
    *,
    experiment_name: str,
    variant_name: str,
    run_name: str,
    repetition: int,
    seed: int,
    hyperparams: Mapping[str, Any],
) -> pd.DataFrame:
    metrics = metrics_df.reset_index()
    metrics.insert(0, "experiment", experiment_name)
    metrics.insert(1, "variant", variant_name)
    metrics.insert(2, "run_name", run_name)
    metrics.insert(3, "repeat", repetition)
    metrics.insert(4, "seed", seed)
    for key, value in hyperparams.items():
        metrics[key] = value
    return metrics


def add_coverage_metadata(
    coverage_df: pd.DataFrame,
    *,
    experiment_name: str,
    variant_name: str,
    run_name: str,
    repetition: int,
    seed: int,
    hyperparams: Mapping[str, Any],
) -> pd.DataFrame:
    coverage = coverage_df.copy()
    coverage.insert(0, "experiment", experiment_name)
    coverage.insert(1, "variant", variant_name)
    coverage.insert(2, "run_name", run_name)
    coverage.insert(3, "repeat", repetition)
    coverage.insert(4, "seed", seed)
    for key, value in hyperparams.items():
        coverage[key] = value
    return coverage


def save_repetition_metrics(
    config: IntegrationConfig,
    experiment_name: str,
    metrics_tables: list[pd.DataFrame],
) -> None:
    if not metrics_tables:
        return

    metrics_all = pd.concat(metrics_tables, ignore_index=True)
    all_path = config.output_dir / f"metrics_all_repetitions_{experiment_name}.csv"
    metrics_all.to_csv(all_path, index=False, float_format="%.6f")

    metadata_cols = {
        "experiment",
        "variant",
        "run_name",
        "repeat",
        "seed",
        "method",
        *config.hyperparams.keys(),
    }
    metric_cols = [
        col
        for col in metrics_all.select_dtypes(include=[np.number]).columns
        if col not in metadata_cols
    ]
    if not metric_cols:
        logging.info("Saved repeated metrics to %s", all_path)
        return

    summary = (
        metrics_all
        .groupby(["experiment", "variant", "method"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary_path = config.output_dir / f"metrics_summary_{experiment_name}.csv"
    summary.to_csv(summary_path, float_format="%.6f")

    logging.info("Saved repeated metrics to %s", all_path)
    logging.info("Saved repeated metrics summary to %s", summary_path)


def save_repetition_coverage(
    config: IntegrationConfig,
    experiment_name: str,
    coverage_tables: list[pd.DataFrame],
) -> None:
    if not coverage_tables:
        return

    coverage_all = pd.concat(coverage_tables, ignore_index=True)
    all_path = config.output_dir / f"empirical_coverage_all_repetitions_{experiment_name}.csv"
    coverage_all.to_csv(all_path, index=False, float_format="%.6f")

    summary_cols = ["empirical_coverage", "coverage_gap", "avg_set_size"]
    available_summary_cols = [col for col in summary_cols if col in coverage_all.columns]
    if not available_summary_cols:
        logging.info("Saved empirical coverage details to %s", all_path)
        return

    group_cols = ["experiment", "variant", "method", "target_coverage", "alpha"]
    summary = (
        coverage_all
        .groupby(group_cols, dropna=False)[available_summary_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary_path = config.output_dir / f"empirical_coverage_summary_{experiment_name}.csv"
    summary.to_csv(summary_path, float_format="%.6f")

    logging.info("Saved empirical coverage details to %s", all_path)
    logging.info("Saved empirical coverage summary to %s", summary_path)


def validate_config(config: IntegrationConfig) -> None:
    errors = []
    if not config.data_path.exists():
        errors.append(f"data_path does not exist: {config.data_path}")
    if not config.methods:
        errors.append("methods cannot be empty")
    unknown_methods = [method for method in config.methods if method.lower() not in METHOD_REGISTRY]
    if unknown_methods:
        logging.warning("Unknown methods will be skipped: %s", ", ".join(unknown_methods))
    if not config.reference_dictionary:
        errors.append("reference_dictionary cannot be empty")
    else:
        for ref, targets in config.reference_dictionary.items():
            duplicated_targets = [target for target in targets if target == ref]
            if duplicated_targets:
                logging.warning(
                    "Reference batch %r also appears in its target list; duplicate values have no effect on filtering.",
                    ref,
                )
    if config.metrics_neighbors_backend not in {"jax", "pynndescent"}:
        errors.append("metrics_neighbors_backend must be 'jax' or 'pynndescent'")
    if errors:
        raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))


def apply_cli_overrides(config: IntegrationConfig, args: argparse.Namespace) -> None:
    if args.use_preprocess_cache:
        config.preprocess_cache = True
    if args.skip_umap:
        config.compute_umap = False
    if args.skip_metrics:
        config.metrics_compute = False
    if args.skip_full_h5ad:
        config.export_full_h5ad = False
    if args.metrics_neighbors_backend is not None:
        config.metrics_neighbors_backend = args.metrics_neighbors_backend


def warn_jax_metrics_backend_if_needed(config: IntegrationConfig) -> None:
    if (
        config.metrics_compute
        and config.metrics_compute_scib
        and config.metrics_neighbors_backend == "jax"
    ):
        logging.warning(
            "Using the default JAX nearest-neighbor backend for scIB metrics. "
            "This is usually the fastest option, but JAX will use the GPU when available and may allocate "
            "large temporary buffers for approximate kNN graph construction. On large datasets, or after "
            "GPU-heavy training steps, this can cause CUDA out-of-memory errors. "
            "Use --metrics-neighbors-backend pynndescent to run the neighbor search on CPU with lower GPU "
            "memory pressure, at the cost of slower metric computation."
        )


def print_dry_run_plan(
    exp_name: str,
    config: IntegrationConfig,
    grid_configs: list[dict[str, Any]],
    n_repetitions: int,
) -> None:
    print(f"\nDry run for {exp_name}")
    print(f"  data_path: {config.data_path}")
    print(f"  output_dir: {config.output_dir}")
    print(f"  methods: {', '.join(config.methods)}")
    print(f"  repetitions: {n_repetitions}")
    print(f"  grid variants: {len(grid_configs)}")
    for experiment_params in grid_configs:
        variant_name = format_grid_name(exp_name, experiment_params)
        for rep in range(n_repetitions):
            seed = config.seed + rep
            run_name = f"{variant_name}_rep{rep}_seed{seed}" if n_repetitions > 1 else variant_name
            print(f"  - {run_name}: seed={seed}, params={experiment_params or '{}'}")


def initialize_manifest(config: IntegrationConfig, args: argparse.Namespace, exp_name: str, runtime: Mapping[str, Any]) -> tuple[Dict[str, Any], Path]:
    manifest_path = config.output_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Could not read existing manifest at %s; creating a new one.", manifest_path)
            manifest = {}
    else:
        manifest = {}

    manifest.update(
        {
            "schema_version": 1,
            "experiment_name": exp_name,
            "created_at": manifest.get("created_at") or _utc_now(),
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "config": _serialize_config(config),
            "runtime": runtime,
            "resume": bool(args.resume),
            "overwrite": bool(args.overwrite),
        }
    )
    manifest.setdefault("runs", {})
    update_run_manifest(manifest_path, manifest)
    return manifest, manifest_path


def main() -> None:
    print("Running main...")
    args = parse_args()
    print("args:", args.config_file, args.experiment_name)

    setup_logging()
    runtime = collect_runtime_environment()
    log_runtime_environment(runtime)

    with open(args.config_file, "r") as f:
        config_dict = json.load(f)
    config_dir = args.config_file.resolve().parent

    if args.experiment_name == "all":
        print("Running all experiments.")
        experiment_names = [key for key in config_dict.keys() if key != "default_params"]
    else:
        experiment_names = [args.experiment_name]

    for exp_name in experiment_names:
        print(f"\nRunning experiment: {exp_name}\n")

        config = json_args(config_dict, exp_name, config_dir=config_dir)
        apply_single_value_hyperparams(config)
        apply_cli_overrides(config, args)
        validate_config(config)

        if args.overwrite:
            if config.output_dir.exists():
                logging.warning("Overwriting output directory: %s", config.output_dir)
                shutil.rmtree(config.output_dir)
            config.output_dir.mkdir(parents=True, exist_ok=True)
        elif config.output_dir.exists() and not args.resume:
            logging.info(
                "Output directory exists and will be reused without deleting: %s. "
                "Use --overwrite to delete it or --resume to reuse checkpoints.",
                config.output_dir,
            )

        base_seed = config.seed
        n_repetitions = max(1, int(getattr(config, "n_repetitions", 1)))
        grid_configs = build_hyperparam_grid(config)
        has_grid = len(grid_configs) > 1
        name_outputs = has_grid or n_repetitions > 1
        all_metrics_tables = []
        all_coverage_tables = []

        if args.dry_run:
            print_dry_run_plan(exp_name, config, grid_configs, n_repetitions)
            continue

        try:
            shutil.copy2(args.config_file, config.output_dir / "source_config.json")
        except Exception as exc:
            logging.warning("Could not copy source config into output directory: %s", exc)

        manifest, manifest_path = initialize_manifest(config, args, exp_name, runtime)

        for experiment_params in grid_configs:
            for key, value in experiment_params.items():
                setattr(config, key, value)

            warn_jax_metrics_backend_if_needed(config)

            variant_name = format_grid_name(exp_name, experiment_params)

            for rep in range(n_repetitions):
                config.seed = base_seed + rep
                run_name = (
                    f"{variant_name}_rep{rep}_seed{config.seed}"
                    if n_repetitions > 1
                    else variant_name
                )

                print(f"Running experiment {run_name}")
                logging.info("Running %s | seed=%s", run_name, config.seed)

                metrics_df, coverage_df = run_single_experiment(
                    config,
                    run_name,
                    name_outputs=name_outputs,
                    export_full_results=(not has_grid) and config.export_full_h5ad,
                    resume=args.resume,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )

                if metrics_df is not None:
                    all_metrics_tables.append(
                        add_metrics_metadata(
                            metrics_df,
                            experiment_name=exp_name,
                            variant_name=variant_name,
                            run_name=run_name,
                            repetition=rep,
                            seed=config.seed,
                            hyperparams=experiment_params,
                        )
                    )
                if coverage_df is not None and not coverage_df.empty:
                    all_coverage_tables.append(
                        add_coverage_metadata(
                            coverage_df,
                            experiment_name=exp_name,
                            variant_name=variant_name,
                            run_name=run_name,
                            repetition=rep,
                            seed=config.seed,
                            hyperparams=experiment_params,
                        )
                    )

        save_repetition_metrics(config, exp_name, all_metrics_tables)
        save_repetition_coverage(config, exp_name, all_coverage_tables)
        logging.info("All methods completed successfully.\n")




def cli() -> None:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()


if __name__ == "__main__":
    cli()
