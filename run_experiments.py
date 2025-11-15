#!/usr/bin/env python
"""
Compare multiple batch integration strategies (scANVI, ComBat, Scanorama, scVI, Harmony, Fountain)
on the MB_processed dataset and export the learned embeddings for downstream analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, List, Optional, Any
import shutil
import os
import itertools

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import torch
import matplotlib.pyplot as plt

import numbers
from collections.abc import Mapping, Sequence

from reilable_integrator import Integrator, DataPreparation

from scipy import sparse
from scvi import model as scvi_model
import scib_metrics as sm
from scib_metrics import pcr_comparison

@dataclass
class IntegrationConfig:
    """Runtime configuration for the integration comparison."""
    
    description: str = "Batch integration experiment"
    data_path: Path = None #Path(__file__).resolve().parents[2] / "data" / "diabetic-kidney-disease_processed.h5ad"
    output_dir: Path = None#Path(__file__).resolve().parent / "embeddings"
    subset_max_cells: Optional[int] = None
    methods: tuple[str, ...] = None #("scanvi", "combat", "scanorama", "scvi", "harmony", "conftr")
    seed: int = 123
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
    
    pca_components: int = 50
    scvi_latent_dim: int = 16
    scvi_max_epochs: int = 300
    scvi_batch_size: int = 256
    scvi_gene_likelihood: str = "zinb"
    scanvi_latent_dim: int = 16
    scanvi_max_epochs: int = 300
    scanvi_batch_size: int = 256
    scanvi_unlabeled_fraction: float = 0.1
    scanvi_unlabeled_category: str = "Unknown"
    
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    umap_metric: str = "euclidean"
    

    # Evaluation metrics
    metrics_k: int = 30                 # vecinos kNN para métricas locales (LISI/kBET/ASW)
    metrics_compute: bool = True        # calcula métricas al final
    metrics_compute_pcr: bool = True    # calcula PCR (requiere una representación "pre")
    metrics_isolated_labels: bool = True
    metrics_output_csv: str = "integration_metrics.csv"

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
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
        return 1 if self.accelerator == "gpu" else "auto"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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
    if config.subset_max_cells is not None and config.subset_max_cells < adata.n_obs:
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
        batch_size=config.scvi_batch_size,
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

    model = scvi_model.SCANVI(
        adata_model,
        n_hidden = 256,
        n_latent=config.scanvi_latent_dim,
    )

    model.train(
        max_epochs=config.scanvi_max_epochs,
        accelerator=config.accelerator,
        devices=config.lightning_devices,
        batch_size=config.scanvi_batch_size,
    )
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


    train_cvae_epochs_default = 600
    train_cvae_lr_default = 5e-4
    train_cvae_kl_anneal_epochs_default = 1
    train_cvae_batch_size_default = config.batch_size #1024
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
        device="cpu"
    )

    cell_types = adata.obs[config.cell_type_col].astype(str).unique().tolist() #list of cell types


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
            verbose=False,
            beta = config.beta,
            epochs_CG_start  = config.epochs_CG_start,
            conf_T_init  = config.conf_T_init,
            conf_T_max_decay = config.conf_T_max_decay,
            lambda_size  = config.lambda_size,
            gamma_tau_align = config.gamma_tau_align,
        )  

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


def run_methods(adata: sc.AnnData, config: IntegrationConfig) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    for method in config.methods:
        method_key = method.lower()
        if method_key not in METHOD_REGISTRY:
            logging.warning("Skipping unknown method '%s'", method)
            continue
        runner = METHOD_REGISTRY[method_key]
        embeddings[method_key] = runner(adata, config)
        logging.info(
            "Finished %s | embedding shape = %s",
            method_key,
            embeddings[method_key].shape,
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

    output_path = config.output_dir / "adata_processed_batch_embeddings.h5ad"
    output_adata.write(output_path, compression="gzip")
    logging.info("Saved combined embeddings to %s", output_path)

    for method, matrix in embeddings.items():
        npy_path = config.output_dir / f"{method}_embedding.npy"
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
) -> pd.DataFrame:
    """
    Calcula métricas scIB/scib-metrics para cada embedding en 'embeddings'.
    Devuelve un DataFrame indexado por método y guarda un CSV con los resultados.
    """
    batches = base_adata.obs[config.batch_key].astype(str).to_numpy()
    labels  = base_adata.obs[config.cell_type_col].astype(str).to_numpy()

    train_idx = np.asarray(train_loader.dataset.indices, dtype=int)
    val_idx = np.asarray(val_loader.dataset.indices, dtype=int)
    
    # PRE: representación previa a la integración (para PCR), si procede
    pre_rep = None
    if config.metrics_compute_pcr:
        pre_rep = get_pre_integration_representation(base_adata, config)

    rows = []
    for method, X in embeddings.items():
        # 1) Vecinos aproximados sobre el embedding integrado
        nn = sm.nearest_neighbors.pynndescent(
            X, n_neighbors=config.metrics_k, random_state=config.seed
        )

        # 2) Métricas de mezcla de batch y conservación biológica
        #    (todas en la dirección "más alto = mejor")
        row = {
            "method": method,
            "iLISI": _as_scalar(sm.ilisi_knn(nn, batches)),
            "cLISI": _as_scalar(sm.clisi_knn(nn, labels)),
            "kBET":  _as_scalar(sm.kbet(nn, batches)),   # <- aquí estaba el problema
            "ASW_label": _as_scalar(sm.silhouette_label(X, labels, rescale=True)),
            "ASW_batch": _as_scalar(sm.silhouette_batch(X, labels, batch=batches, rescale=True)),
            "graph_connectivity": _as_scalar(sm.graph_connectivity(nn, labels)),
        }
        row.update({
            "nmi": _as_scalar(sm.nmi_ari_cluster_labels_leiden(nn, labels, optimize_resolution=True, seed=config.seed)["nmi"]),
            "ari": _as_scalar(sm.nmi_ari_cluster_labels_leiden(nn, labels, optimize_resolution=True, seed=config.seed)["ari"]),
        })
        if config.metrics_isolated_labels:
            row["isolated_labels"] = _as_scalar(sm.isolated_labels(X, labels, batch=batches, rescale=True))

        row["logreg_covgap"] = np.nan
        row["logreg_avg_size"] = np.nan
        if cal_indices is not None:
            from reilable_integrator import evaluate_latent_logreg
            try:
                # Comprobación básica de rangos

                target_coverage = [0.90]
                covgap = evaluate_latent_logreg(
                    np.asarray(X, dtype=np.float32),          # latent_epoch_np
                    train_idx ,
                    val_idx,
                    b_array,          
                    ct_array,          
                    ref_to_targets_idx,
                    cal_indices,
                    target_coverage,
                )

                row["logreg_covgap"] = _as_scalar(covgap)
                #row["logreg_avg_size"] = _as_scalar(avg_size)



            except Exception as e:
                logging.warning("LogReg eval failed for %s: %s", method, e)
                row["logreg_covgap"] = np.nan
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

    # Guarda CSV con resultados
    csv_path = Path(config.output_dir) / config.metrics_output_csv
    df.to_csv(csv_path, float_format="%.6f")
    logging.info("Saved metrics table to %s", csv_path)

    return df

def _metrics_df_to_uns_dict(df: pd.DataFrame) -> dict:
    out = {}
    for method, row in df.iterrows():
        out[str(method)] = {str(k): _as_scalar(v) for k, v in row.items()}
    return out

def _serialize_config(config: IntegrationConfig) -> Dict[str, object]:
    cfg = asdict(config)
    cfg["data_path"] = str(config.data_path)
    cfg["output_dir"] = str(config.output_dir)
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

    return parser.parse_args()



def json_args(config_dict, experiment_name) ->IntegrationConfig:

    default_params = config_dict.get("default_params", {})

    experiment_params = config_dict.get(experiment_name)
    
    final_config = default_params.copy()
    final_config.update(experiment_params)

    return IntegrationConfig(**final_config)


#def build_config_from_args(args: argparse.Namespace) -> IntegrationConfig:
    config_kwargs = {}
    if args.data_path is not None:
        config_kwargs["data_path"] = args.data_path
    if args.output_dir is not None:
        config_kwargs["output_dir"] = args.output_dir
    if args.max_cells is not None:
        config_kwargs["subset_max_cells"] = args.max_cells
    if args.methods is not None:
        config_kwargs["methods"] = tuple(m.lower() for m in args.methods)
    if args.seed is not None:
        config_kwargs["seed"] = args.seed
    if args.min_cells_per_type is not None:
        config_kwargs["min_cells_per_type"] = args.min_cells_per_type
    if args.n_top_genes is not None:
        config_kwargs["n_top_genes"] = None if args.n_top_genes <= 0 else args.n_top_genes
    return IntegrationConfig(**config_kwargs)


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


def main() -> None:
    print("Running main...")
    args = parse_args()
    print("args:", args.config_file, args.experiment_name)

    with open(args.config_file, 'r') as f:
        config_dict = json.load(f)

    if args.experiment_name == "all":
        print("Running all experiments.")
        experiment_names = [key for key in config_dict.keys() if key != "default_params"]
    else:
        experiment_names = [args.experiment_name]

    
    for exp_name in experiment_names:

        print(f"\nRunning experiment: {exp_name}\n")

        config = json_args(config_dict, exp_name)
        
        # we assign single-value hyperparameters directly to attributes
        for key, value_list in config.hyperparams.items():
            if isinstance(value_list, list) and len(value_list) == 1:
                single_value = value_list[0]
                setattr(config, key, single_value) # Asignar valor único como atributo e.g config.beta = single_value

        params_list = [clave for clave, lista_valor in config.hyperparams.items() if len(lista_valor) > 1]

        if len(params_list) > 0:

            grid_params = {key: val for key, val in config.hyperparams.items() if len(val) > 1}
            grid_keys = list(grid_params.keys())
            grid_values = list(grid_params.values())

            all_configs_as_dicts = []

            # All combinations of hyperparameter values
            for combo_values in itertools.product(*grid_values):
                # Creamos un diccionario para esta combinación específica
                combo_dict = dict(zip(grid_keys, combo_values))
                all_configs_as_dicts.append(combo_dict)
            
            for i, experiment_params in enumerate(all_configs_as_dicts):

                param_parts = []
                for key, value in experiment_params.items():
                    setattr(config, key, value)

                    param_parts.append(key)
                    formatted_value = str(value)#.replace('.', 'p')
                    param_parts.append(formatted_value)
                
                params_string = "_".join(param_parts)
                new_exp_name = f"{exp_name}_{params_string}"
                

                

                print(f"Running experiment {new_exp_name}")
                
                
                config.metrics_output_csv = "integration_metrics_" + new_exp_name + ".csv"
                

                setup_logging()
                set_global_seeds(config.seed)

                base_adata = load_and_prepare_data(config)

                data_preparation = DataPreparation()
                
                data_preparation.data_loader_prep(
                    obs_df = base_adata.obs,
                    batch_key = config.batch_key,
                    cell_type_col = config.cell_type_col,
                    reference_dictionary = config.reference_dictionary
                )
                
                train_cvae_batch_size_default = 256
                calibration_fraction_default = 0.35  # Set >0 to hold out a calibration set (e.g., 0.4)
                build_data_val_fraction_default = 0.2

                print("\nBuilding data tensors...")
                model_matrix = base_adata.layers[config.lognorm_layer]
                if sparse.issparse(model_matrix):
                    model_matrix = model_matrix.toarray()
                else:
                    model_matrix = np.asarray(model_matrix)
                model_matrix = model_matrix.astype(np.float32, copy=False)
                counts_tensor, train_loader, val_loader, cal_indices = data_preparation.build_data_tensors(
                    model_matrix,
                    seed_offset=config.seed,
                    batch_size=train_cvae_batch_size_default,
                    calibration_fraction=calibration_fraction_default,
                    val_fraction = build_data_val_fraction_default
                )
            

                logging.info("Accelerator: %s | Devices: %s", config.accelerator, config.lightning_devices)

                
                embeddings = run_methods(base_adata, config)


                # --- métricas ---
                metrics_df = None
                if config.metrics_compute:
                    metrics_df = compute_integration_metrics(base_adata,
                                                            embeddings,
                                                            config,
                                                            train_loader,
                                                            val_loader,
                                                            cal_indices,
                                                            data_preparation.b_tensor.cpu().numpy(),
                                                            data_preparation.ct_tensor.cpu().numpy(),
                                                            data_preparation.ref_to_targets_idx)
                    
                    # Opcional: imprimir un resumen corto
                    logging.info("Metrics summary:\n%s", metrics_df.round(3).to_string())

                umap_embeddings, umap_figures = compute_umap_embeddings(base_adata, embeddings, config, new_exp_name)


                #export_results(base_adata, embeddings, config, umap_embeddings, umap_figures, metrics=metrics_df)







        else:
        

            setup_logging()
            set_global_seeds(config.seed)

            base_adata = load_and_prepare_data(config)

            data_preparation = DataPreparation()
            
            data_preparation.data_loader_prep(
                obs_df = base_adata.obs,
                batch_key = config.batch_key,
                cell_type_col = config.cell_type_col,
                reference_dictionary = config.reference_dictionary
            )
            
            train_cvae_batch_size_default = 256
            calibration_fraction_default = 0.35  # Set >0 to hold out a calibration set (e.g., 0.4)
            build_data_val_fraction_default = 0.2

            print("\nBuilding data tensors...")
            model_matrix = base_adata.layers[config.lognorm_layer]
            if sparse.issparse(model_matrix):
                model_matrix = model_matrix.toarray()
            else:
                model_matrix = np.asarray(model_matrix)
            model_matrix = model_matrix.astype(np.float32, copy=False)
            counts_tensor, train_loader, val_loader, cal_indices = data_preparation.build_data_tensors(
                model_matrix,
                seed_offset=config.seed,
                batch_size=train_cvae_batch_size_default,
                calibration_fraction=calibration_fraction_default,
                val_fraction = build_data_val_fraction_default
            )
        

            logging.info("Accelerator: %s | Devices: %s", config.accelerator, config.lightning_devices)

            
            embeddings = run_methods(base_adata, config)




            # --- métricas ---
            metrics_df = None
            if config.metrics_compute:
                metrics_df = compute_integration_metrics(base_adata,
                                                        embeddings,
                                                        config,
                                                        train_loader,
                                                        val_loader,
                                                        cal_indices,
                                                        data_preparation.b_tensor.cpu().numpy(),
                                                        data_preparation.ct_tensor.cpu().numpy(),
                                                        data_preparation.ref_to_targets_idx)
                
                # Opcional: imprimir un resumen corto
                logging.info("Metrics summary:\n%s", metrics_df.round(3).to_string())

            umap_embeddings, umap_figures = compute_umap_embeddings(base_adata, embeddings, config)


            export_results(base_adata, embeddings, config, umap_embeddings, umap_figures, metrics=metrics_df)


        logging.info("All methods completed successfully.\n")




if __name__ == "__main__":
    main()
