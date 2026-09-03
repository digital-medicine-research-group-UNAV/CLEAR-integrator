# CLEAR Integration

CLEAR is a supervised single-cell batch integration model built around the existing conditional VAE and conformal training objective used by the experiment scripts in this folder. This repository layout packages that implementation without changing the scientific training logic.

## Installation

From this directory:

```bash
pip install -e .
```

The optional differentiable sorting backend can be installed with:

```bash
pip install -e ".[torchsort]"
```

## Minimal Usage

```python
import scanpy as sc
from clear_integration import CLEARIntegrationModel

adata = sc.read_h5ad("path/to/data.h5ad")

model = CLEARIntegrationModel(
    adata,
    batch_key="batch",
    cell_type_col="cell_type",
    layer="lognorm_gaussian",
    reference_dictionary={"reference_batch": ["target_batch"]},
    epochs=350,
    batch_size=1024,
)
model.train(verbose=1)
adata.obsm["X_clear"] = model.get_latent_representation()
model.save("results/clear_model.pt")
```

## Paper Experiments

Run paper-style benchmark experiments from this repository root.

### Entry Points

Primary entry point:

```bash
python paper/run_experiments.py --config-file paper/immune-config.json --experiment-name experimento-batch-coverage
```

Repository-root compatibility wrapper:

```bash
python run_experiments.py --config-file paper/immune-config.json --experiment-name experimento-batch-coverage
```

Installed console script:

```bash
clear-run-experiments --config-file paper/immune-config.json --experiment-name experimento-batch-coverage
```

If you are using the parent `pixi` environment from `estancia/scripts`, run from that directory:

```bash
pixi run python experiments/CLEAR/paper/run_experiments.py \
  --config-file experiments/CLEAR/paper/immune-config.json \
  --experiment-name experimento-batch-coverage
```

Or run from this `CLEAR` directory with the same environment:

```bash
pixi run python paper/run_experiments.py \
  --config-file paper/immune-config.json \
  --experiment-name experimento-batch-coverage
```

### CLI Options

`--config-file PATH`:
Required. JSON file containing `default_params` and one or more named experiments. Relative paths inside the JSON, such as `data_path` and `output_dir`, are resolved relative to the JSON file location.

`--experiment-name NAME`:
Experiment key to run from the JSON file. If omitted, the runner uses `all` and executes every top-level experiment except `default_params`.

`--overwrite`:
Deletes the selected experiment output directory before running. Use this when you want a clean run and do not want to reuse any previous outputs. This is destructive for that output directory.

`--resume`:
Reuses already saved method outputs/checkpoints when their expected files and shapes are valid. Use this after an interrupted run, or after a metrics/UMAP failure, to avoid retraining completed methods.

`--dry-run`:
Prints the execution plan without loading data or training models. This is useful for checking which experiments, repetitions, seeds, and hyperparameter-grid variants will run.

`--use-preprocess-cache`:
Enables reuse of cached preprocessed AnnData when `preprocess_cache_dir` is configured. This can save time on repeated runs with the same filtering/HVG/log-normalization setup.

`--skip-umap`:
Skips UMAP computation and UMAP figures. Embeddings and metrics can still be produced.

`--skip-metrics`:
Skips integration metrics. Use this when you only need embeddings/UMAPs or when metric computation is failing because of memory limits.

`--skip-full-h5ad`:
Skips exporting the combined `.h5ad` with all embeddings. Individual `.npy` embeddings are still saved.

`--metrics-neighbors-backend {jax,pynndescent}`:
Selects the nearest-neighbor backend used by scIB metrics. `jax` is the default and is usually faster, but can allocate large GPU buffers and cause CUDA OOM on large datasets. `pynndescent` runs the neighbor search on CPU with lower GPU memory pressure, but is slower. This affects kNN-based metrics such as iLISI, cLISI, kBET, graph connectivity, NMI, and ARI; it does not change training or embeddings.

### Common Launch Patterns

Check the plan without running:

```bash
python paper/run_experiments.py \
  --config-file paper/kidsney-config.json \
  --experiment-name experimento-batch-coverage \
  --dry-run
```

Resume a run after interruption:

```bash
python paper/run_experiments.py \
  --config-file paper/kidsney-config.json \
  --experiment-name experimento-batch-coverage \
  --resume
```

Resume and recompute metrics with lower GPU memory pressure:

```bash
python paper/run_experiments.py \
  --config-file paper/kidsney-config.json \
  --experiment-name experimento-batch-coverage \
  --resume \
  --metrics-neighbors-backend pynndescent
```

Run all experiments defined in a config:

```bash
python paper/run_experiments.py --config-file paper/immune-config.json
```

Run without metrics:

```bash
python paper/run_experiments.py \
  --config-file paper/kidsney-config.json \
  --experiment-name experimento-batch-coverage \
  --skip-metrics
```

Run without UMAPs and without full `.h5ad` export:

```bash
python paper/run_experiments.py \
  --config-file paper/kidsney-config.json \
  --experiment-name experimento-batch-coverage \
  --skip-umap \
  --skip-full-h5ad
```

Start from a clean output directory:

```bash
python paper/run_experiments.py \
  --config-file paper/kidsney-config.json \
  --experiment-name experimento-batch-coverage \
  --overwrite
```

### JSON Configuration

The config file has a `default_params` block plus experiment-specific blocks. The final runtime config is built as:

```text
default_params + selected_experiment_overrides + CLI_overrides
```

Important experiment fields:

`data_path`:
Input `.h5ad` file.

`output_dir`:
Directory where embeddings, metrics, manifests, UMAPs, and training plots are written.

`batch_key`:
Column in `adata.obs` identifying batches/domains.

`cell_type_col`:
Column in `adata.obs` identifying supervised biological labels.

`methods`:
Methods to run. Supported values are `conftr`, `harmony`, `scanvi`, `scvi`, `combat`, and `scanorama`.

`reference_dictionary`:
Mapping from reference batch to target batches for CLEAR/ConfTr calibration and filtering. Example:

```json
{
  "healthy_6": ["control_1", "control_2", "healthy_6", "diabetic_2"]
}
```

`hyperparams`:
Dictionary of hyperparameters for grid runs. A single-value list is applied directly. Lists with more than one value create a Cartesian product of variants. Variant outputs are named with the hyperparameter values.

Key CLEAR/ConfTr hyperparameters:

`beta` controls KL weighting in the VAE objective.

`epochs_CG_start` controls when conformal training starts.

`conf_T_init` and `conf_T_max_decay` define the conformal temperature schedule.

`lambda_size` controls the set-size penalty.

`gamma_tau_align` controls class-conditional threshold alignment across batches.

`batch_size` controls CLEAR/ConfTr training batch size.

Data/preprocessing config:

`subset_max_cells` optionally limits the number of cells after loading. `0` means use all available cells.

`seed` controls random seeds for splitting, model initialization, and stochastic methods where supported.

`n_repetitions` repeats the selected experiment with seeds `seed + repetition_index`.

`min_cells_per_type` filters rare supervised labels before integration.

`n_top_genes` controls highly variable gene selection.

`counts_layer` names the raw/count layer used where count models need it.

`lognorm_layer` names the log-normalized layer used by CLEAR/ConfTr and some evaluations.

Baseline model config:

`pca_components` controls PCA dimensionality for preprocessing/PCR.

`scvi_latent_dim`, `scvi_max_epochs`, `scvi_batch_size`, and `scvi_gene_likelihood` configure scVI.

`scanvi_latent_dim`, `scanvi_max_epochs`, `scanvi_batch_size`, `scanvi_unlabeled_fraction`, and `scanvi_unlabeled_category` configure scANVI.

`conftr_batch_size` optionally overrides the CLEAR/ConfTr batch size. If unset, the runner uses `batch_size`.

Metric-related config:

`metrics_compute` enables or disables all metrics.

`metrics_compute_scib` enables scIB metrics.

`metrics_compute_logreg` enables held-out latent logistic-regression conformal evaluation.

`metrics_compute_pcr` enables PCR.

`metrics_isolated_labels` enables isolated-label scoring.

`metrics_k` controls kNN size for local metrics.

`metrics_neighbors_backend` selects `jax` or `pynndescent` for scIB neighbor search.

`metrics_batch_size` controls the DataLoader batch size used for the held-out logistic-regression conformal evaluation.

Output-related config:

`compute_umap` controls UMAP computation and figures.

`umap_n_neighbors`, `umap_min_dist`, `umap_spread`, and `umap_metric` configure UMAP figures.

`export_full_h5ad` controls final combined `.h5ad` export.

`preprocess_cache` and `preprocess_cache_dir` control preprocessing cache reuse.

The runner writes `run_manifest.json` and `source_config.json` into the output directory so each run records the command, resolved configuration, runtime environment, and method status.

## Repository Layout

```text
.
├── pyproject.toml
├── README.md
├── paper/
│   ├── immune-config.json
│   ├── kidsney-config.json
│   └── run_experiments.py
├── results/
├── run_experiments.py
└── src/
    └── clear_integration/
        ├── data/
        ├── model/
        ├── module/
        ├── nn/
        ├── paper/
        ├── train/
        └── utils/
```

`run_experiments.py` is a compatibility wrapper. New code should import from `clear_integration` or use `paper/run_experiments.py`.
