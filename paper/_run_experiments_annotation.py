#!/usr/bin/env python
"""Semi-supervised annotation experiments for the CLEAR paper.

This runner is intentionally standalone: it reuses the existing CLEAR paper
configuration and preprocessing helpers, but does not modify the integration
implementation used by the original experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

np = None
pd = None
sc = None
torch = None
sparse = None
LogisticRegression = None
Integrator = None
IntegrationConfig = None
load_and_prepare_data_cached = None


def load_runtime_dependencies() -> None:
    """Import the heavy CLEAR runtime only when training/evaluation is requested."""

    global np
    global pd
    global sc
    global torch
    global sparse
    global LogisticRegression
    global Integrator
    global IntegrationConfig
    global load_and_prepare_data_cached

    if IntegrationConfig is not None:
        return

    import numpy as _np
    import pandas as _pd
    import scanpy as _sc
    import torch as _torch
    from scipy import sparse as _sparse
    from sklearn.linear_model import LogisticRegression as _LogisticRegression

    from clear_integration import Integrator as _Integrator
    from clear_integration.paper._run_experiments import (
        IntegrationConfig as _IntegrationConfig,
        load_and_prepare_data_cached as _load_and_prepare_data_cached,
    )

    np = _np
    pd = _pd
    sc = _sc
    torch = _torch
    sparse = _sparse
    LogisticRegression = _LogisticRegression
    Integrator = _Integrator
    IntegrationConfig = _IntegrationConfig
    load_and_prepare_data_cached = _load_and_prepare_data_cached


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def collect_runtime_environment() -> dict[str, Any]:
    packages = {}
    if torch is not None:
        packages["torch"] = getattr(torch, "__version__", "unknown")
        packages["torch_cuda"] = getattr(torch.version, "cuda", None)
    return {
        "captured_at": _utc_now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "environment": {"cwd": str(Path.cwd()), "argv": sys.argv},
        "torch_cuda": {
            "available": bool(torch is not None and torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch is not None else 0,
        },
    }


def log_runtime_environment(runtime: Mapping[str, Any]) -> None:
    packages = runtime.get("packages", {})
    torch_cuda = runtime.get("torch_cuda", {})
    logging.info(
        "Runtime: python=%s torch=%s torch_cuda=%s cuda_available=%s",
        runtime.get("python"),
        packages.get("torch", "not-imported"),
        packages.get("torch_cuda"),
        torch_cuda.get("available"),
    )


def _resolve_config_path(value: Any, config_dir: Path) -> Any:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        cfg = asdict(config)
    else:
        cfg = dict(vars(config))
    for key in ("data_path", "output_dir", "preprocess_cache_dir"):
        if cfg.get(key) is not None:
            cfg[key] = str(cfg[key])
    if cfg.get("methods") is not None:
        cfg["methods"] = list(cfg["methods"])
    return cfg


def set_global_seeds(seed: int) -> None:
    load_runtime_dependencies()
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def apply_single_value_hyperparams(config: Any) -> None:
    for key, value_list in getattr(config, "hyperparams", {}).items():
        if isinstance(value_list, list) and len(value_list) == 1:
            setattr(config, key, value_list[0])


def build_hyperparam_grid(config: Any) -> list[dict[str, Any]]:
    grid_params = {
        key: value
        for key, value in getattr(config, "hyperparams", {}).items()
        if isinstance(value, list) and len(value) > 1
    }
    if not grid_params:
        return [{}]
    import itertools

    keys = list(grid_params.keys())
    values = list(grid_params.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_grid_name(exp_name: str, experiment_params: Mapping[str, Any]) -> str:
    if not experiment_params:
        return exp_name
    parts = []
    for key, value in experiment_params.items():
        parts.extend([key, str(value)])
    return f"{exp_name}_{'_'.join(parts)}"


def validate_config(config: Any) -> None:
    errors = []
    if not Path(config.data_path).exists():
        errors.append(f"data_path does not exist: {config.data_path}")
    if not getattr(config, "methods", None):
        errors.append("methods cannot be empty")
    if not getattr(config, "reference_dictionary", None):
        errors.append("reference_dictionary cannot be empty")
    if errors:
        raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))


ANNOTATION_EXTRA_KEYS = {
    "target_holdout_fraction",
    "reference_calibration_fraction",
    "semi_target_calibration_fraction",
    "target_coverage",
    "annotation_schemes",
    "set_methods",
}

SUPPORTED_SCHEMES = ("simple", "semi_supervised")
SUPPORTED_SET_METHODS = ("threshold", "mondrian", "aps", "aps_mondrian")


@dataclass
class AnnotationConfig:
    """Annotation-specific options layered on top of IntegrationConfig."""

    target_holdout_fraction: float = 0.50
    reference_calibration_fraction: float = 0.35
    semi_target_calibration_fraction: float = 0.35
    target_coverage: float = 0.90
    annotation_schemes: tuple[str, ...] = field(default_factory=lambda: ("simple", "semi_supervised"))
    set_methods: tuple[str, ...] = field(
        default_factory=lambda: ("threshold", "mondrian", "aps", "aps_mondrian")
    )

    def __post_init__(self) -> None:
        self.target_holdout_fraction = float(self.target_holdout_fraction)
        self.reference_calibration_fraction = float(self.reference_calibration_fraction)
        self.semi_target_calibration_fraction = float(self.semi_target_calibration_fraction)
        self.target_coverage = float(self.target_coverage)
        self.annotation_schemes = tuple(_normalize_scheme_list(self.annotation_schemes))
        self.set_methods = tuple(_normalize_set_method_list(self.set_methods))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLEAR conformal annotation experiments",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        required=False,
        help="JSON configuration file with default_params and named experiments.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="all",
        help="Experiment key to run. Defaults to all experiments in the config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the selected experiment output directory before running.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed annotation runs when all expected output files are present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without loading data or training models.",
    )
    parser.add_argument(
        "--target-holdout-fraction",
        type=float,
        default=None,
        help="Override the target-batch holdout fraction used for unknown cells.",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=None,
        help="Override target conformal coverage. Example: 0.90 means alpha=0.10.",
    )
    parser.add_argument(
        "--annotation-schemes",
        nargs="+",
        default=None,
        help="Schemes to run: simple, semi_supervised. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--set-methods",
        nargs="+",
        default=None,
        help=(
            "Conformal set-construction variants to run: threshold, mondrian, aps, aps_mondrian. "
            "Comma-separated values are also accepted."
        ),
    )
    parser.add_argument(
        "--use-preprocess-cache",
        action="store_true",
        help="Reuse cached preprocessed AnnData when preprocess_cache_dir is configured.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight synthetic smoke test for annotation splitting and conformal scoring.",
    )
    args = parser.parse_args()
    if not args.smoke_test and args.config_file is None:
        parser.error("--config-file is required unless --smoke-test is used")
    return args


def _normalize_scheme_list(value: Any) -> list[str]:
    if value is None:
        return ["simple", "semi_supervised"]
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    schemes = [item.strip() for item in raw_items if item.strip()]
    unknown = [scheme for scheme in schemes if scheme not in SUPPORTED_SCHEMES]
    if unknown:
        raise ValueError(
            "Unknown annotation scheme(s): "
            + ", ".join(unknown)
            + f". Supported: {', '.join(SUPPORTED_SCHEMES)}"
        )
    return schemes or ["simple", "semi_supervised"]


def _normalize_set_method_list(value: Any) -> list[str]:
    if value is None:
        return list(SUPPORTED_SET_METHODS)
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    methods = [item.strip() for item in raw_items if item.strip()]
    unknown = [method for method in methods if method not in SUPPORTED_SET_METHODS]
    if unknown:
        raise ValueError(
            "Unknown conformal set method(s): "
            + ", ".join(unknown)
            + f". Supported: {', '.join(SUPPORTED_SET_METHODS)}"
        )
    return _unique_preserve_order(methods) or list(SUPPORTED_SET_METHODS)


def _annotation_options_from_payload(payload: Mapping[str, Any]) -> AnnotationConfig:
    values = {key: payload[key] for key in ANNOTATION_EXTRA_KEYS if key in payload}
    return AnnotationConfig(**values)


def _resolve_config_paths(final_config: dict[str, Any], config_dir: Path) -> None:
    if "data_path" in final_config:
        final_config["data_path"] = _resolve_config_path(final_config["data_path"], config_dir)
    if "output_dir" in final_config:
        final_config["output_dir"] = _resolve_config_path(final_config["output_dir"], config_dir)
    if final_config.get("preprocess_cache_dir") is not None:
        final_config["preprocess_cache_dir"] = _resolve_config_path(
            final_config["preprocess_cache_dir"],
            config_dir,
        )


def json_args_annotation(
    config_dict: Mapping[str, Any],
    experiment_name: str,
    *,
    config_dir: Path,
    require_runtime: bool,
) -> tuple[IntegrationConfig, AnnotationConfig]:
    default_params = dict(config_dict.get("default_params", {}))
    experiment_params = config_dict.get(experiment_name)
    if experiment_params is None:
        available = [key for key in config_dict.keys() if key != "default_params"]
        raise KeyError(f"Experiment {experiment_name!r} not found in config. Available: {available}")

    final_config = default_params.copy()
    final_config.update(experiment_params)
    annotation_options = _annotation_options_from_payload(final_config)
    for key in ANNOTATION_EXTRA_KEYS:
        final_config.pop(key, None)

    _resolve_config_paths(final_config, config_dir)
    if require_runtime:
        load_runtime_dependencies()
        integration_config = IntegrationConfig(**final_config)
    else:
        integration_config = SimpleNamespace(**final_config)
        integration_config.data_path = Path(integration_config.data_path).expanduser()
        integration_config.output_dir = Path(integration_config.output_dir).expanduser()
        integration_config.methods = tuple(getattr(integration_config, "methods", ()))
        integration_config.hyperparams = dict(getattr(integration_config, "hyperparams", {}))
        integration_config.seed = int(getattr(integration_config, "seed", 123))
        integration_config.n_repetitions = int(getattr(integration_config, "n_repetitions", 1))
        integration_config.reference_dictionary = dict(getattr(integration_config, "reference_dictionary", {}))
        integration_config.ref_batch = next(iter(integration_config.reference_dictionary), None)
        if integration_config.ref_batch is not None:
            integration_config.batches = (
                list(integration_config.reference_dictionary[integration_config.ref_batch])
                + [integration_config.ref_batch]
            )
        else:
            integration_config.batches = []
    return integration_config, annotation_options


def apply_cli_overrides(config: IntegrationConfig, annotation: AnnotationConfig, args: argparse.Namespace) -> None:
    if args.use_preprocess_cache:
        config.preprocess_cache = True
    if args.target_holdout_fraction is not None:
        annotation.target_holdout_fraction = float(args.target_holdout_fraction)
    if args.target_coverage is not None:
        annotation.target_coverage = float(args.target_coverage)
    if args.annotation_schemes is not None:
        annotation.annotation_schemes = tuple(_normalize_scheme_list(args.annotation_schemes))
    if args.set_methods is not None:
        annotation.set_methods = tuple(_normalize_set_method_list(args.set_methods))


def normalize_to_clear_only(config: IntegrationConfig) -> None:
    configured = tuple(str(method).lower() for method in config.methods)
    ignored = [method for method in configured if method != "conftr"]
    if ignored:
        logging.warning("Annotation runner trains only CLEAR/conftr; ignoring methods: %s", ", ".join(ignored))
    if "conftr" not in configured:
        logging.warning("Configured methods do not include conftr; running CLEAR/conftr for annotation.")
    config.methods = ("conftr",)


def validate_annotation_config(annotation: AnnotationConfig) -> None:
    errors = []
    if not 0.0 <= annotation.target_holdout_fraction < 1.0:
        errors.append("target_holdout_fraction must be in [0, 1)")
    if not 0.0 <= annotation.reference_calibration_fraction < 1.0:
        errors.append("reference_calibration_fraction must be in [0, 1)")
    if not 0.0 <= annotation.semi_target_calibration_fraction < 1.0:
        errors.append("semi_target_calibration_fraction must be in [0, 1)")
    if not 0.0 < annotation.target_coverage < 1.0:
        errors.append("target_coverage must be in (0, 1)")
    if not annotation.annotation_schemes:
        errors.append("annotation_schemes cannot be empty")
    if not annotation.set_methods:
        errors.append("set_methods cannot be empty")
    if errors:
        raise ValueError("Invalid annotation configuration:\n- " + "\n- ".join(errors))


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _stratified_holdout(
    obs: pd.DataFrame,
    indices: np.ndarray,
    *,
    group_cols: list[str],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return kept and held-out positional indices, preserving at least one kept cell per stratum."""

    kept: list[np.ndarray] = []
    held: list[np.ndarray] = []
    if indices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    group_frame = obs.iloc[indices][group_cols].astype(str)
    grouped = group_frame.groupby(group_cols, sort=True, observed=False).indices
    for relative_positions in grouped.values():
        group_indices = indices[np.asarray(relative_positions, dtype=int)]
        group_indices = np.asarray(group_indices, dtype=int)
        if group_indices.size <= 1 or fraction <= 0.0:
            kept.append(group_indices)
            continue
        n_holdout = int(np.floor(group_indices.size * fraction))
        n_holdout = max(1, n_holdout)
        n_holdout = min(n_holdout, group_indices.size - 1)
        selected = rng.choice(group_indices, size=n_holdout, replace=False)
        selected = np.asarray(selected, dtype=int)
        remaining = np.setdiff1d(group_indices, selected, assume_unique=False)
        kept.append(remaining)
        held.append(selected)

    kept_arr = np.sort(np.concatenate(kept)) if kept else np.asarray([], dtype=int)
    held_arr = np.sort(np.concatenate(held)) if held else np.asarray([], dtype=int)
    return kept_arr, held_arr


def make_annotation_splits(
    adata: sc.AnnData,
    config: IntegrationConfig,
    annotation: AnnotationConfig,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = adata.obs
    batch_values = obs[config.batch_key].astype(str).to_numpy()

    ref_batch = config.ref_batch
    configured_targets = config.reference_dictionary.get(ref_batch, [])
    target_batches = _unique_preserve_order([str(batch) for batch in configured_targets if str(batch) != ref_batch])

    ref_all = np.where(batch_values == ref_batch)[0].astype(int)
    target_all = np.where(np.isin(batch_values, target_batches))[0].astype(int)

    ref_train, ref_cal = _stratified_holdout(
        obs,
        ref_all,
        group_cols=[config.cell_type_col],
        fraction=annotation.reference_calibration_fraction,
        rng=rng,
    )
    target_known, target_unknown = _stratified_holdout(
        obs,
        target_all,
        group_cols=[config.batch_key, config.cell_type_col],
        fraction=annotation.target_holdout_fraction,
        rng=rng,
    )
    target_fit, target_cal = _stratified_holdout(
        obs,
        target_known,
        group_cols=[config.batch_key, config.cell_type_col],
        fraction=annotation.semi_target_calibration_fraction,
        rng=rng,
    )

    known = np.sort(np.concatenate([ref_train, ref_cal, target_known])).astype(int)
    return {
        "ref_all": ref_all,
        "ref_train": ref_train,
        "ref_cal": ref_cal,
        "target_all": target_all,
        "target_known": target_known,
        "target_unknown": target_unknown,
        "target_fit": target_fit,
        "target_cal": target_cal,
        "known": known,
    }


def _split_summary(adata: sc.AnnData, config: IntegrationConfig, splits: Mapping[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sizes": {key: int(np.asarray(value).size) for key, value in splits.items()}}
    for key in ("ref_train", "ref_cal", "target_known", "target_unknown", "target_fit", "target_cal"):
        idx = np.asarray(splits[key], dtype=int)
        if idx.size == 0:
            summary[f"{key}_by_label"] = {}
            continue
        summary[f"{key}_by_label"] = (
            adata.obs.iloc[idx][config.cell_type_col].astype(str).value_counts().sort_index().to_dict()
        )
    return summary


def _matrix_from_layer(adata: sc.AnnData, layer: str) -> np.ndarray:
    matrix = adata.X if layer == "X" else adata.layers[layer]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
    return matrix.astype(np.float32, copy=False)


def _present_reference_dictionary(config: IntegrationConfig, obs: pd.DataFrame) -> dict[str, list[str]]:
    present_batches = set(obs[config.batch_key].astype(str).unique().tolist())
    cleaned = {}
    for ref, targets in config.reference_dictionary.items():
        if ref not in present_batches:
            raise ValueError(f"Reference batch {ref!r} is absent from the known-cell training data.")
        cleaned[str(ref)] = [str(target) for target in targets if str(target) in present_batches]
    return cleaned


def train_clear_known(
    adata_known: sc.AnnData,
    config: IntegrationConfig,
) -> tuple[Integrator, torch.nn.Module, torch.Tensor, np.ndarray, list[str]]:
    logging.info("Training CLEAR on %d known cells", adata_known.n_obs)
    model_matrix = _matrix_from_layer(adata_known, config.lognorm_layer)

    train_cvae_epochs_default = 350
    train_cvae_lr_default = 5e-4
    train_cvae_kl_anneal_epochs_default = 1
    train_cvae_batch_size_default = config.conftr_batch_size or config.batch_size
    calibration_fraction_default = 0.35
    build_data_val_fraction_default = 0.2

    integrator = Integrator(
        seed_offset=config.seed,
        epochs=train_cvae_epochs_default,
        lr=train_cvae_lr_default,
        kl_anneal_epochs=train_cvae_kl_anneal_epochs_default,
        batch_size=train_cvae_batch_size_default,
        calibration_fraction=calibration_fraction_default,
        val_fraction=build_data_val_fraction_default,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    cell_types = adata_known.obs[config.cell_type_col].astype(str).unique().tolist()
    reference_dictionary = _present_reference_dictionary(config, adata_known.obs)
    model, counts_tensor = integrator.train_cvae_on_counts(
        model_matrix,
        df_obs=adata_known.obs,
        batch_key=config.batch_key,
        cell_type_col=config.cell_type_col,
        cell_types=cell_types,
        ref_batch=reference_dictionary,
        verbose=1,
        beta=config.beta,
        epochs_CG_start=config.epochs_CG_start,
        conf_T_init=config.conf_T_init,
        conf_T_max_decay=config.conf_T_max_decay,
        lambda_size=config.lambda_size,
        gamma_tau_align=config.gamma_tau_align,
    )

    model.eval()
    with torch.no_grad():
        latent, _ = model.encode(
            counts_tensor.to(integrator.device),
            integrator.b_tensor.to(integrator.device),
            integrator.ct_tensor.to(integrator.device),
        )
    return integrator, model, counts_tensor, latent.cpu().numpy().astype(np.float32, copy=False), cell_types


def _original_to_known_positions(known_indices: np.ndarray) -> dict[int, int]:
    return {int(original_idx): int(position) for position, original_idx in enumerate(known_indices.tolist())}


def _positions_for_original_indices(original_indices: np.ndarray, position_map: Mapping[int, int]) -> np.ndarray:
    positions = [position_map[int(idx)] for idx in np.asarray(original_indices, dtype=int) if int(idx) in position_map]
    return np.asarray(positions, dtype=int)


def _fit_conformal_classifier(
    known_latent: np.ndarray,
    known_labels: np.ndarray,
    train_positions: np.ndarray,
) -> tuple[Optional[LogisticRegression], str]:
    """Fit the latent-space label classifier shared by every conformal set method."""

    train_positions = np.asarray(train_positions, dtype=int)
    train_labels = known_labels[train_positions]
    if np.unique(train_labels).size < 2:
        return None, "skipped: classifier training needs at least two classes"

    clf = LogisticRegression(max_iter=10000)
    clf.fit(known_latent[train_positions], train_labels)
    return clf, "ok"


# ---------------------------------------------------------------------------
# Conformal calibration primitives.
#
# Two orthogonal toggles produce the four supported variants:
#   * set rule:      "threshold" (per-label probability cutoff) vs "aps"
#                    (Adaptive Prediction Sets over the normalized score vector).
#   * calibration:   marginal (one quantile) vs Mondrian (one quantile per class).
#
# Notation: ``S[i, c]`` is the candidate-conditioned probability
# ``P(c | encode(x_i, batch_i, c))``. APS additionally uses the row-normalized
# ``pi[i, c] = S[i, c] / sum_c S[i, c]``.
# ---------------------------------------------------------------------------


def _marginal_quantile(nonconformity_scores: np.ndarray, *, target_coverage: float) -> float:
    """Split-conformal quantile q with P(score <= q) >= target_coverage on calibration.

    An empty calibration set returns +inf, i.e. every candidate is accepted.
    """

    alpha = 1.0 - target_coverage
    scores = np.asarray(nonconformity_scores, dtype=np.float64)
    n = scores.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(rank, n - 1))
    return float(np.sort(scores)[rank])


def _class_quantiles(
    nonconformity_scores: np.ndarray,
    true_cols: np.ndarray,
    *,
    n_candidates: int,
    target_coverage: float,
    fallback: float,
) -> np.ndarray:
    """Per-class (Mondrian) conformal quantiles; classes without calibration use ``fallback``."""

    scores = np.asarray(nonconformity_scores, dtype=np.float64)
    true_cols = np.asarray(true_cols, dtype=int)
    q = np.full(n_candidates, float(fallback), dtype=np.float64)
    for c in range(n_candidates):
        mask = true_cols == c
        if mask.any():
            q[c] = _marginal_quantile(scores[mask], target_coverage=target_coverage)
    return q


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Row-normalize candidate scores into a probability vector (zero rows stay zero)."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.reshape(scores.shape)
    row_sum = scores.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 0.0, row_sum, 1.0)
    return scores / safe


def _aps_calibration_scores(pi_cal: np.ndarray, true_cols: np.ndarray) -> np.ndarray:
    """APS nonconformity: cumulative normalized mass down to (and incl.) the true class.

    Deterministic / conservative convention: classes tied with the true class are
    placed after it (true class counted once), so coverage is never under-stated.
    """

    pi_cal = np.asarray(pi_cal, dtype=np.float64)
    true_cols = np.asarray(true_cols, dtype=int)
    n = pi_cal.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = pi_cal[i]
        t = int(true_cols[i])
        out[i] = float(p[p > p[t]].sum() + p[t])
    return out


def _set_threshold(score_row: np.ndarray, q_marginal: float) -> list[int]:
    """Marginal threshold rule: include c iff (1 - S[c]) <= q (may be empty)."""

    return [int(c) for c in range(score_row.size) if (1.0 - float(score_row[c])) <= q_marginal]


def _set_threshold_mondrian(score_row: np.ndarray, q_per_class: np.ndarray) -> list[int]:
    """Class-conditional threshold rule: include c iff (1 - S[c]) <= q_c (may be empty)."""

    return [int(c) for c in range(score_row.size) if (1.0 - float(score_row[c])) <= float(q_per_class[c])]


def _set_aps(pi_row: np.ndarray, q_marginal: float) -> list[int]:
    """Marginal APS: add labels in descending pi until cumulative >= q. Never empty."""

    order = np.argsort(-pi_row, kind="stable")
    included: list[int] = []
    cumulative = 0.0
    for c in order:
        included.append(int(c))
        cumulative += float(pi_row[c])
        if cumulative >= q_marginal:
            break
    return included


def _set_aps_mondrian(pi_row: np.ndarray, q_per_class: np.ndarray) -> list[int]:
    """Class-conditional APS: include c iff its cumulative-to-c mass <= q_c; always keep arg-max."""

    included: list[int] = []
    for c in range(pi_row.size):
        cumulative_to_c = float(pi_row[pi_row > pi_row[c]].sum() + pi_row[c])
        if cumulative_to_c <= float(q_per_class[c]):
            included.append(int(c))
    top = int(np.argmax(pi_row)) if pi_row.size else -1
    if top >= 0 and top not in included:
        included.append(top)
    return sorted(included)


def _batch_and_cell_type_maps(adata_known: sc.AnnData, config: IntegrationConfig, cell_types: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    batches = sorted(adata_known.obs[config.batch_key].astype(str).unique().tolist())
    batch_to_idx = {batch: idx for idx, batch in enumerate(batches)}
    cell_type_to_idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
    return batch_to_idx, cell_type_to_idx


def _candidate_conditioned_scores(
    adata: sc.AnnData,
    eval_indices: np.ndarray,
    config: IntegrationConfig,
    adata_known: sc.AnnData,
    model: torch.nn.Module,
    integrator: Integrator,
    clf: LogisticRegression,
    cell_types: list[str],
    candidate_labels: list[str],
) -> np.ndarray:
    eval_indices = np.asarray(eval_indices, dtype=int)
    if eval_indices.size == 0:
        return np.zeros((0, len(candidate_labels)), dtype=np.float32)

    batch_to_idx, cell_type_to_idx = _batch_and_cell_type_maps(adata_known, config, cell_types)
    class_to_col = {str(label): pos for pos, label in enumerate(clf.classes_)}

    matrix = _matrix_from_layer(adata[eval_indices].copy(), config.lognorm_layer)
    eval_batches = adata.obs.iloc[eval_indices][config.batch_key].astype(str).to_numpy()
    missing_batches = sorted(set(eval_batches) - set(batch_to_idx))
    if missing_batches:
        raise ValueError(
            "Cannot encode unknown cells from batches absent during CLEAR training: "
            + ", ".join(missing_batches)
        )
    batch_codes = np.asarray([batch_to_idx[batch] for batch in eval_batches], dtype=np.int64)

    scores = np.zeros((eval_indices.size, len(candidate_labels)), dtype=np.float32)
    batch_size = int(config.conftr_batch_size or config.batch_size or 1024)
    device = integrator.device
    model.eval()

    for candidate_pos, label in enumerate(candidate_labels):
        if label not in cell_type_to_idx or label not in class_to_col:
            continue
        label_code = int(cell_type_to_idx[label])
        class_col = int(class_to_col[label])
        label_scores: list[np.ndarray] = []
        for start in range(0, eval_indices.size, batch_size):
            stop = min(start + batch_size, eval_indices.size)
            x_tensor = torch.from_numpy(matrix[start:stop]).float().to(device, non_blocking=True)
            b_tensor = torch.from_numpy(batch_codes[start:stop]).long().to(device, non_blocking=True)
            y_tensor = torch.full((stop - start,), label_code, dtype=torch.long, device=device)
            with torch.no_grad():
                latent, _ = model.encode(x_tensor, b_tensor, y_tensor)
            probs = clf.predict_proba(latent.cpu().numpy().astype(np.float32, copy=False))
            label_scores.append(probs[:, class_col].astype(np.float32, copy=False))
        scores[:, candidate_pos] = np.concatenate(label_scores) if label_scores else np.asarray([], dtype=np.float32)

    return scores


def _prediction_set_string(labels: list[str]) -> str:
    return "|".join(labels)


def _empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cell_id",
            "batch",
            "true_label",
            "prediction_set",
            "set_size",
            "contains_true",
            "top_label",
            "top_probability",
        ]
    )


def _predictions_from_index_sets(
    adata: sc.AnnData,
    eval_indices: np.ndarray,
    config: IntegrationConfig,
    candidate_labels: list[str],
    scores: np.ndarray,
    index_sets: list[list[int]],
) -> pd.DataFrame:
    """Assemble the prediction table from per-cell included candidate indices."""

    pred_rows = []
    eval_obs = adata.obs.iloc[eval_indices]
    for row_pos, original_idx in enumerate(eval_indices.tolist()):
        row_scores = scores[row_pos]
        included = [candidate_labels[c] for c in index_sets[row_pos]]
        top_idx = int(np.argmax(row_scores)) if row_scores.size else -1
        top_label = candidate_labels[top_idx] if top_idx >= 0 else ""
        true_label = str(eval_obs.iloc[row_pos][config.cell_type_col])
        pred_rows.append(
            {
                "cell_id": str(adata.obs_names[original_idx]),
                "batch": str(eval_obs.iloc[row_pos][config.batch_key]),
                "true_label": true_label,
                "prediction_set": _prediction_set_string(included),
                "set_size": int(len(included)),
                "contains_true": bool(true_label in included),
                "top_label": top_label,
                "top_probability": float(row_scores[top_idx]) if top_idx >= 0 else np.nan,
            }
        )
    return pd.DataFrame(pred_rows)


def evaluate_scheme(
    *,
    scheme: str,
    adata: sc.AnnData,
    adata_known: sc.AnnData,
    config: IntegrationConfig,
    annotation: AnnotationConfig,
    splits: Mapping[str, np.ndarray],
    known_indices: np.ndarray,
    known_latent: np.ndarray,
    model: torch.nn.Module,
    integrator: Integrator,
    cell_types: list[str],
    candidate_labels: list[str],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Evaluate one scheme under every requested conformal set-construction method.

    Returns a mapping ``set_method -> (predictions, metrics, metadata)``. The
    classifier and the candidate-conditioned score matrices for the calibration
    and evaluation cells are computed once and shared across all methods.
    """

    position_map = _original_to_known_positions(known_indices)
    known_labels = adata_known.obs[config.cell_type_col].astype(str).to_numpy()
    reference_label_set = set(candidate_labels)
    cand_to_idx = {label: idx for idx, label in enumerate(candidate_labels)}
    target_coverage = annotation.target_coverage
    methods = list(annotation.set_methods)
    n_candidates = len(candidate_labels)

    if scheme == "simple":
        train_original = splits["ref_train"]
        cal_original = splits["ref_cal"]
    elif scheme == "semi_supervised":
        target_fit = np.asarray(splits["target_fit"], dtype=int)
        target_cal = np.asarray(splits["target_cal"], dtype=int)
        target_fit = target_fit[
            adata.obs.iloc[target_fit][config.cell_type_col].astype(str).isin(reference_label_set).to_numpy()
        ]
        target_cal = target_cal[
            adata.obs.iloc[target_cal][config.cell_type_col].astype(str).isin(reference_label_set).to_numpy()
        ]
        train_original = np.sort(np.concatenate([splits["ref_train"], target_fit]))
        cal_original = target_cal
    else:
        raise ValueError(f"Unknown annotation scheme: {scheme}")

    train_positions = _positions_for_original_indices(train_original, position_map)
    cal_original = np.asarray(cal_original, dtype=int)

    clf, status = _fit_conformal_classifier(known_latent, known_labels, train_positions)

    unknown = np.asarray(splits["target_unknown"], dtype=int)
    unknown_labels = adata.obs.iloc[unknown][config.cell_type_col].astype(str).to_numpy()
    evaluable_mask = np.asarray([label in reference_label_set for label in unknown_labels], dtype=bool)
    eval_indices = unknown[evaluable_mask]
    skipped_indices = unknown[~evaluable_mask]
    skipped_counts = (
        adata.obs.iloc[skipped_indices][config.cell_type_col].astype(str).value_counts().sort_index().to_dict()
        if skipped_indices.size
        else {}
    )

    def _metadata(method: str, *, status_value: str, n_cal_usable: int, extra: Mapping[str, Any]) -> dict[str, Any]:
        meta = {
            "scheme": scheme,
            "set_method": method,
            "status": status_value,
            "n_train": int(train_positions.size),
            "n_cal": int(cal_original.size),
            "n_cal_usable": int(n_cal_usable),
            "n_eval": int(eval_indices.size),
            "n_skipped_unseen_labels": int(skipped_indices.size),
            "skipped_unseen_label_counts": skipped_counts,
        }
        meta.update(dict(extra))
        return meta

    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}

    if clf is None or eval_indices.size == 0:
        for method in methods:
            predictions = _empty_prediction_frame()
            metrics = summarize_predictions(
                predictions,
                scheme=scheme,
                set_method=method,
                target_coverage=target_coverage,
                n_train=train_positions.size,
                n_cal=cal_original.size,
                n_skipped_unseen_labels=skipped_indices.size,
            )
            results[method] = (predictions, metrics, _metadata(method, status_value=status, n_cal_usable=0, extra={}))
        return results

    eval_scores = _candidate_conditioned_scores(
        adata, eval_indices, config, adata_known, model, integrator, clf, cell_types, candidate_labels,
    )
    eval_pi = _normalize_scores(eval_scores)

    cal_labels = (
        adata.obs.iloc[cal_original][config.cell_type_col].astype(str).to_numpy()
        if cal_original.size
        else np.asarray([], dtype=str)
    )
    cal_usable_mask = np.asarray([label in cand_to_idx for label in cal_labels], dtype=bool)
    cal_indices_usable = cal_original[cal_usable_mask] if cal_original.size else cal_original
    cal_true_cols = (
        np.asarray([cand_to_idx[label] for label in cal_labels[cal_usable_mask]], dtype=int)
        if cal_indices_usable.size
        else np.asarray([], dtype=int)
    )

    if cal_indices_usable.size:
        cal_scores = _candidate_conditioned_scores(
            adata, cal_indices_usable, config, adata_known, model, integrator, clf, cell_types, candidate_labels,
        )
        thr_nonconf = 1.0 - cal_scores[np.arange(cal_indices_usable.size), cal_true_cols]
        aps_nonconf = _aps_calibration_scores(_normalize_scores(cal_scores), cal_true_cols)
        status_eff = status
    else:
        thr_nonconf = np.asarray([], dtype=np.float64)
        aps_nonconf = np.asarray([], dtype=np.float64)
        status_eff = "warning: calibration set is empty; all candidate labels pass"

    q_thr = _marginal_quantile(thr_nonconf, target_coverage=target_coverage)
    q_aps = _marginal_quantile(aps_nonconf, target_coverage=target_coverage)
    q_thr_class = _class_quantiles(
        thr_nonconf, cal_true_cols, n_candidates=n_candidates, target_coverage=target_coverage, fallback=q_thr
    )
    q_aps_class = _class_quantiles(
        aps_nonconf, cal_true_cols, n_candidates=n_candidates, target_coverage=target_coverage, fallback=q_aps
    )

    n_eval = int(eval_indices.size)
    for method in methods:
        if method == "threshold":
            index_sets = [_set_threshold(eval_scores[i], q_thr) for i in range(n_eval)]
            extra = {"q_marginal": q_thr}
        elif method == "mondrian":
            index_sets = [_set_threshold_mondrian(eval_scores[i], q_thr_class) for i in range(n_eval)]
            extra = {"q_per_class": {candidate_labels[c]: float(q_thr_class[c]) for c in range(n_candidates)}}
        elif method == "aps":
            index_sets = [_set_aps(eval_pi[i], q_aps) for i in range(n_eval)]
            extra = {"q_marginal": q_aps}
        elif method == "aps_mondrian":
            index_sets = [_set_aps_mondrian(eval_pi[i], q_aps_class) for i in range(n_eval)]
            extra = {"q_per_class": {candidate_labels[c]: float(q_aps_class[c]) for c in range(n_candidates)}}
        else:
            raise ValueError(f"Unknown conformal set method: {method}")

        predictions = _predictions_from_index_sets(
            adata, eval_indices, config, candidate_labels, eval_scores, index_sets
        )
        metrics = summarize_predictions(
            predictions,
            scheme=scheme,
            set_method=method,
            target_coverage=target_coverage,
            n_train=train_positions.size,
            n_cal=cal_indices_usable.size,
            n_skipped_unseen_labels=skipped_indices.size,
        )
        results[method] = (
            predictions,
            metrics,
            _metadata(method, status_value=status_eff, n_cal_usable=cal_indices_usable.size, extra=extra),
        )

    return results


def _metric_row(
    predictions: pd.DataFrame,
    *,
    scheme: str,
    set_method: str,
    target_coverage: float,
    group: str,
    batch: str,
    cell_type: str,
    n_train: int,
    n_cal: int,
    n_skipped_unseen_labels: int,
) -> dict[str, Any]:
    n_eval = int(len(predictions))
    if n_eval == 0:
        empirical_coverage = np.nan
        coverage_nonempty = np.nan
        avg_set_size = np.nan
        singleton_rate = np.nan
        empty_rate = np.nan
        top1_accuracy = np.nan
    else:
        contains_true = predictions["contains_true"].astype(bool)
        set_size = predictions["set_size"].astype(int)
        nonempty = set_size > 0
        empirical_coverage = float(contains_true.mean())
        coverage_nonempty = float(contains_true[nonempty].mean()) if bool(nonempty.any()) else np.nan
        avg_set_size = float(predictions["set_size"].astype(float).mean())
        singleton_rate = float((set_size == 1).mean())
        empty_rate = float((set_size == 0).mean())
        top1_accuracy = float((predictions["top_label"].astype(str) == predictions["true_label"].astype(str)).mean())

    coverage_gap = float(target_coverage - empirical_coverage) if n_eval > 0 else np.nan
    return {
        "scheme": scheme,
        "set_method": set_method,
        "group": group,
        "batch": batch,
        "cell_type": cell_type,
        "target_coverage": float(target_coverage),
        "alpha": float(1.0 - target_coverage),
        "empirical_coverage": empirical_coverage,
        "coverage_gap": coverage_gap,
        "coverage_nonempty": coverage_nonempty,
        "avg_set_size": avg_set_size,
        "singleton_rate": singleton_rate,
        "empty_rate": empty_rate,
        "top1_accuracy": top1_accuracy,
        "n_train": int(n_train),
        "n_cal": int(n_cal),
        "n_eval": n_eval,
        "n_skipped_unseen_labels": int(n_skipped_unseen_labels),
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    scheme: str,
    set_method: str,
    target_coverage: float,
    n_train: int,
    n_cal: int,
    n_skipped_unseen_labels: int,
) -> pd.DataFrame:
    rows = [
        _metric_row(
            predictions,
            scheme=scheme,
            set_method=set_method,
            target_coverage=target_coverage,
            group="all",
            batch="all",
            cell_type="all",
            n_train=n_train,
            n_cal=n_cal,
            n_skipped_unseen_labels=n_skipped_unseen_labels,
        )
    ]
    if not predictions.empty:
        for batch, group_df in predictions.groupby("batch", sort=True, dropna=False):
            rows.append(
                _metric_row(
                    group_df,
                    scheme=scheme,
                    set_method=set_method,
                    target_coverage=target_coverage,
                    group="batch",
                    batch=str(batch),
                    cell_type="all",
                    n_train=n_train,
                    n_cal=n_cal,
                    n_skipped_unseen_labels=0,
                )
            )
        for label, group_df in predictions.groupby("true_label", sort=True, dropna=False):
            rows.append(
                _metric_row(
                    group_df,
                    scheme=scheme,
                    set_method=set_method,
                    target_coverage=target_coverage,
                    group="cell_type",
                    batch="all",
                    cell_type=str(label),
                    n_train=n_train,
                    n_cal=n_cal,
                    n_skipped_unseen_labels=0,
                )
            )
        for (batch, label), group_df in predictions.groupby(["batch", "true_label"], sort=True, dropna=False):
            rows.append(
                _metric_row(
                    group_df,
                    scheme=scheme,
                    set_method=set_method,
                    target_coverage=target_coverage,
                    group="batch_cell_type",
                    batch=str(batch),
                    cell_type=str(label),
                    n_train=n_train,
                    n_cal=n_cal,
                    n_skipped_unseen_labels=0,
                )
            )
    return pd.DataFrame(rows)


def _run_prefix(run_name: str, name_outputs: bool) -> str:
    return f"{run_name}_" if name_outputs else ""


def known_embedding_path(config: IntegrationConfig, run_name: str, name_outputs: bool) -> Path:
    return config.output_dir / f"{_run_prefix(run_name, name_outputs)}clear_known_embedding.npy"


def prediction_path(config: IntegrationConfig, run_name: str, name_outputs: bool, scheme: str, method: str) -> Path:
    return config.output_dir / f"{_run_prefix(run_name, name_outputs)}annotation_predictions_{scheme}_{method}.csv"


def metrics_path(config: IntegrationConfig, run_name: str, name_outputs: bool, scheme: str, method: str) -> Path:
    return config.output_dir / f"{_run_prefix(run_name, name_outputs)}annotation_metrics_{scheme}_{method}.csv"


def run_completed(config: IntegrationConfig, annotation: AnnotationConfig, run_name: str, name_outputs: bool) -> bool:
    if not known_embedding_path(config, run_name, name_outputs).exists():
        return False
    for scheme in annotation.annotation_schemes:
        for method in annotation.set_methods:
            if not prediction_path(config, run_name, name_outputs, scheme, method).exists():
                return False
            if not metrics_path(config, run_name, name_outputs, scheme, method).exists():
                return False
    return True


def save_skipped_unknowns(
    adata: sc.AnnData,
    config: IntegrationConfig,
    skipped_indices: np.ndarray,
    path: Path,
) -> None:
    rows = []
    for idx in np.asarray(skipped_indices, dtype=int).tolist():
        rows.append(
            {
                "cell_id": str(adata.obs_names[idx]),
                "batch": str(adata.obs.iloc[idx][config.batch_key]),
                "true_label": str(adata.obs.iloc[idx][config.cell_type_col]),
                "reason": "label_absent_from_reference_batch",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_single_annotation_experiment(
    config: IntegrationConfig,
    annotation: AnnotationConfig,
    run_name: str,
    *,
    name_outputs: bool,
    resume: bool,
    manifest: Optional[dict[str, Any]],
    manifest_path: Optional[Path],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if resume and run_completed(config, annotation, run_name, name_outputs):
        logging.info("Skipping completed annotation run %s", run_name)
        metrics_tables = []
        prediction_tables = []
        for scheme in annotation.annotation_schemes:
            for method in annotation.set_methods:
                metrics_tables.append(pd.read_csv(metrics_path(config, run_name, name_outputs, scheme, method)))
                prediction_tables.append(pd.read_csv(prediction_path(config, run_name, name_outputs, scheme, method)))
        return pd.concat(metrics_tables, ignore_index=True), pd.concat(prediction_tables, ignore_index=True)

    set_global_seeds(config.seed)
    run_state = None
    if manifest is not None and manifest_path is not None:
        run_state = manifest.setdefault("runs", {}).setdefault(run_name, {})
        run_state["status"] = "running"
        run_state["started_at"] = run_state.get("started_at") or _utc_now()
        write_json_atomic(manifest_path, manifest)

    base_adata = load_and_prepare_data_cached(config)
    splits = make_annotation_splits(base_adata, config, annotation, seed=config.seed)
    split_summary = _split_summary(base_adata, config, splits)

    reference_labels = sorted(
        base_adata.obs.iloc[splits["ref_all"]][config.cell_type_col].astype(str).unique().tolist()
    )
    unknown_labels = base_adata.obs.iloc[splits["target_unknown"]][config.cell_type_col].astype(str)
    unseen_unknown_mask = ~unknown_labels.isin(reference_labels).to_numpy()
    skipped_unknown = splits["target_unknown"][unseen_unknown_mask]

    if splits["target_unknown"].size == 0:
        logging.warning("No target_unknown cells were created for %s; metrics will be empty.", run_name)
    if splits["known"].size == 0:
        raise ValueError("No known cells available for CLEAR training.")
    if splits["ref_train"].size == 0:
        raise ValueError("No reference training cells available for classifier fitting.")

    adata_known = base_adata[splits["known"]].copy()
    integrator, model, _, known_latent, cell_types = train_clear_known(adata_known, config)
    np.save(known_embedding_path(config, run_name, name_outputs), known_latent, allow_pickle=False)

    skipped_path = config.output_dir / f"{_run_prefix(run_name, name_outputs)}annotation_skipped_unknown.csv"
    save_skipped_unknowns(base_adata, config, skipped_unknown, skipped_path)

    scheme_metrics = []
    scheme_predictions = []
    scheme_metadata = {}
    for scheme in annotation.annotation_schemes:
        method_results = evaluate_scheme(
            scheme=scheme,
            adata=base_adata,
            adata_known=adata_known,
            config=config,
            annotation=annotation,
            splits=splits,
            known_indices=splits["known"],
            known_latent=known_latent,
            model=model,
            integrator=integrator,
            cell_types=cell_types,
            candidate_labels=reference_labels,
        )
        for method in annotation.set_methods:
            predictions, metrics, metadata = method_results[method]
            predictions.insert(0, "set_method", method)
            predictions.insert(0, "scheme", scheme)
            metrics.insert(0, "run_name", run_name)

            pred_path = prediction_path(config, run_name, name_outputs, scheme, method)
            met_path = metrics_path(config, run_name, name_outputs, scheme, method)
            predictions.to_csv(pred_path, index=False, float_format="%.6f")
            metrics.to_csv(met_path, index=False, float_format="%.6f")
            logging.info("Saved %s/%s predictions to %s", scheme, method, pred_path)
            logging.info("Saved %s/%s metrics to %s", scheme, method, met_path)

            scheme_predictions.append(predictions)
            scheme_metrics.append(metrics)
            scheme_metadata[f"{scheme}/{method}"] = metadata

    if manifest is not None and manifest_path is not None and run_state is not None:
        run_state["status"] = "done"
        run_state["finished_at"] = _utc_now()
        run_state["split_summary"] = split_summary
        run_state["annotation"] = asdict(annotation)
        run_state["candidate_reference_labels"] = reference_labels
        run_state["skipped_unknown_path"] = str(skipped_path)
        run_state["schemes"] = scheme_metadata
        write_json_atomic(manifest_path, manifest)

    all_metrics = pd.concat(scheme_metrics, ignore_index=True) if scheme_metrics else None
    all_predictions = pd.concat(scheme_predictions, ignore_index=True) if scheme_predictions else None
    return all_metrics, all_predictions


def add_run_metadata(
    df: pd.DataFrame,
    *,
    experiment_name: str,
    variant_name: str,
    run_name: str,
    repetition: int,
    seed: int,
    hyperparams: Mapping[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "experiment", experiment_name)
    out.insert(1, "variant", variant_name)
    out.insert(2, "repeat", int(repetition))
    out.insert(3, "seed", int(seed))
    if "run_name" not in out.columns:
        out.insert(4, "run_name", run_name)
    for key, value in hyperparams.items():
        out[key] = value
    return out


def save_repetition_outputs(
    config: IntegrationConfig,
    experiment_name: str,
    metrics_tables: list[pd.DataFrame],
    prediction_tables: list[pd.DataFrame],
) -> None:
    if metrics_tables:
        metrics_all = pd.concat(metrics_tables, ignore_index=True)
        metrics_all_path = config.output_dir / f"annotation_metrics_all_repetitions_{experiment_name}.csv"
        metrics_all.to_csv(metrics_all_path, index=False, float_format="%.6f")

        summary_cols = [
            "empirical_coverage",
            "coverage_gap",
            "coverage_nonempty",
            "avg_set_size",
            "singleton_rate",
            "empty_rate",
            "top1_accuracy",
        ]
        available = [col for col in summary_cols if col in metrics_all.columns]
        if available:
            group_cols = ["experiment", "variant", "scheme", "set_method", "group", "batch", "cell_type"]
            group_cols = [col for col in group_cols if col in metrics_all.columns]
            summary = metrics_all.groupby(group_cols, dropna=False)[available].agg(["mean", "std"])
            summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
            summary_path = config.output_dir / f"annotation_metrics_summary_{experiment_name}.csv"
            summary.to_csv(summary_path, float_format="%.6f")
            logging.info("Saved annotation metrics summary to %s", summary_path)
        logging.info("Saved annotation metrics details to %s", metrics_all_path)

    if prediction_tables:
        predictions_all = pd.concat(prediction_tables, ignore_index=True)
        predictions_path = config.output_dir / f"annotation_predictions_all_repetitions_{experiment_name}.csv"
        predictions_all.to_csv(predictions_path, index=False, float_format="%.6f")
        logging.info("Saved annotation predictions to %s", predictions_path)


def initialize_manifest(
    config: IntegrationConfig,
    annotation: AnnotationConfig,
    args: argparse.Namespace,
    experiment_name: str,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    manifest_path = config.output_dir / "annotation_run_manifest.json"
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
            "experiment_name": experiment_name,
            "created_at": manifest.get("created_at") or _utc_now(),
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "config": _serialize_config(config),
            "annotation_config": asdict(annotation),
            "runtime": runtime,
            "resume": bool(args.resume),
            "overwrite": bool(args.overwrite),
        }
    )
    manifest.setdefault("runs", {})
    write_json_atomic(manifest_path, manifest)
    return manifest, manifest_path


def print_dry_run_plan(
    exp_name: str,
    config: IntegrationConfig,
    annotation: AnnotationConfig,
    grid_configs: list[dict[str, Any]],
    n_repetitions: int,
) -> None:
    print(f"\nAnnotation dry run for {exp_name}")
    print(f"  data_path: {config.data_path}")
    print(f"  output_dir: {config.output_dir}")
    print("  method: conftr")
    print(f"  schemes: {', '.join(annotation.annotation_schemes)}")
    print(f"  set_methods: {', '.join(annotation.set_methods)}")
    print(f"  target_holdout_fraction: {annotation.target_holdout_fraction}")
    print(f"  target_coverage: {annotation.target_coverage}")
    print(f"  repetitions: {n_repetitions}")
    print(f"  grid variants: {len(grid_configs)}")
    for experiment_params in grid_configs:
        variant_name = format_grid_name(exp_name, experiment_params)
        for rep in range(n_repetitions):
            seed = config.seed + rep
            run_name = f"{variant_name}_rep{rep}_seed{seed}" if n_repetitions > 1 else variant_name
            print(f"  - {run_name}: seed={seed}, params={experiment_params or '{}'}")


def run_synthetic_smoke_test() -> None:
    """Fast local smoke test for annotation split and conformal helper behavior."""

    global np
    global pd
    global LogisticRegression

    import numpy as _np
    import pandas as _pd
    from sklearn.linear_model import LogisticRegression as _LogisticRegression

    np = _np
    pd = _pd
    LogisticRegression = _LogisticRegression

    rows = []
    for batch in ("ref", "target_a", "target_b"):
        for label in ("alpha", "beta"):
            for _ in range(8):
                rows.append({"batch": batch, "cell_type": label})
    for _ in range(4):
        rows.append({"batch": "target_a", "cell_type": "unseen"})

    obs = pd.DataFrame(rows)
    adata = SimpleNamespace(obs=obs, obs_names=pd.Index([f"cell_{idx}" for idx in range(len(obs))]))
    config = SimpleNamespace(
        batch_key="batch",
        cell_type_col="cell_type",
        ref_batch="ref",
        reference_dictionary={"ref": ["target_a", "target_b"]},
    )
    annotation = AnnotationConfig(
        target_holdout_fraction=0.5,
        reference_calibration_fraction=0.25,
        semi_target_calibration_fraction=0.25,
        target_coverage=0.90,
        annotation_schemes=("simple", "semi_supervised"),
    )

    splits = make_annotation_splits(adata, config, annotation, seed=7)
    assert set(splits["target_unknown"]).isdisjoint(set(splits["known"]))
    assert set(splits["target_cal"]).issubset(set(splits["target_known"]))
    assert set(splits["target_fit"]).issubset(set(splits["target_known"]))

    reference_labels = sorted(obs.iloc[splits["ref_all"]]["cell_type"].astype(str).unique().tolist())
    assert reference_labels == ["alpha", "beta"]

    known_indices = splits["known"]
    position_map = _original_to_known_positions(known_indices)
    known_labels = obs.iloc[known_indices]["cell_type"].astype(str).to_numpy()
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(known_indices.size, 4)).astype(np.float32)
    latent[known_labels == "alpha", 0] += 2.0
    latent[known_labels == "beta", 0] -= 2.0

    train_positions = _positions_for_original_indices(splits["ref_train"], position_map)
    cal_positions = _positions_for_original_indices(splits["ref_cal"], position_map)
    clf, status = _fit_conformal_classifier(latent, known_labels, train_positions)
    assert clf is not None
    assert status == "ok"

    # --- conformal primitives on synthetic candidate scores ---
    candidate_labels = ["alpha", "beta"]
    cand_to_idx = {label: idx for idx, label in enumerate(candidate_labels)}
    class_to_col = {str(label): pos for pos, label in enumerate(clf.classes_)}
    col_order = [class_to_col[label] for label in candidate_labels]
    cal_labels = known_labels[cal_positions]
    S_cal = clf.predict_proba(latent[cal_positions])[:, col_order]
    cal_true_cols = np.asarray([cand_to_idx[str(label)] for label in cal_labels], dtype=int)

    thr_nonconf = 1.0 - S_cal[np.arange(S_cal.shape[0]), cal_true_cols]
    q_thr = _marginal_quantile(thr_nonconf, target_coverage=annotation.target_coverage)
    q_thr_class = _class_quantiles(
        thr_nonconf, cal_true_cols, n_candidates=2, target_coverage=annotation.target_coverage, fallback=q_thr
    )
    assert np.isfinite(q_thr)
    assert q_thr_class.shape == (2,)

    aps_nonconf = _aps_calibration_scores(_normalize_scores(S_cal), cal_true_cols)
    q_aps = _marginal_quantile(aps_nonconf, target_coverage=annotation.target_coverage)
    q_aps_class = _class_quantiles(
        aps_nonconf, cal_true_cols, n_candidates=2, target_coverage=annotation.target_coverage, fallback=q_aps
    )
    assert np.isfinite(q_aps)

    # APS variants must never be empty; aps_mondrian must always keep the arg-max.
    demo_rows = _normalize_scores(np.array([[0.9, 0.1], [0.4, 0.6], [0.5, 0.5]], dtype=np.float64))
    for row in demo_rows:
        assert len(_set_aps(row, q_aps)) >= 1
        mondrian_set = _set_aps_mondrian(row, q_aps_class)
        assert len(mondrian_set) >= 1
        assert int(np.argmax(row)) in mondrian_set
    # The plain threshold rule may legitimately yield an empty set under a strict cutoff.
    assert _set_threshold(np.array([0.0, 0.0]), -1.0) == []

    predictions = pd.DataFrame(
        [
            {
                "cell_id": "cell_x",
                "batch": "target_a",
                "true_label": "alpha",
                "prediction_set": "alpha|beta",
                "set_size": 2,
                "contains_true": True,
                "top_label": "alpha",
                "top_probability": 0.8,
            },
            {
                "cell_id": "cell_y",
                "batch": "target_b",
                "true_label": "beta",
                "prediction_set": "alpha",
                "set_size": 1,
                "contains_true": False,
                "top_label": "alpha",
                "top_probability": 0.7,
            },
        ]
    )
    metrics = summarize_predictions(
        predictions,
        scheme="simple",
        set_method="threshold",
        target_coverage=annotation.target_coverage,
        n_train=train_positions.size,
        n_cal=cal_positions.size,
        n_skipped_unseen_labels=1,
    )
    all_row = metrics.loc[metrics["group"] == "all"].iloc[0]
    assert int(all_row["n_eval"]) == 2
    assert float(all_row["empirical_coverage"]) == 0.5
    assert "set_method" in metrics.columns
    assert "coverage_nonempty" in metrics.columns

    print("Synthetic annotation smoke test passed.")


def main() -> None:
    args = parse_args()
    setup_logging()
    if args.smoke_test:
        run_synthetic_smoke_test()
        return
    if not args.dry_run:
        load_runtime_dependencies()
        try:
            import torch.multiprocessing as mp

            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    runtime = collect_runtime_environment()
    log_runtime_environment(runtime)

    config_file = args.config_file.resolve()
    config_dir = config_file.parent
    with config_file.open("r", encoding="utf-8") as handle:
        config_dict = json.load(handle)

    if args.experiment_name == "all":
        experiment_names = [key for key in config_dict.keys() if key != "default_params"]
    else:
        experiment_names = [args.experiment_name]

    for exp_name in experiment_names:
        logging.info("Preparing annotation experiment %s", exp_name)
        config, annotation = json_args_annotation(
            config_dict,
            exp_name,
            config_dir=config_dir,
            require_runtime=not args.dry_run,
        )
        apply_single_value_hyperparams(config)
        apply_cli_overrides(config, annotation, args)
        normalize_to_clear_only(config)
        validate_config(config)
        validate_annotation_config(annotation)

        if args.overwrite:
            if config.output_dir.exists():
                logging.warning("Overwriting output directory: %s", config.output_dir)
                shutil.rmtree(config.output_dir)
            config.output_dir.mkdir(parents=True, exist_ok=True)
        elif config.output_dir.exists() and not args.resume:
            logging.info(
                "Output directory exists and will be reused without deleting: %s. "
                "Use --overwrite to delete it or --resume to skip completed annotation runs.",
                config.output_dir,
            )

        base_seed = config.seed
        n_repetitions = max(1, int(getattr(config, "n_repetitions", 1)))
        grid_configs = build_hyperparam_grid(config)
        has_grid = len(grid_configs) > 1
        name_outputs = has_grid or n_repetitions > 1
        metrics_tables: list[pd.DataFrame] = []
        prediction_tables: list[pd.DataFrame] = []

        if args.dry_run:
            print_dry_run_plan(exp_name, config, annotation, grid_configs, n_repetitions)
            continue

        try:
            shutil.copy2(config_file, config.output_dir / "annotation_source_config.json")
        except Exception as exc:
            logging.warning("Could not copy source config into output directory: %s", exc)

        manifest, manifest_path = initialize_manifest(config, annotation, args, exp_name, runtime)

        for experiment_params in grid_configs:
            for key, value in experiment_params.items():
                setattr(config, key, value)
            variant_name = format_grid_name(exp_name, experiment_params)

            for rep in range(n_repetitions):
                config.seed = base_seed + rep
                run_name = (
                    f"{variant_name}_rep{rep}_seed{config.seed}"
                    if n_repetitions > 1
                    else variant_name
                )
                logging.info("Running annotation experiment %s | seed=%s", run_name, config.seed)
                try:
                    metrics_df, predictions_df = run_single_annotation_experiment(
                        config,
                        annotation,
                        run_name,
                        name_outputs=name_outputs,
                        resume=args.resume,
                        manifest=manifest,
                        manifest_path=manifest_path,
                    )
                except Exception as exc:
                    logging.error("Annotation run %s failed: %s", run_name, exc)
                    logging.error(traceback.format_exc())
                    if manifest is not None and manifest_path is not None:
                        run_state = manifest.setdefault("runs", {}).setdefault(run_name, {})
                        run_state["status"] = "failed"
                        run_state["error"] = str(exc)
                        run_state["traceback"] = traceback.format_exc()
                        write_json_atomic(manifest_path, manifest)
                    continue

                if metrics_df is not None:
                    metrics_tables.append(
                        add_run_metadata(
                            metrics_df,
                            experiment_name=exp_name,
                            variant_name=variant_name,
                            run_name=run_name,
                            repetition=rep,
                            seed=config.seed,
                            hyperparams=experiment_params,
                        )
                    )
                if predictions_df is not None:
                    prediction_tables.append(
                        add_run_metadata(
                            predictions_df,
                            experiment_name=exp_name,
                            variant_name=variant_name,
                            run_name=run_name,
                            repetition=rep,
                            seed=config.seed,
                            hyperparams=experiment_params,
                        )
                    )

        save_repetition_outputs(config, exp_name, metrics_tables, prediction_tables)
        logging.info("Annotation experiment %s completed.", exp_name)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
