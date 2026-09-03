#!/usr/bin/env python
"""Shared helpers for CLEAR manuscript plotting scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CLEAR_DIR = SCRIPT_DIR.parents[1]
CONFIG_DIR = CLEAR_DIR / "paper"
RESULTS_DIR = CLEAR_DIR / "results"
OUTDIR = CLEAR_DIR / "plots" / "results"
VALID_COVGAP_THRESHOLD = 0.01

METHOD_ORDER = ["combat", "harmony", "scanorama", "scanvi", "scvi", "conftr"]
METHOD_LABELS = {
    "combat": "Combat",
    "harmony": "Harmony",
    "scanorama": "Scanorama",
    "scanvi": "scANVI",
    "scvi": "scVI",
    "conftr": "CLEAR",
}
METHOD_COLORS_COVGAP = {
    "combat": "#8A8F98",
    "harmony": "#4E79A7",
    "scanorama": "#59A14F",
    "scanvi": "#B07AA1",
    "scvi": "#76B7B2",
    "conftr": "#D55E00",
}
METHOD_COLORS_RANK = {
    "Combat": "#377EB8",
    "Harmony": "#FF7F00",
    "Scanorama": "#4DAF4A",
    "scANVI": "#984EA3",
    "scVI": "#8C564B",
    "CLEAR": "#E41A1C",
}

TASK_SPECS = [
    ("MB-assay", "mb-config.json", "experimento-batch-coverage"),
    ("ATAC-assay", "atac-config.json", "experimento-batch-coverage"),
    ("Kidney-donors", "kidsney-config.json", "experimento-batch-coverage"),
    ("Kidney-assay", "kidsney-config.json", "experimento-assay-coverage"),
    ("gexV9-donors", "gexV9-config.json", "experimento-batch-coverage"),
    ("gexV9-stage", "gexV9-config.json", "experimento-development-coverage"),
    ("gexV9-tissue", "gexV9-config.json", "experimento-tissue-coverage"),
    ("Immune-donors", "immune-config.json", "experimento-batch-coverage"),
    ("Immune-assay", "immune-config.json", "experimento-assay-coverage"),
    ("Immune-stage", "immune-config.json", "experimento-development-coverage"),
    ("Hypo-assay", "hypo-config.json", "experimento-assay-coverage"),
]
TASK_ORDER = [task for task, _, _ in TASK_SPECS]

METRIC_RENAMES = {
    "ASW_label": "ASW-L",
    "ASW_batch": "ASW-B",
    "graph_connectivity": "GC",
    "isolated_labels": "IsLa",
    "logreg_covgap": "CovGap",
    "logreg_train_target_covgap": "TrainTargetCovGap",
    "logreg_avg_size": "AvgSetSize",
    "nmi": "NMI",
    "ari": "ARI",
}


@dataclass(frozen=True)
class LoadedExperiment:
    task: str
    config_name: str
    experiment_name: str
    output_dir: Path
    metrics_path: Path
    metric_source: str
    batch_key: str


def configure_matplotlib(kind: str) -> None:
    if kind == "covgap":
        settings = {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#222222",
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 9.0,
        }
    elif kind == "rank":
        settings = {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#222222",
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
        }
    else:
        raise ValueError(f"Unknown Matplotlib style: {kind}")

    settings.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    mpl.rcParams.update(settings)


def save_all(fig: mpl.figure.Figure, basename: str, *, tight: bool = False) -> list[Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    for ext in ("png", "pdf", "svg"):
        path = OUTDIR / f"{basename}.{ext}"
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def discover_expected_experiments() -> tuple[list[LoadedExperiment], list[str]]:
    loaded: list[LoadedExperiment] = []
    warnings: list[str] = []

    for task, config_name, experiment_name in TASK_SPECS:
        config_path = CONFIG_DIR / config_name
        if not config_path.exists():
            warnings.append(f"{task}: missing config {config_path.relative_to(CLEAR_DIR)}")
            continue

        with config_path.open() as handle:
            config = json.load(handle)

        if experiment_name not in config:
            warnings.append(f"{task}: {config_name} has no experiment '{experiment_name}'")
            continue

        spec = config[experiment_name]
        output_dir = (CONFIG_DIR / spec["output_dir"]).resolve()
        summary_path = output_dir / f"metrics_summary_{experiment_name}.csv"
        integration_path = output_dir / "integration_metrics.csv"

        if summary_path.exists():
            metrics_path = summary_path
            metric_source = "metrics_summary"
        elif integration_path.exists():
            metrics_path = integration_path
            metric_source = "integration_metrics"
        else:
            rel_output = output_dir.relative_to(CLEAR_DIR) if output_dir.is_relative_to(CLEAR_DIR) else output_dir
            warnings.append(f"{task}: no metric file yet under {rel_output}")
            continue

        loaded.append(
            LoadedExperiment(
                task=task,
                config_name=config_name,
                experiment_name=experiment_name,
                output_dir=output_dir,
                metrics_path=metrics_path,
                metric_source=metric_source,
                batch_key=spec.get("batch_key", "batch"),
            )
        )

    return loaded, warnings


def _read_metrics(path: Path, source: str) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    if "method" not in metrics.columns:
        raise ValueError(f"{path} is missing required column: method")

    if source == "metrics_summary":
        selected_columns = ["method"]
        if "experiment" in metrics.columns:
            selected_columns.append("experiment")
        if "variant" in metrics.columns:
            selected_columns.append("variant")
        selected_columns.extend(column for column in metrics.columns if column.endswith("_mean"))
        metrics = metrics[selected_columns].copy()
        metrics = metrics.rename(columns={column: column.removesuffix("_mean") for column in metrics.columns})

    return metrics.rename(columns=METRIC_RENAMES)


def load_metric_table(require_covgap: bool = False) -> tuple[pd.DataFrame, list[LoadedExperiment], list[str]]:
    loaded, warnings = discover_expected_experiments()
    frames: list[pd.DataFrame] = []

    for item in loaded:
        metrics = _read_metrics(item.metrics_path, item.metric_source)
        if require_covgap and "CovGap" not in metrics.columns:
            raise ValueError(f"{item.metrics_path} is missing required metric: CovGap")

        metrics["MethodKey"] = metrics["method"].astype(str).str.lower()
        metrics = metrics[metrics["MethodKey"].isin(METHOD_ORDER)].copy()
        metrics["Method"] = metrics["MethodKey"].map(METHOD_LABELS)
        metrics["Task"] = item.task
        metrics["TaskPath"] = item.output_dir.relative_to(CLEAR_DIR).as_posix()
        metrics["Experiment"] = item.experiment_name
        metrics["Config"] = item.config_name
        metrics["BatchKey"] = item.batch_key
        metrics["MetricSource"] = item.metric_source

        numeric_columns = metrics.select_dtypes(include=[np.number]).columns.tolist()
        metadata = [
            "Task",
            "TaskPath",
            "Experiment",
            "Config",
            "BatchKey",
            "MetricSource",
            "MethodKey",
            "Method",
        ]
        frames.append(
            metrics[metadata + numeric_columns]
            .groupby(metadata, as_index=False, observed=True)
            .mean(numeric_only=True)
        )

    if not frames:
        raise FileNotFoundError("No CLEAR metric files are available yet for the configured paper tasks.")

    df = pd.concat(frames, ignore_index=True)
    task_order = [task for task in TASK_ORDER if task in set(df["Task"])]
    method_labels = [METHOD_LABELS[key] for key in METHOD_ORDER if key in set(df["MethodKey"])]
    df["Task"] = pd.Categorical(df["Task"], categories=task_order, ordered=True)
    df["Method"] = pd.Categorical(df["Method"], categories=method_labels, ordered=True)

    if "CovGap" in df.columns:
        df["CovGap"] = pd.to_numeric(df["CovGap"], errors="coerce")
        df["Valid_CovGap"] = df["CovGap"] <= VALID_COVGAP_THRESHOLD
        df["Coverage_shortfall"] = df["CovGap"].clip(lower=0)

    return df.sort_values(["Task", "Method"]).reset_index(drop=True), loaded, warnings


def available_metrics(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    metrics = []
    for metric in candidates:
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any():
            metrics.append(metric)
    return metrics


def method_keys(df: pd.DataFrame) -> list[str]:
    present = set(df["MethodKey"])
    return [method for method in METHOD_ORDER if method in present]


def method_labels(df: pd.DataFrame) -> list[str]:
    return [METHOD_LABELS[method] for method in method_keys(df)]


def write_loaded_experiments(loaded: list[LoadedExperiment], path: Path) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "task": item.task,
                "output_dir": item.output_dir.relative_to(CLEAR_DIR).as_posix(),
                "metrics_file": item.metrics_path.relative_to(CLEAR_DIR).as_posix(),
                "metric_source": item.metric_source,
                "experiment": item.experiment_name,
                "config": item.config_name,
                "batch_key": item.batch_key,
            }
            for item in loaded
        ]
    ).to_csv(path, index=False)


def print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print("Warnings: skipped pending or unavailable CLEAR experiments:")
    for warning in warnings:
        print(f"- {warning}")
