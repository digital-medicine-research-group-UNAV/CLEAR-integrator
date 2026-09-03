import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchsort
except Exception:
    torchsort = None

from torch.utils.data import DataLoader, TensorDataset, Subset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import sparse

import umap
from sklearn.linear_model import LogisticRegression 


try:
    import seaborn as sns
    sns.set_context("talk")
except ModuleNotFoundError:
    sns = None

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

plt.style.use("seaborn-v0_8")

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

vae_latent_dim = 16
vae_hidden_layers = (564, 164)
umap_basis_model = None
batch_key = None
cell_type_col = None
obs_df = None
seed = 0

TRUE_LABEL_PVALUE_ALPHAS = (0.01, 0.05, 0.10, 0.20)


def _normalize_verbose(verbose: int | bool) -> int:
    if isinstance(verbose, bool):
        return 2 if verbose else 0
    return max(0, min(2, int(verbose)))


def _vprint(verbose: int | bool, level: int, *args, **kwargs) -> None:
    if _normalize_verbose(verbose) >= level:
        print(*args, **kwargs)


def save_latent_plot(latent_np: np.ndarray, title_suffix: str, filename: str) -> None:
    """Project latent codes to 2D and persist scatterplots grouped by batch and cell type."""
    if latent_np.shape[1] > 2:
        coords = PCA(n_components=2).fit_transform(latent_np)
    else:
        coords = latent_np

    latent_df = pd.DataFrame(
        {
            "Z1": coords[:, 0],
            "Z2": coords[:, 1],
            batch_key: obs_df[batch_key].values,
            cell_type_col: obs_df[cell_type_col].values,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, hue_col, title in zip(
        axes,
        [batch_key, cell_type_col],
        [f"Latent space by batch ({title_suffix})", f"Latent space by cell type ({title_suffix})"],
    ):
        if sns:
            sns.scatterplot(
                data=latent_df,
                x="Z1",
                y="Z2",
                hue=hue_col,
                ax=ax,
                s=5,
                alpha=0.7,
            )
            ax.legend(loc="best", fontsize=8)
        else:
            for label in latent_df[hue_col].unique():
                mask = latent_df[hue_col] == label
                ax.scatter(
                    latent_df.loc[mask, "Z1"],
                    latent_df.loc[mask, "Z2"],
                    s=5,
                    alpha=0.7,
                    label=label,
                )
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close(fig)



def reset_umap_basis():
    """Call this if you change latent dimensionality or want to refit the UMAP basis."""
    global umap_basis_model
    umap_basis_model = None

def save_latent_umap(latent_np: np.ndarray, title_suffix: str, filename: str) -> None:
    """
    UMAP of the VAE latent representation with a *fixed basis* across calls.
    First call: fit UMAP on `latent_np`. Subsequent calls: transform into the same UMAP space.
    Accepts the same args as save_latent_plot(latent_np, title_suffix, filename).
    """
    global umap_basis_model

    need_fit = (
        umap_basis_model is None
        or getattr(umap_basis_model, "n_features_in_", None) != latent_np.shape[1]
    )

    if need_fit:
        umap_basis_model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.3,
            metric="euclidean",
            random_state=seed,
            n_jobs=1
        )
        coords = umap_basis_model.fit_transform(latent_np)
    else:
        coords = umap_basis_model.transform(latent_np)

    umap_df = pd.DataFrame(
        {
            "U1": coords[:, 0],
            "U2": coords[:, 1],
            batch_key: obs_df[batch_key].values,
            cell_type_col: obs_df[cell_type_col].values,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, hue_col, title in zip(
        axes,
        [batch_key, cell_type_col],
        [f"UMAP (latent) by batch ({title_suffix})", f"UMAP (latent) by cell type ({title_suffix})"],
    ):
        if 'sns' in globals() and sns is not None:
            sns.scatterplot(data=umap_df, x="U1", y="U2", hue=hue_col, ax=ax, s=5, alpha=0.7)
            ax.legend(loc="best", fontsize=8)
        else:
            for label in umap_df[hue_col].unique():
                m = umap_df[hue_col] == label
                ax.scatter(umap_df.loc[m, "U1"], umap_df.loc[m, "U2"], s=5, alpha=0.7, label=label)
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_title(title)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close(fig)


def plot_training_curves(model, plots_dir=plots_dir, highlight_epoch: int = 21, verbose: int | bool = 2):
    """
    Genera:
      - training_curves_losses.png: SOLO total train/val, eje Y log, vline en highlight_epoch
      - training_curves_coverage.png: cobertura (global y por clase) + y=0.9 + vline en highlight_epoch
      - training_history.csv: histórico completo
    """
    if not hasattr(model, "history") or len(model.history) == 0:
        _vprint(verbose, 1, "No hay histórico en model.history; ¿ejecutaste el entrenamiento?")
        return

    hist = pd.DataFrame(model.history).sort_values("epoch").reset_index(drop=True)
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(plots_dir, "training_history.csv")
    hist.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    eps = 1e-8
    y_train = np.maximum(hist["train_total_loss"].to_numpy(dtype=float), eps)
    ax.plot(hist["epoch"], y_train, label="Train total loss", linewidth=2)

    if "val_total_loss" in hist.columns:
        y_val = hist["val_total_loss"].to_numpy(dtype=float)
        if np.isfinite(y_val).any():
            y_val = np.maximum(np.where(np.isfinite(y_val), y_val, np.nan), eps)
            ax.plot(hist["epoch"], y_val, linestyle="--", label="Val total loss", linewidth=2)

    ax.axvline(highlight_epoch, color="red", linestyle=":", linewidth=1.8, label=f"Nueva loss (época {highlight_epoch})")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Train vs. Validation (Total Loss: ELBO + ConfTr)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_curves_losses.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="Objetivo 0.90")

    if "coverage_overall" in hist.columns:
        ax.plot(hist["epoch"], hist["coverage_overall"], marker="o", linewidth=2, label="Coverage overall")

    cov_cols = [c for c in hist.columns if c.startswith("coverage_") and c != "coverage_overall"]
    for c in sorted(cov_cols):
        ax.plot(hist["epoch"], hist[c], alpha=0.6, label=c.replace("coverage_", "cov_"))

    ax.axvline(highlight_epoch, color="red", linestyle=":", linewidth=1.8, label=f"Nueva loss (época {highlight_epoch})")

    ax.set_xlabel("Epochs"); ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1)
    ax.set_title("Coverage over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_curves_coverage.png"), dpi=150)
    plt.close(fig)

    _vprint(verbose, 1, f"Histórico guardado en {csv_path}")



def plot_conftr_diagnostics(model, plots_dir=plots_dir):
    """
    Genera:
      - conftr_inclusion_curves.png: true_incl y other_incl para train/val.
      - conftr_diagnostics_subplots.png: resto de métricas ConfTr en subgráficas.
    """
    import math
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if not hasattr(model, "history") or len(model.history) == 0:
        print("No hay histórico en model.history; no se pueden pintar diagnósticos ConfTr.")
        return

    hist = pd.DataFrame(model.history).sort_values("epoch").reset_index(drop=True)
    if "epoch" not in hist.columns:
        print("El histórico no contiene la columna 'epoch'; no se pueden pintar diagnósticos ConfTr.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    inclusion_pairs = [
        ("train_conftr_mean_include_true", "Train true_incl", "-"),
        ("val_conftr_mean_include_true", "Val true_incl", "--"),
        ("train_conftr_mean_include_others", "Train other_incl", "-"),
        ("val_conftr_mean_include_others", "Val other_incl", "--"),
    ]
    has_inclusion = any(
        c in hist.columns and np.isfinite(hist[c].to_numpy(dtype=float)).any()
        for c, _, _ in inclusion_pairs
    )
    if has_inclusion:
        fig, ax = plt.subplots(figsize=(9, 5))
        for col, label, linestyle in inclusion_pairs:
            if col not in hist.columns:
                continue
            y = hist[col].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            ax.plot(hist["epoch"], y, linewidth=2, linestyle=linestyle, marker="o", markersize=3, label=label)

        ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Soft inclusion")
        ax.set_title("ConfTr inclusion diagnostics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "conftr_inclusion_curves.png"), dpi=150)
        plt.close(fig)
    else:
        print("No hay columnas de true_incl/other_incl en model.history.")

    extra_pairs = [
        ("train_coverage_gap", "Train coverage_gap", "-"),
        ("val_coverage_gap", "Val coverage_gap", "--"),
        ("train_avg_set_size", "Train avg_set_size", "-"),
        ("val_avg_set_size", "Val avg_set_size", "--"),
        ("train_mean_include_true", "Train mean_include_true", "-"),
        ("val_mean_include_true", "Val mean_include_true", "--"),
        ("train_mean_include_others", "Train mean_include_others", "-"),
        ("val_mean_include_others", "Val mean_include_others", "--"),
    ]

    excluded = {
        "scaled",
        "n_alpha_tasks",
        "B_cal",
        "B_pred",
        "true_p_n",
    }
    metric_names = []
    for col in hist.columns:
        if col in {pair[0] for pair in extra_pairs}:
            name = col.replace("train_", "", 1).replace("val_", "", 1)
        elif col.startswith("train_conftr_"):
            name = col.replace("train_conftr_", "", 1)
        elif col.startswith("val_conftr_"):
            name = col.replace("val_conftr_", "", 1)
        else:
            continue
        if name not in excluded and name not in metric_names:
            metric_names.append(name)

    valid_metric_names = []
    for name in metric_names:
        if name in {"coverage_gap", "avg_set_size", "mean_include_true", "mean_include_others"}:
            cols = (f"train_{name}", f"val_{name}")
        else:
            cols = (f"train_conftr_{name}", f"val_conftr_{name}")
        if any(c in hist.columns and np.isfinite(hist[c].to_numpy(dtype=float)).any() for c in cols):
            valid_metric_names.append(name)

    if not valid_metric_names:
        print("No hay métricas ConfTr adicionales en model.history.")
        return

    ncols = 3
    nrows = int(math.ceil(len(valid_metric_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, name in zip(axes_flat, valid_metric_names):
        if name in {"coverage_gap", "avg_set_size", "mean_include_true", "mean_include_others"}:
            train_col = f"train_{name}"
            val_col = f"val_{name}"
            title = name
        else:
            train_col = f"train_conftr_{name}"
            val_col = f"val_conftr_{name}"
            title = name

        plotted = False
        plot_specs = [(train_col, "Train", "-", None), (val_col, "Val", "--", None)]
        if name == "coverage_gap":
            plot_specs.append(("external_coverage_gap", "External val", ":", "red"))

        for col, label, linestyle, color in plot_specs:
            if col not in hist.columns:
                continue
            y = hist[col].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            ax.plot(hist["epoch"], y, linewidth=1.8, linestyle=linestyle, marker="o", markersize=2.5, label=label, color=color)
            plotted = True

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)

    for ax in axes_flat[len(valid_metric_names):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "conftr_diagnostics_subplots.png"), dpi=150)
    plt.close(fig)
    print(f"Diagnósticos ConfTr guardados en {plots_dir}")



def _softsort_1d(values: torch.Tensor, tau: float, ascending: bool = True):
    if 'torchsort' in globals() and torchsort is not None:
        if hasattr(torchsort, "soft_sort"):
            return torchsort.soft_sort(values, regularization_strength=tau, descending=not ascending)
        if hasattr(torchsort, "softsort"):
            return torchsort.softsort(values, tau=tau, direction=("ASCENDING" if ascending else "DESCENDING"))
    return None

def _smooth_quantile_1d(values: torch.Tensor, q: float, tau: float = 1e-2) -> torch.Tensor:
    """
    Differentiable-ish quantile for 1D tensors.
    Prefers SoftSort; falls back to straight-through estimator (STE).
    """
    n = values.shape[0]
    q = max(0.0, min(1.0, float(q)))
    if n <= 1:
        return values.mean()

    s = _softsort_1d(values, tau=tau, ascending=True)
    if s is not None:
        if s.ndim > 1: s = s.reshape(-1)
        idx = q * (n - 1)
        low = int(torch.floor(torch.tensor(idx, device=values.device)).item())
        high = min(low + 1, n - 1)
        w = idx - low
        return torch.lerp(s[low], s[high], torch.tensor(w, device=values.device, dtype=s.dtype))
    else:
        try:
            hard = torch.quantile(values.detach(), q)
        except AttributeError:
            k = max(1, int(round(q * n)))
            hard = torch.kthvalue(values.detach(), k=k).values
        return values.mean() + (hard - values.mean()).detach()


def _smooth_calibrate_thr(scores: torch.Tensor,
                          y: torch.Tensor,
                          alpha: float,
                          eps_sort: float = 1e-2) -> torch.Tensor:
    """
    THR calibration on the calibration subset: tau = smooth quantile of E(x_i, y_i).
    `scores` are class *scores used by THR* (probabilities or logits/log-probs).
    """
    e_true = scores.gather(1, y.view(-1, 1)).squeeze(1)
    n = max(1, e_true.numel())
    q = alpha * (1.0 + 1.0 / n)
    tau = _smooth_quantile_1d(e_true, q=q, tau=eps_sort)
    return tau


def _smooth_calibrate_thr_mondrian(scores: torch.Tensor,
                                   y: torch.Tensor,
                                   K: int,
                                   alpha: float,
                                   eps_sort: float = 1e-2) -> torch.Tensor:
    """
    Per-class (Mondrian) smooth thresholds tau_k.
    `scores` are class scores (log-probs recommended), shape (B_cal, K).
    Returns tau_vec with shape (K,).
    """
    tau_global = _smooth_calibrate_thr(scores, y, alpha=alpha, eps_sort=eps_sort)

    taus = []
    for k in range(K):
        mask_k = (y == k)
        if mask_k.any():
            e_k = scores[mask_k, k].reshape(-1)
            n_k = max(1, e_k.numel())
            q_k = alpha * (1.0 + 1.0 / n_k)
            tau_k = _smooth_quantile_1d(e_k, q=q_k, tau=eps_sort)
        else:
            tau_k = tau_global
        taus.append(tau_k)
    return torch.stack(taus)

def _smooth_pred_thr_mondrian(scores: torch.Tensor,
                              tau_vec: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
    """Soft membership with per-class thresholds: sigma((scores - tau_k)/T)."""
    T = max(1e-6, float(temperature))
    return torch.sigmoid((scores - tau_vec.view(1, -1)) / T)


def _alpha_metric_suffix(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "_")


def _empty_true_label_pvalue_diag(alphas=TRUE_LABEL_PVALUE_ALPHAS):
    diag = {
        "true_p_mean": float("nan"),
        "true_p_min": float("nan"),
        "true_p_uniform_ks": float("nan"),
        "true_p_superuniform_violation": float("nan"),
        "true_p_n": 0.0,
    }
    for alpha in alphas:
        diag[f"true_p_coverage_at_{_alpha_metric_suffix(alpha)}"] = float("nan")
    return diag


def _true_label_pvalue_diagnostics_mondrian(
    scores: torch.Tensor,
    y: torch.Tensor,
    cal_idx: torch.Tensor,
    pred_idx: torch.Tensor,
    K: int,
    alphas=TRUE_LABEL_PVALUE_ALPHAS,
):
    """
    Detached diagnostic for class-conditional true-label conformal p-values.
    Larger scores are better, so p_y(x) = (1 + #{cal same-class scores <= score_y(x)}) / (n_k + 1).
    """
    diag = _empty_true_label_pvalue_diag(alphas=alphas)
    if cal_idx.numel() == 0 or pred_idx.numel() == 0:
        return diag

    with torch.no_grad():
        scores_d = scores.detach()
        y_d = y.detach()
        pvals = []

        for k in range(K):
            cal_k = cal_idx[y_d[cal_idx] == k]
            pred_k = pred_idx[y_d[pred_idx] == k]
            if cal_k.numel() == 0 or pred_k.numel() == 0:
                continue

            cal_scores_k = torch.sort(scores_d[cal_k, k].reshape(-1)).values
            pred_scores_k = scores_d[pred_k, k].reshape(-1)
            counts_le = torch.searchsorted(cal_scores_k, pred_scores_k, right=True).to(scores_d.dtype)
            p_k = (counts_le + 1.0) / float(cal_scores_k.numel() + 1)
            pvals.append(p_k)

        if len(pvals) == 0:
            return diag

        p = torch.cat(pvals).clamp(0.0, 1.0)
        p_sorted = torch.sort(p).values
        n = p_sorted.numel()
        idx = torch.arange(1, n + 1, device=p_sorted.device, dtype=p_sorted.dtype)

        ecdf_at_p = idx / float(n)
        ecdf_before_p = (idx - 1.0) / float(n)
        superuniform_violation = torch.clamp(ecdf_at_p - p_sorted, min=0.0).max()
        uniform_ks = torch.maximum(
            torch.abs(ecdf_at_p - p_sorted),
            torch.abs(p_sorted - ecdf_before_p),
        ).max()

        diag.update(
            {
                "true_p_mean": float(p.mean().detach().cpu()),
                "true_p_min": float(p.min().detach().cpu()),
                "true_p_uniform_ks": float(uniform_ks.detach().cpu()),
                "true_p_superuniform_violation": float(superuniform_violation.detach().cpu()),
                "true_p_n": float(n),
            }
        )
        for alpha in alphas:
            diag[f"true_p_coverage_at_{_alpha_metric_suffix(alpha)}"] = float((p > alpha).float().mean().detach().cpu())

    return diag


def _smooth_calibrate_thr_mondrian_by_group(
    scores: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    K: int,
    alpha: float,
    eps_sort: float = 1e-2,
    min_n: int = 8,
):
    """
    Returns:
      taus_gk: (G,K) per-group per-class tau_k^g
      mask_gk: (G,K) bool -> True if class k had >= min_n samples in group g
    """
    G = int(g.max().item()) + 1
    taus = torch.zeros(G, K, device=scores.device)
    mask = torch.zeros(G, K, dtype=torch.bool, device=scores.device)

    tau_global = _smooth_calibrate_thr_mondrian(scores, y, K, alpha, eps_sort) 

    for gg in range(G):
        mg = (g == gg)
        for k in range(K):
            mk = mg & (y == k)
            if mk.sum().item() >= min_n:
                e_k = scores[mk, k].reshape(-1)
                n_k = e_k.numel()
                q_k = alpha * (1.0 + 1.0 / n_k)
                taus[gg, k] = _smooth_quantile_1d(e_k, q=q_k, tau=eps_sort)
                mask[gg, k] = True
            else:
                taus[gg, k] = tau_global[k]
    return taus, mask


    


def _balanced_indices_for_calibration(
    y: torch.Tensor,
    cal_mask: Optional[torch.Tensor],
    *,
    min_per_class: int = 5,
    min_total: int = 16,
    num_classes: Optional[int] = None,
    rng: Optional[torch.Generator] = None
):
    """
    Build B_cal indices with (a) reference-only preference, (b) per-class minimum, (c) global cap.
    Falls back to a random half split if nothing else is possible.
    """
    device = y.device
    N = y.size(0)
    g = rng if rng is not None else torch.Generator(device=device)
    if rng is None:
        g.manual_seed(torch.randint(0, 2**31-1, (1,), device=device).item())

    all_idx = torch.arange(N, device=device)

    if cal_mask is None:
        pool_idx = all_idx
    else:
        pool_idx = all_idx[cal_mask]
        if pool_idx.numel() == 0:
            return torch.tensor([], device=device, dtype=torch.long), all_idx

    cap_frac: float = 0.5
    max_cal = max(1, int(cap_frac * pool_idx.numel()))

    if num_classes is None:
        K = int(y.max().item()) + 1
    else:
        K = num_classes

    if min_per_class <= 0:
        take = min(max_cal, pool_idx.numel())
        cal_idx = pool_idx[torch.randperm(pool_idx.numel(), generator=g, device=device)[:take]]

    else:

        parts = []
        ys = y[pool_idx]
        for k in range(K):

            k_idx = pool_idx[ys == k]
            if k_idx.numel() == 0:
                continue

            take_k = min(int(k_idx.numel()*0.5), min_per_class)

            perm = torch.randperm(k_idx.numel(), generator=g, device=device)[:take_k]
            parts.append(k_idx[perm])

        if len(parts) == 0:
            take = min(max_cal, pool_idx.numel())
            cal_idx = pool_idx[torch.randperm(pool_idx.numel(), generator=g, device=device)[:take]]

        else:
            cal_idx = torch.unique(torch.cat(parts))
            if cal_idx.numel() < min_total:
                remaining = pool_idx[~torch.isin(pool_idx, cal_idx)]
                need = min_total - cal_idx.numel()
                need = min(need, remaining.numel())
                if need > 0:
                    extra = remaining[torch.randperm(remaining.numel(), generator=g, device=device)[:need]]
                    cal_idx = torch.unique(torch.cat([cal_idx, extra]))

            if cal_idx.numel() > max_cal:
                perm = torch.randperm(cal_idx.numel(), generator=g, device=device)[:max_cal]
                cal_idx = cal_idx[perm]

    pred_mask = torch.ones(N, dtype=torch.bool, device=device)
    pred_mask[cal_idx] = False
    pred_idx = all_idx[pred_mask]

    if pred_idx.numel() == 0:
        pred_idx = all_idx


    return cal_idx, pred_idx


def conftr_loss_mondrian_balanced(
    scores: torch.Tensor, y: torch.Tensor, alpha: float,
    *,
    cal_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    eps_sort: float = 1e-2,
    kappa: float = 1.0,
    lambda_size: float = 1.0,
    cover_weight: float = 0.5,
    other_weight: float = 0.5,
    min_cal_per_class: int = 0,
    min_cal_total: int = 16,
    num_classes: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    batch: Optional[torch.Tensor] = None,
    ref_batch_id: int = 0,
    gamma_tau_align: float = 0.6,
    
):
    """
    Conformal training with *Mondrian* thresholds (per-class tau_k) and the same
    balanced B_cal / B_pred split you already use.
    Pass `scores` as log-probs (recommended) or probabilities; be consistent. And an
    optional exchangeability regularizer that aligns per-batch classwise
    quantiles (τ_k^g) to the reference batch
    """
    device = scores.device
    K = num_classes if num_classes is not None else scores.size(1)

    cal_idx, pred_idx = _balanced_indices_for_calibration(
        y, cal_mask,
        min_per_class=min_cal_per_class,
        min_total=min_cal_total,
        num_classes=K,
        rng=rng
    )

    
    if cal_idx.numel() == 0:
        zero = scores.new_tensor(0.0)
        diag = {
            "tau": 0.0 if "thr_balanced" in __name__ else None,
            "tau_by_class": torch.zeros(
                num_classes if num_classes is not None else scores.size(1),
                device=scores.device
            ).cpu() if "mondrian" in __name__ else None,
            "mean_set_size_pred": 0.0,
            "mean_L_class": 0.0,
            "mean_Omega": 0.0,
            "B_cal": 0,
            "B_pred": int(pred_idx.numel()),
            "pred_idx": pred_idx.detach().cpu(),
            "cal_idx": cal_idx.detach().cpu(),
            "cer": 0.0,
            "conftr_base_loss": zero.detach(),
            "conftr_total_loss": zero.detach(),
            "mean_include_true": 0.0,
            "mean_include_others": 0.0,
            "tau_mean": 0.0,
            "tau_min": 0.0,
            "tau_max": 0.0,
            **_empty_true_label_pvalue_diag(),
        }
        return zero, diag

    tau_vec = _smooth_calibrate_thr_mondrian(
        scores[cal_idx], y[cal_idx], K=K, alpha=alpha, eps_sort=eps_sort
    )

    C = _smooth_pred_thr_mondrian(scores[pred_idx], tau_vec, temperature=temperature)

    rows = torch.arange(pred_idx.numel(), device=device)
    include_true = C[rows, y[pred_idx]]
    include_others = (C.sum(dim=1) - include_true)

    L_class = cover_weight * (1.0 - include_true) + other_weight * include_others

    Omega = torch.relu(C.sum(dim=1) - float(kappa))

    base_loss = torch.log(1e-6 + (L_class + float(lambda_size) * Omega).mean())

    cer = scores.new_tensor(0.0)
    info_align = {}

    if batch is not None and (gamma_tau_align > 0.0):

        taus_gk, mask_gk = _smooth_calibrate_thr_mondrian_by_group(
            scores=scores, y=y, g=batch, K=K, alpha=alpha, eps_sort=eps_sort)
        

        tau_ref = taus_gk[ref_batch_id]

        if gamma_tau_align > 0.0:

            diff = torch.abs(taus_gk - tau_ref.unsqueeze(0))
            
            valid = mask_gk.clone()

            if ref_batch_id < valid.size(0):
                valid[ref_batch_id, :] = False

            if valid.any():
                cer = cer + gamma_tau_align * diff[valid].mean()
                info_align["tau_align_mean_abs"] = float(diff[valid].mean().detach().cpu())
            else:
                info_align["tau_align_mean_abs"] = 0.0

    loss = base_loss + cer


    diag = {
        "tau_by_class": tau_vec.detach().cpu(),
        "conftr_base_loss": base_loss.detach(),
        "conftr_total_loss": loss.detach(),
        "mean_set_size_pred": float(C.sum(dim=1).mean().detach().cpu()),
        "mean_include_true": float(include_true.mean().detach().cpu()),
        "mean_include_others": float(include_others.mean().detach().cpu()),
        "mean_L_class": float(L_class.mean().detach().cpu()),
        "mean_Omega": float(Omega.mean().detach().cpu()),
        "B_cal": int(cal_idx.numel()),
        "B_pred": int(pred_idx.numel()),
        "pred_idx": pred_idx.detach().cpu(),
        "cal_idx": cal_idx.detach().cpu(),
        "cer": float(cer.detach().cpu()),
        "tau_mean": float(tau_vec.mean().detach().cpu()),
        "tau_min": float(tau_vec.min().detach().cpu()),
        "tau_max": float(tau_vec.max().detach().cpu()),
        **_true_label_pvalue_diagnostics_mondrian(scores, y, cal_idx, pred_idx, K=K),
        **info_align,
    }
    return loss, diag




def one_hot(index: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 0:
        shape = index.shape + (0,)
        return torch.zeros(shape, device=index.device, dtype=torch.float32)
    return F.one_hot(index.long(), num_classes=num_classes).float()


def mlp(layer_sizes: Tuple[int, ...], dropout: float = 0.0, activation=nn.SiLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != layer_sizes[-1]:
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)



@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int = 16
    num_batches: int = None
    num_cell_types: int = 0
    hidden: Tuple[int, ...] = vae_hidden_layers
    dropout: float = 0.1


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        in_dim = cfg.input_dim + cfg.num_batches + cfg.num_cell_types
        self.net = mlp((in_dim, *cfg.hidden), dropout=cfg.dropout)
        self.mu = nn.Linear(cfg.hidden[-1], cfg.latent_dim)
        self.logvar = nn.Linear(cfg.hidden[-1], cfg.latent_dim)

    def forward(self, x: torch.Tensor, b_oh: torch.Tensor, y_oh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, b_oh, y_oh], dim=-1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10.0, 5.0)
        return mu, logvar


class DecoderGaussian(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.latent_dim + cfg.num_batches + cfg.num_cell_types
        self.net = mlp((in_dim, *cfg.hidden), dropout=cfg.dropout)
        self.mu_x = nn.Linear(cfg.hidden[-1], cfg.input_dim)
        self.logvar_x = nn.Linear(cfg.hidden[-1], cfg.input_dim)
        with torch.no_grad():
            self.logvar_x.bias.fill_(-1.5)

    def forward(self, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b_oh = one_hot(b, self.cfg.num_batches)
        y_oh = one_hot(y, self.cfg.num_cell_types)
        h = self.net(torch.cat([z, b_oh, y_oh], dim=-1))
        mu = self.mu_x(h)
        logvar = self.logvar_x(h).clamp(-5.0, 5.0)
        return mu, torch.exp(logvar)




class ConditionalVAE(nn.Module):
    def __init__(self, cfg: VAEConfig, enable_classification=True):
        super().__init__()
        self.cfg = cfg
        self.enable_classification = enable_classification
        self.encoder = Encoder(cfg)
        self.decoder = DecoderGaussian(cfg)

        self.alpha = 0.1
        
        self._cov_epoch_total = None
        self._cov_epoch_covered = None

        self.current_batch_class_acc = 0.0 
        self.current_conftr_diag = {}

        self.conf_eps = 1e-2
        self.cover_weight: float = 1.0
        self.other_weight: float = 0.5
        self.min_cal_per_class: int = 10
        self.min_cal_total:int = 64
        self.num_classes = cfg.num_cell_types
        self.kappa: float = 1.5

        self.conf_T: float = None
        self.lambda_size: float = None
        self.gamma_tau_align: float = None     


        
        if enable_classification and cfg.num_cell_types > 0:
            self.classifier = nn.Sequential(
                nn.Linear(cfg.latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, cfg.num_cell_types)
            )
        else:
            self.classifier = None
            
        

    def encode(self, x: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b_oh = one_hot(b, self.cfg.num_batches)
        y_oh = one_hot(y, self.cfg.num_cell_types)
        return self.encoder(x, b_oh, y_oh)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(z, b, y)

    def log_prob_x_given_z(self, x: torch.Tensor, z: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        EPS = 1e-8
        mu_x, var_x = self.decode(z, b, y)
        log_prob = -0.5 * ((x - mu_x) ** 2 / (var_x + EPS) + torch.log(var_x + EPS) + math.log(2 * math.pi))
        return log_prob.sum(dim=-1)


    def forward(self, x: torch.Tensor,
                      b: torch.Tensor,
                      y: torch.Tensor,
                      reference_batch_dict: dict = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        
        mu_z, logvar_z = self.encode(x, b, y)
        z = self.reparameterize(mu_z, logvar_z)
        recon_loglik = self.log_prob_x_given_z(x, z, b, y)
        kl = 0.5 * (mu_z.pow(2) + torch.exp(logvar_z) - logvar_z - 1.0).sum(dim=-1)

        classification_info: Optional[float] = None
        classification_loss: Optional[torch.Tensor] = None
        self.current_conftr_diag = {}
        debug_conftr_diags = []
        if self.classifier is not None and self.enable_classification:
            
            logits = self.classifier(mu_z)
            log_probs = F.log_softmax(logits, dim=1)

            per_task_losses = []
            per_task_sizes = []
            per_task_mean_set = []
            counted = torch.zeros_like(b, dtype=torch.bool)
            
            for ref_idx, target_idxs in reference_batch_dict.items():
                mask = (b == ref_idx)
                
                for t in target_idxs:
                    mask |= (b == t)
                
                if not mask.any():
                    continue       
            
                scores_s = log_probs[mask]
                y_s = y[mask]
                b_s = b[mask]
                  
                cal_mask_s = (b_s == ref_idx)
                  
                loss_s, diag_s = conftr_loss_mondrian_balanced(
                    scores=scores_s,
                    y=y_s,
                    alpha=self.alpha,
                    cal_mask=cal_mask_s,
                    temperature=self.conf_T,
                    eps_sort=self.conf_eps,
                    kappa=self.kappa,
                    lambda_size=self.lambda_size,
                    cover_weight=self.cover_weight,
                    other_weight=self.other_weight,
                    min_cal_per_class=self.min_cal_per_class,
                    min_cal_total=self.min_cal_total,
                    num_classes=self.num_classes,
                    batch=b_s.long(),
                    ref_batch_id=ref_idx,
                    gamma_tau_align=self.gamma_tau_align,
                )

                

                if int(diag_s.get("B_cal", 0)) == 0:
                    continue

                debug_conftr_diags.append(diag_s)
                per_task_losses.append(loss_s)
                per_task_sizes.append(mask.sum())
                per_task_mean_set.append(float(diag_s.get("mean_set_size_pred", 0.0)))


                if (self._cov_epoch_total is not None) and (self._cov_epoch_covered is not None):
                    with torch.no_grad():
                        new_samples_mask = mask & (~counted)
                        if new_samples_mask.any():
                            y_ns = y[new_samples_mask]
                            rows = torch.arange(new_samples_mask.sum().item(), device=y.device)
                            true_scores_ns = log_probs[new_samples_mask][rows, y_ns]
                            tau_vec = diag_s["tau_by_class"].to(y.device)
                            hard_cover_ns = (true_scores_ns >= tau_vec[y_ns]).to(torch.float32)

                            K = self.cfg.num_cell_types
                            total_add = torch.bincount(y_ns, minlength=K)
                            covered_add = torch.bincount(y_ns, weights=hard_cover_ns, minlength=K)

                            self._cov_epoch_total[:K] += total_add.to(self._cov_epoch_total.device)
                            self._cov_epoch_covered[:K] += covered_add.to(self._cov_epoch_covered.device)

                            counted = counted | new_samples_mask

            if len(per_task_losses) > 0:
                w = torch.stack([s.to(mu_z).float() for s in per_task_sizes])
                w = w / w.sum().clamp_min(1.0)
                
                classification_loss = torch.stack(per_task_losses).mul(w).sum()

                mean_set_tensor = torch.tensor(per_task_mean_set, device=mu_z.device, dtype=mu_z.dtype)
                classification_info = float((w * mean_set_tensor).sum().detach().item())

            if len(debug_conftr_diags) > 0:
                def _diag_scalar(key: str, reducer: str = "mean") -> float:
                    vals = []
                    for d in debug_conftr_diags:
                        if key not in d:
                            continue
                        v = d[key]
                        if v is None:
                            continue
                        if torch.is_tensor(v):
                            if v.numel() == 1:
                                vals.append(v.detach().float().reshape(()))
                        elif isinstance(v, (int, float)):
                            vals.append(torch.as_tensor(float(v), device=mu_z.device))

                    if len(vals) == 0:
                        return float("nan")
                    stacked = torch.stack(vals)
                    if reducer == "min":
                        value = stacked.min()
                    elif reducer == "max":
                        value = stacked.max()
                    elif reducer == "sum":
                        value = stacked.sum()
                    else:
                        value = stacked.mean()
                    return float(value.detach().cpu())

                self.current_conftr_diag = {
                    "conftr_base_loss": _diag_scalar("conftr_base_loss"),
                    "conftr_total_loss": _diag_scalar("conftr_total_loss"),
                    "cer": _diag_scalar("cer"),
                    "tau_align_mean_abs": _diag_scalar("tau_align_mean_abs"),
                    "mean_set_size_pred": _diag_scalar("mean_set_size_pred"),
                    "mean_include_true": _diag_scalar("mean_include_true"),
                    "mean_include_others": _diag_scalar("mean_include_others"),
                    "mean_L_class": _diag_scalar("mean_L_class"),
                    "mean_Omega": _diag_scalar("mean_Omega"),
                    "B_cal": _diag_scalar("B_cal"),
                    "B_pred": _diag_scalar("B_pred"),
                    "tau_mean": _diag_scalar("tau_mean"),
                    "tau_min": _diag_scalar("tau_min"),
                    "tau_max": _diag_scalar("tau_max"),
                    "true_p_mean": _diag_scalar("true_p_mean"),
                    "true_p_min": _diag_scalar("true_p_min", reducer="min"),
                    "true_p_uniform_ks": _diag_scalar("true_p_uniform_ks", reducer="max"),
                    "true_p_superuniform_violation": _diag_scalar("true_p_superuniform_violation", reducer="max"),
                    "true_p_n": _diag_scalar("true_p_n", reducer="sum"),
                }
                for alpha in TRUE_LABEL_PVALUE_ALPHAS:
                    key = f"true_p_coverage_at_{_alpha_metric_suffix(alpha)}"
                    self.current_conftr_diag[key] = _diag_scalar(key)

        return recon_loglik, kl, classification_info, classification_loss


    def get_classification_score(self) -> float:
        return self.current_batch_class_acc


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 5e-4
    batch_size: int = 124
    weight_decay: float = 5e-4
    kl_anneal_epochs: int = 5
    grad_clip: Optional[float] =  1.0
    lr_gamma: float =  0.5
    lr_patience: int = 4
    min_lr: float = 1e-5
    early_stopping_patience: int = 8


def _kl_beta(epoch: int, cfg: TrainConfig) -> float:
    if cfg.kl_anneal_epochs <= 0:
        return 1.0
    return min(2.0, (epoch + 1) / cfg.kl_anneal_epochs)


def train_cvae(
    model: ConditionalVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: TrainConfig,
    reference_batch_dict: dict = None,
    cal_indices: Optional[list] = None,
    cell_types: Optional[list] = None,
    epoch_callback: Optional[Callable[[int, "ConditionalVAE", bool], None]] = None,
    device: torch.device = torch.device("cpu"),
    beta : float = None,
    epochs_CG_start : int = None,
    conf_T_init : float = None,
    conf_T_max_decay : float = None,
    lambda_size : float = None,
    gamma_tau_align : float = None,
    verbose: int | bool = 2,
    ) -> None:
    
    verbose = _normalize_verbose(verbose)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler_kwargs = {
        "mode": "min",
        "factor": cfg.lr_gamma,
        "patience": cfg.lr_patience,
        "min_lr": cfg.min_lr,
    }
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            verbose=(verbose >= 2),
            **scheduler_kwargs,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **scheduler_kwargs)

    if not hasattr(model, "history"):
        model.history = []

    best_state = None
    no_improve_cov_overall = 0
    best_monitor_abs_gap = float("inf")
    best_refinement_metric = float("inf")

    beta = 1.0 if beta is None else beta
    epochs_CG_start = 20 if epochs_CG_start is None else epochs_CG_start
    conf_T_init = 1.5 if conf_T_init is None else conf_T_init
    conf_T_max_decay = 0.5 if conf_T_max_decay is None else conf_T_max_decay
    lambda_size = 1.0 if lambda_size is None else lambda_size
    gamma_tau_align = 1.5 if gamma_tau_align is None else gamma_tau_align

    epochs_CG_end =  epochs_CG_start + 25
    warm_T_epochs = int(epochs_CG_end - epochs_CG_start)
    conf_T_hold_epochs = 8

    model.conf_T = 1.00
    model.conf_T_init = conf_T_init
    model.conf_T_max_decay = conf_T_max_decay
    total_decay_amount = model.conf_T_init - model.conf_T_max_decay

    model.lambda_size = lambda_size
    model.gamma_tau_align = gamma_tau_align

    _vprint(verbose, 1, "\nPARAMETERS:")
    _vprint(verbose, 1, f"  epochs_CG_start: {epochs_CG_start}")
    _vprint(verbose, 1, f"  epochs_CG_end: {epochs_CG_end}")
    _vprint(verbose, 1, f"  batch_size: {cfg.batch_size}")
    _vprint(verbose, 1, "ConfTr parameters: ")
    _vprint(verbose, 1, f"  conf_T_init: {model.conf_T_init}")
    _vprint(verbose, 1, f"  conf_T_max_decay: {model.conf_T_max_decay}")
    _vprint(verbose, 1, f"  warm_T_epochs: {warm_T_epochs}")
    _vprint(verbose, 1, f"  conf_T_hold_epochs: {conf_T_hold_epochs}")
    _vprint(verbose, 1, f"  kappa: {model.kappa}")
    _vprint(verbose, 1, f"  lambda_size: {model.lambda_size}")
    _vprint(verbose, 1, f"  cover_weight: {model.cover_weight}")
    _vprint(verbose, 1, f"  other_weight: {model.other_weight}")
    _vprint(verbose, 1, f"  gamma_tau_align: {model.gamma_tau_align}")

   

    Flag_lr_scheduler_step = False
    start_best_val_metric_flag = False
    best_val_true_incl = float("-inf")
    no_improve_val_true_incl = 0
    true_incl_patience = max(1, 12)
    best_val_conftr_total_loss = float("inf")
    no_improve_val_conftr_total_loss = 0
    best_val_cer = float("inf")
    no_improve_val_cer = 0
    best_val_true_p_superuniform_violation = float("inf")
    no_improve_val_true_p_superuniform_violation = 0
    cer_patience = max(1, int(math.ceil(cfg.early_stopping_patience / 2)))
    temp_target_stable_epochs = 0
    allow_refinement_stop_before_temp_target = False

    def _collect_model_conftr_debug(bucket, model):
        d = getattr(model, "current_conftr_diag", {})
        if not isinstance(d, dict) or len(d) == 0:
            return

        for key, value in d.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                bucket[key].append(float(value))

    conftr_debug_reducers = {
        "true_p_min": "min",
        "true_p_uniform_ks": "max",
        "true_p_superuniform_violation": "max",
        "true_p_n": "sum",
    }

    def _safe_stat(bucket, key, reducer=None):
        vals = bucket.get(key, [])
        if len(vals) == 0:
            return float("nan")
        reducer = reducer or conftr_debug_reducers.get(key, "mean")
        arr = np.asarray(vals, dtype=float)
        if reducer == "min":
            return float(np.nanmin(arr))
        if reducer == "max":
            return float(np.nanmax(arr))
        if reducer == "sum":
            return float(np.nansum(arr))
        return float(np.nanmean(arr))

    def _safe_mean(bucket, key):
        return _safe_stat(bucket, key, reducer="mean")

    def _add_bucket_means(entry, prefix, bucket, keys):
        for key in keys:
            entry[f"{prefix}_{key}"] = float(_safe_stat(bucket, key))

    def _print_conftr_debug(prefix, bucket):
        if len(bucket) == 0:
            _vprint(verbose, 2, f"{prefix}: no ConfTr diagnostics collected")
            return

        _vprint(
            verbose,
            2,
            f"{prefix}: "
            f"raw_total={_safe_mean(bucket, 'conftr_total_loss'):.3f} | "
            f"raw_base={_safe_mean(bucket, 'conftr_base_loss'):.3f} | "
            f"cer={_safe_mean(bucket, 'cer'):.4f} | "
            f"tau_align={_safe_mean(bucket, 'tau_align_mean_abs'):.4f} | "
            f"set_size={_safe_mean(bucket, 'mean_set_size_pred'):.3f} | "
            f"true_incl={_safe_mean(bucket, 'mean_include_true'):.3f} | "
            f"other_incl={_safe_mean(bucket, 'mean_include_others'):.3f} | "
            f"L_class={_safe_mean(bucket, 'mean_L_class'):.4f} | "
            f"Omega={_safe_mean(bucket, 'mean_Omega'):.4f} | "
            f"Bcal={_safe_mean(bucket, 'B_cal'):.1f} | "
            f"Bpred={_safe_mean(bucket, 'B_pred'):.1f} | "
            f"tau_mean={_safe_mean(bucket, 'tau_mean'):.3f} | "
            f"true_p_mean={_safe_mean(bucket, 'true_p_mean'):.3f} | "
            f"true_p_min={_safe_stat(bucket, 'true_p_min'):.3f} | "
            f"su_violation={_safe_stat(bucket, 'true_p_superuniform_violation'):.3f}"
        )

    def _coverage_stats_from_model(model, cell_types):
        with torch.no_grad():
            cov_total = model._cov_epoch_total.detach().cpu().numpy()
            cov_cov = model._cov_epoch_covered.detach().cpu().numpy()
            cov_total_sum = model._cov_epoch_total.sum().item()
            cov_cov_sum = model._cov_epoch_covered.sum().item()
            cov_overall = (cov_cov_sum / max(1.0, cov_total_sum)) if cov_total_sum > 0 else float("nan")
            per_class_cov = {}
            for k, name in enumerate(cell_types):
                if cov_total[k] > 0:
                    per_class_cov[name] = float(cov_cov[k] / cov_total[k])

            target_coverage = 1.0 - model.alpha
            if np.isfinite(cov_overall):
                coverage_gap = target_coverage - cov_overall
                coverage_gap_monitor = abs(coverage_gap)
                coverage_shortfall = max(coverage_gap, 0.0)
            else:
                coverage_gap = float("nan")
                coverage_gap_monitor = float("nan")
                coverage_shortfall = float("nan")

        return {
            "cov_overall": cov_overall,
            "per_class_cov": per_class_cov,
            "coverage_gap": coverage_gap,
            "coverage_gap_monitor": coverage_gap_monitor,
            "coverage_shortfall": coverage_shortfall,
        }

    def _reset_epoch_coverage(model):
        K = model.cfg.num_cell_types
        model._cov_epoch_total = torch.zeros(K, dtype=torch.float32, device=device)
        model._cov_epoch_covered = torch.zeros(K, dtype=torch.float32, device=device)
    for epoch in range(cfg.epochs):
        model.train()
        
        debug_train_conftr = defaultdict(list)
        debug_val_conftr = defaultdict(list)
        _reset_epoch_coverage(model)

         
        total_elbo = 0.0
        total_n = 0
        total_loss = 0.0
        total_recon_nll = 0.0
        total_kl = 0.0
        total_conftr_scaled_loss = 0.0
        

        if epoch <= epochs_CG_start + 1:
            model.conf_T = model.conf_T_init
        else:
            effective_epoch = epoch - (epochs_CG_start + 1)
            held_step = effective_epoch // conf_T_hold_epochs
            decay_steps = max(1, math.ceil(warm_T_epochs / conf_T_hold_epochs))
            decay_progress = min(1.0, held_step / decay_steps)
            decay = total_decay_amount * decay_progress
            model.conf_T = float(model.conf_T_init - decay)

        if abs(model.conf_T - model.conf_T_max_decay) <= 1e-12:
            temp_target_stable_epochs += 1
        else:
            temp_target_stable_epochs = 0
        can_stop_after_temp_stable = temp_target_stable_epochs >= 6

        if epoch == epochs_CG_start + 1:
            _vprint(verbose, 1, "\n--- Starting Conformal Training with SmoothCal loss ---\n")
            best_monitor_abs_gap = float("inf")
            best_refinement_metric = float("inf")
            no_improve_cov_overall = 0
            best_val_conftr_total_loss = float("inf")
            no_improve_val_conftr_total_loss = 0
            best_val_cer = float("inf")
            no_improve_val_cer = 0
            best_val_true_p_superuniform_violation = float("inf")
            no_improve_val_true_p_superuniform_violation = 0

        for xb, bb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            bb = bb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            recon_loglik, kl, batch_acc, batch_cls_loss = model(xb, bb, yb, reference_batch_dict=reference_batch_dict)
            _collect_model_conftr_debug(debug_train_conftr, model)
            elbo = recon_loglik - beta * kl
            
            
            loss = -elbo.mean()
            

            if epoch > epochs_CG_start and batch_cls_loss is not None:
  
                batch_loss_weight = 1.5
                weighted_batch_loss = batch_loss_weight * batch_cls_loss

                cvae_magnitude = abs(loss.item())
                batch_magnitude = abs(weighted_batch_loss.detach().item())

                if batch_magnitude > 1e-3:
                    target_ratio = 0.5
                    scaling_factor = (target_ratio * cvae_magnitude) / batch_magnitude
                    scaling_factor = min(scaling_factor, 5000.0)

                    scaled_batch_loss = scaling_factor * weighted_batch_loss
                    loss += scaled_batch_loss

                    total_conftr_scaled_loss += scaled_batch_loss.detach().item()* xb.size(0)
                    

            opt.zero_grad() 
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_elbo += elbo.detach().sum().item()
            total_n += xb.size(0)
            total_loss += loss.detach().item() * xb.size(0)

            total_recon_nll += recon_loglik.detach().sum().item()
            total_kl += beta * kl.detach().sum().item()

        train_elbo = total_elbo / max(1, total_n)
        train_recon_nll = total_recon_nll / max(1, total_n)
        train_kl = total_kl / max(1, total_n)
        train_conftr_scaled_loss = total_conftr_scaled_loss / max(1, total_n)
        train_loss = total_loss / max(1, total_n)
        train_cov_stats = _coverage_stats_from_model(model, cell_types)
        
        

        val_elbo = None
        val_cov_stats = {
            "cov_overall": float("nan"),
            "per_class_cov": {},
            "coverage_gap": float("nan"),
            "coverage_gap_monitor": float("nan"),
            "coverage_shortfall": float("nan"),
        }
        if val_loader is not None:

            model.eval()
            _reset_epoch_coverage(model)

            total_val_elbo = 0.0
            val_n = 0
            total_val_recon_nll = 0.0
            total_val_kl = 0.0
            total_val_conftr_scaled_loss = 0.0
            total_val_loss = 0.0
            with torch.no_grad():
                for xb, bb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    bb = bb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    
                    
                    recon_loglik, kl, batch_acc, batch_cls_loss = model(xb, bb, yb, reference_batch_dict=reference_batch_dict)
                    _collect_model_conftr_debug(debug_val_conftr, model)
                    elbo = recon_loglik - beta * kl


                    loss_val = -elbo.mean()


                    if epoch > epochs_CG_start and batch_cls_loss is not None and torch.isfinite(batch_cls_loss):

                        
                        batch_loss_weight = 1.5
                        weighted_batch_loss = batch_loss_weight * batch_cls_loss

                        cvae_magnitude = abs(loss_val.item())
                        batch_magnitude = abs(weighted_batch_loss.detach().item())

                        if batch_magnitude > 1e-3:
                            target_ratio = 0.4
                            scaling_factor = (target_ratio * cvae_magnitude) / batch_magnitude
                            scaling_factor = min(scaling_factor, 5000.0)

                            scaled_batch_loss = scaling_factor * weighted_batch_loss
                            loss_val += scaled_batch_loss

                            total_val_conftr_scaled_loss += scaled_batch_loss.detach().item()* xb.size(0)
                            

                    total_val_elbo += elbo.detach().sum().item()
                    val_n += xb.size(0)
                    total_val_loss += loss_val.detach().item() * xb.size(0)

                    total_val_recon_nll += recon_loglik.detach().sum().item()
                    total_val_kl += beta * kl.detach().sum().item()


            val_elbo = total_val_elbo / max(1, val_n)
            val_recon_nll = total_val_recon_nll / max(1, val_n)
            val_kl = total_val_kl / max(1, val_n)
            val_conftr_scaled_loss = total_val_conftr_scaled_loss / max(1, val_n)
            val_loss = total_val_loss / max(1, val_n)
            val_cov_stats = _coverage_stats_from_model(model, cell_types)

            metric = val_loss

        else:
            metric = train_elbo

        cov_overall = train_cov_stats["cov_overall"]
        per_class_cov = train_cov_stats["per_class_cov"]
        coverage_gap_monitor = train_cov_stats["coverage_gap_monitor"]

        if (epoch + 1) % 1 == 0 or epoch == 0:
            msg = (
                f"Epoch {epoch+1:03d} | beta={beta:.3f} | LR={opt.param_groups[0]['lr']:.6f} "
                f"| T={model.conf_T:.2f} | gamma_tau={model.gamma_tau_align:.2f} "
                f"| CW={model.cover_weight:.2f} | kappa={model.kappa:.2f} "
                f"| Cov(overall)={cov_overall:.3f} | AbsGap={coverage_gap_monitor:.3f}"
            )
            _vprint(verbose, 1, msg)

            _vprint(verbose, 2, f"--Train Loss: ELBO={-train_elbo:.3f} | ConfTr={train_conftr_scaled_loss:.3f} | Total={train_loss:.3f} ")
            _vprint(verbose, 2, f"              Recon NLL: {train_recon_nll:.3f} | KL: {train_kl:.3f}")
            if val_elbo is not None:
                _vprint(verbose, 2, f"--Val Loss: ELBO={-val_elbo:.3f} | ConfTr={val_conftr_scaled_loss:.3f} | Total={val_loss:.3f} ")
                _vprint(verbose, 2, f"            Recon NLL: {val_recon_nll:.3f} | KL: {val_kl:.3f}")
            
            if per_class_cov:
                pcs = ", ".join([f"{k}:{v:.2f}" for k, v in per_class_cov.items()])
                _vprint(verbose, 2, f"--Per-class coverage: {pcs}\n")

            _print_conftr_debug("--Train ConfTr raw", debug_train_conftr)
            if val_elbo is not None:
                _print_conftr_debug("--Val ConfTr raw", debug_val_conftr)

        if Flag_lr_scheduler_step == False:
            scheduler.step(metric)

        callback_covgap = float("inf")
        if epoch_callback is not None and (epoch + 1) % 1 == 0:
            callback_covgap = epoch_callback(epoch + 1, model, False)
            _vprint(verbose, 2, f"callback_covgap: {callback_covgap}")
                    
        entry = {
            "epoch": int(epoch + 1),
            "beta": float(beta),
            "conf_T": float(model.conf_T),
            "train_total_loss": float(train_loss),
            "train_elbo_loss": float(-train_elbo),
            "train_recon_nll": float(train_recon_nll),
            "train_kl": float(train_kl),
            "train_conftr_scaled": float(train_conftr_scaled_loss),
            "val_total_loss": float(val_loss) if 'val_loss' in locals() else float('nan'),
            "val_elbo_loss": float(-val_elbo) if 'val_elbo' in locals() and val_elbo is not None else float('nan'),
            "val_recon_nll": float(val_recon_nll) if 'val_recon_nll' in locals() else float('nan'),
            "val_kl": float(val_kl) if 'val_kl' in locals() else float('nan'),
            "val_conftr_scaled": float(val_conftr_scaled_loss) if 'val_conftr_scaled_loss' in locals() else float('nan'),
            "coverage_overall": float(cov_overall),
            "train_coverage_gap": float(train_cov_stats["coverage_gap_monitor"]),
            "val_coverage_gap": float(val_cov_stats["coverage_gap_monitor"]),
            "train_coverage_shortfall": float(train_cov_stats["coverage_shortfall"]),
            "val_coverage_shortfall": float(val_cov_stats["coverage_shortfall"]),
            "train_avg_set_size": float(_safe_mean(debug_train_conftr, "mean_set_size_pred")),
            "val_avg_set_size": float(_safe_mean(debug_val_conftr, "mean_set_size_pred")),
            "train_mean_include_true": float(_safe_mean(debug_train_conftr, "mean_include_true")),
            "val_mean_include_true": float(_safe_mean(debug_val_conftr, "mean_include_true")),
            "train_mean_include_others": float(_safe_mean(debug_train_conftr, "mean_include_others")),
            "val_mean_include_others": float(_safe_mean(debug_val_conftr, "mean_include_others")),
            "cover_weight": float(model.cover_weight),
            "other_weight": float(model.other_weight),
            "kappa": float(model.kappa),
            "external_coverage_gap": float(callback_covgap),
        }

        conftr_debug_keys = [
            "conftr_base_loss",
            "conftr_total_loss",
            "cer",
            "tau_align_mean_abs",
            "mean_set_size_pred",
            "mean_include_true",
            "mean_include_others",
            "mean_L_class",
            "mean_Omega",
            "B_cal",
            "B_pred",
            "tau_mean",
            "tau_min",
            "tau_max",
            "true_p_mean",
            "true_p_min",
            "true_p_uniform_ks",
            "true_p_superuniform_violation",
            "true_p_n",
        ]
        conftr_debug_keys.extend(
            f"true_p_coverage_at_{_alpha_metric_suffix(alpha)}"
            for alpha in TRUE_LABEL_PVALUE_ALPHAS
        )
        _add_bucket_means(entry, "train_conftr", debug_train_conftr, conftr_debug_keys)
        _add_bucket_means(entry, "val_conftr", debug_val_conftr, conftr_debug_keys)
       
        for ct_name in cell_types:
            entry[f"coverage_{ct_name}"] = float(per_class_cov.get(ct_name, float('nan')))
        model.history.append(entry)

        val_true_incl = entry["val_mean_include_true"]
        if not np.isfinite(val_true_incl):
            val_true_incl = entry["train_mean_include_true"]
        if epoch > epochs_CG_start and np.isfinite(val_true_incl):
            if val_true_incl > best_val_true_incl + 1e-4:
                best_val_true_incl = val_true_incl
                no_improve_val_true_incl = 0
            else:
                no_improve_val_true_incl += 1

            #if no_improve_val_true_incl >= true_incl_patience:
                #model.cover_weight += 1
            #    no_improve_val_true_incl = 0
            #    _vprint(
            #        verbose,
            #        2,
            #        f"True inclusion stalled for {true_incl_patience} epochs; "
            #        f"increasing cover_weight to {model.cover_weight:.2f}"
            #    )

        monitor_cov_stats = val_cov_stats
        monitor_debug = debug_val_conftr
        monitor_source = "validation"
        if not np.isfinite(monitor_cov_stats["coverage_shortfall"]):
            monitor_cov_stats = train_cov_stats
            monitor_debug = debug_train_conftr
            monitor_source = "training"

        monitor_abs_gap = monitor_cov_stats["coverage_gap_monitor"]
        monitor_shortfall = monitor_cov_stats["coverage_shortfall"]
        monitor_avg_set_size = _safe_mean(monitor_debug, "mean_set_size_pred")

        if np.isfinite(monitor_abs_gap) and monitor_abs_gap < best_monitor_abs_gap:
            _vprint(
                verbose,
                2,
                f"New best {monitor_source} coverage gap found: {monitor_abs_gap:.4f} "
                f"(previous: {best_monitor_abs_gap:.4f})"
            )
            best_monitor_abs_gap = monitor_abs_gap

        val_cer = _safe_mean(debug_val_conftr, "cer")
        if not np.isfinite(val_cer):
            val_cer = _safe_mean(debug_train_conftr, "cer")
        if epoch > epochs_CG_start and np.isfinite(val_cer):
            if val_cer < best_val_cer - 1e-4:
                best_val_cer = val_cer
                no_improve_val_cer = 0
            else:
                no_improve_val_cer += 1

            if no_improve_val_cer >= cer_patience:
                model.gamma_tau_align += 0
                no_improve_val_cer = 0
                _vprint(
                    verbose,
                    2,
                    f"CER stalled for {cer_patience} epochs; "
                    f"gamma_tau_align remains {model.gamma_tau_align:.2f}"
                )

        val_conftr_total_loss = _safe_mean(debug_val_conftr, "conftr_total_loss")
        if not np.isfinite(val_conftr_total_loss):
            val_conftr_total_loss = _safe_mean(debug_train_conftr, "conftr_total_loss")
        if epoch > epochs_CG_start and np.isfinite(val_conftr_total_loss):
            if val_conftr_total_loss < best_val_conftr_total_loss - 1e-4:
                best_val_conftr_total_loss = val_conftr_total_loss
                no_improve_val_conftr_total_loss = 0
            else:
                no_improve_val_conftr_total_loss += 1

            if no_improve_val_conftr_total_loss >= cfg.early_stopping_patience:
                if can_stop_after_temp_stable:
                    _vprint(
                        verbose,
                        1,
                        f"ConfTr total loss did not improve for "
                        f"{cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1}."
                    )
                    break

        true_p_superuniform_violation = _safe_stat(debug_val_conftr, "true_p_superuniform_violation")
        true_p_superuniform_source = "validation"
        if not np.isfinite(true_p_superuniform_violation):
            true_p_superuniform_violation = _safe_stat(debug_train_conftr, "true_p_superuniform_violation")
            true_p_superuniform_source = "training"
        if epoch > epochs_CG_start and np.isfinite(true_p_superuniform_violation):
            min_delta = 0.0001 * abs(best_val_true_p_superuniform_violation)
            if (
                not np.isfinite(best_val_true_p_superuniform_violation)
                or true_p_superuniform_violation < best_val_true_p_superuniform_violation - min_delta
            ):
                best_val_true_p_superuniform_violation = true_p_superuniform_violation
                no_improve_val_true_p_superuniform_violation = 0
            else:
                no_improve_val_true_p_superuniform_violation += 1

            if (
                epoch > epochs_CG_end
                and no_improve_val_true_p_superuniform_violation >= cfg.early_stopping_patience
            ):
                if can_stop_after_temp_stable:
                    _vprint(
                        verbose,
                        1,
                        f"{true_p_superuniform_source.title()} true_p_superuniform_violation did not improve "
                        f"by at least 0.01% for {cfg.early_stopping_patience} epochs, "
                        f"stopping at epoch {epoch + 1}."
                    )
                    break

        coverage_target_reached = (
            np.isfinite(monitor_shortfall)
            and monitor_shortfall <= 0.01
            and epoch > epochs_CG_start
        )

        if coverage_target_reached or (start_best_val_metric_flag == True):
            
            start_best_val_metric_flag = True

            if Flag_lr_scheduler_step == False:
                _vprint(verbose, 1, f"{monitor_source.title()} coverage target nearly reached, reducing LR at epoch {epoch + 1}")
                opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] * 0.8
                Flag_lr_scheduler_step = True
                if monitor_source == "validation":
                    allow_refinement_stop_before_temp_target = True

            refinement_metric = monitor_avg_set_size if np.isfinite(monitor_avg_set_size) else monitor_abs_gap
            if np.isfinite(refinement_metric) and refinement_metric < best_refinement_metric - 1e-4:
                best_refinement_metric = refinement_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve_cov_overall = 0
                _vprint(
                    verbose,
                    2,
                    f"\nSaved best model state based on {monitor_source} refinement metric: "
                    f"{best_refinement_metric:.4f} "
                    f"(shortfall={monitor_shortfall:.4f}, avg_set_size={monitor_avg_set_size:.3f})\n"
                )
            else:
                no_improve_cov_overall += 1
            
            _vprint(verbose, 2, f"no_improve_refinement: {no_improve_cov_overall}")

            if no_improve_cov_overall >= cfg.early_stopping_patience:
                if can_stop_after_temp_stable or allow_refinement_stop_before_temp_target:
                    _vprint(verbose, 1, f"{monitor_source.title()} refinement metric did not improve for {cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1}.")
                    break
            



                

            
            
            
            

                




    if best_state is not None:
        model.load_state_dict(best_state)
        _vprint(verbose, 1, "Loaded best model state based on callback coverage metric.")

    if epoch_callback is not None:
        epoch_callback(epoch + 1, model,True)    
                
    



class DataPreparation():

    def data_loader_prep(
        self,
        obs_df: pd.DataFrame,
        batch_key: str,
        cell_type_col: str,
        reference_dictionary: dict,
        seed: int = 0):
    
        verbose = getattr(self, "verbose", 2)
        cell_types = obs_df[cell_type_col].astype(str).unique().tolist()
        reference_batch_default =[]
        batches = []
        for key, val in reference_dictionary.items():
            reference_batch_default.append(key)
            batches.extend(val)
            batches.append(key)

        batches = sorted(obs_df[batch_key].unique())


        batch_to_idx = {batch: idx for idx, batch in enumerate(batches)}
        cell_type_to_idx = {ct: idx for idx, ct in enumerate(cell_types)}
        self.ref_to_targets_idx = {
            batch_to_idx[ref]: [batch_to_idx[t] for t in targets]
            for ref, targets in reference_dictionary.items()}


        _vprint(verbose, 2, "\nBatch to index mapping:", batch_to_idx)
        _vprint(verbose, 2, "Cell type to index mapping:", cell_type_to_idx)
        _vprint(verbose, 2, "Reference to targets index mapping:", self.ref_to_targets_idx)

        batch_codes = obs_df[batch_key].map(batch_to_idx).to_numpy()
        cell_type_codes = obs_df[cell_type_col].map(cell_type_to_idx).to_numpy()

        self.b_tensor = torch.from_numpy(batch_codes).long()
        self.ct_tensor = torch.from_numpy(cell_type_codes).long()

        self.reference_batch_idx_list = []
        for ref in reference_batch_default:

            reference_batch_idx = batch_to_idx[ref]
            self.reference_batch_idx_list.append(reference_batch_idx)

            _vprint(verbose, 2, f"Reference batch '{ref}' has index {reference_batch_idx}")



    def build_data_tensors(
        self,
        counts_array: np.ndarray,
        seed_offset: int = 0,
        val_fraction: float = 0.2,
        batch_size: int = 256,
        calibration_fraction: float = 0.3,
    ):  
        
        verbose = getattr(self, "verbose", 2)
        counts_tensor = torch.from_numpy(counts_array).float()
        if counts_tensor.shape[0] != self.b_tensor.shape[0] or counts_tensor.shape[0] != self.ct_tensor.shape[0]:
            raise ValueError("Counts tensor must align with batch and cell type annotations")
        dataset = TensorDataset(counts_tensor, self.b_tensor, self.ct_tensor)

        rng_local = np.random.default_rng(124 + 10 + seed_offset)
        all_indices = np.arange(len(dataset))

        if calibration_fraction > 0.0:
            cal_indices_list = []
            cal_indices = np.asarray([], dtype=np.int64)
            for reference_batch_idx in self.reference_batch_idx_list:

                ref_mask = (self.b_tensor.cpu().numpy() == reference_batch_idx)
                ref_all = np.where(ref_mask)[0]
                cal_size = int(np.floor(calibration_fraction * len(ref_all)))

                if cal_size > 0:
                    ref_ct = self.ct_tensor[ref_all].cpu().numpy()
                    unique_ct, inverse = np.unique(ref_ct, return_inverse=True)

                    cal_chunks = []
                    leftovers = []
                    for ct_idx, ct_value in enumerate(unique_ct):
                        group_indices = ref_all[inverse == ct_idx]
                        target = calibration_fraction * group_indices.size
                        take = int(np.floor(target))
                        if take > 0:
                            cal_chunks.append(
                                rng_local.choice(group_indices, size=take, replace=False)
                            )
                        leftovers.append((target - take, group_indices))

                    cal_indices = np.concatenate(cal_chunks) if cal_chunks else np.asarray([], dtype=np.int64)

                    need_extra = cal_size - cal_indices.size
                    if need_extra > 0:
                        leftovers.sort(key=lambda t: t[0], reverse=True)
                        extras = []
                        for _, group_indices in leftovers:
                            if need_extra <= 0:
                                break
                            available = np.setdiff1d(group_indices, cal_indices, assume_unique=False)
                            if available.size > 0:
                                grab = min(need_extra, available.size)
                                extras.append(rng_local.choice(available, size=grab, replace=False))
                                need_extra -= grab
                        if extras:
                            cal_indices = np.concatenate((cal_indices, *extras))
                else:
                    cal_indices = np.asarray([], dtype=np.int64)

                cal_indices_list.append(cal_indices)
        else:
            cal_indices = np.asarray([], dtype=np.int64)

        cal_index_flatten = np.concatenate(cal_indices_list) if calibration_fraction > 0.0 else np.asarray([], dtype=np.int64)
        
        if cal_index_flatten.size > 0:
            pool_indices = np.setdiff1d(all_indices, cal_index_flatten, assume_unique=False)
        else:
            pool_indices = all_indices

        if val_fraction > 0.0:
            n_val = int(np.floor(len(pool_indices) * val_fraction))
            if n_val > 0:
                val_sel = rng_local.choice(pool_indices, size=n_val, replace=False)
                train_sel = np.setdiff1d(pool_indices, val_sel, assume_unique=False)
                val_ds = Subset(dataset, val_sel.tolist())
            else:
                train_sel = pool_indices
                val_ds = None
        else:
            train_sel = pool_indices
            val_ds = None

        train_ds = Subset(dataset, train_sel.tolist())

        try:
            num_workers = getattr(self, "num_workers", 0)
        except AttributeError:
            num_workers = 0
        try:
            pin_memory = getattr(self, "pin_memory", False)
        except AttributeError:
            pin_memory = False

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )
        else:
            val_loader = None

        _vprint(verbose, 2, f"Train dataset length: {len(train_ds)}")
        unique_labels, counts = torch.unique(dataset.tensors[2][train_ds.indices], return_counts=True)
        label_counts_pytorch = dict(zip(unique_labels.tolist(), counts.tolist()))
        _vprint(verbose, 2, label_counts_pytorch)

        _vprint(verbose, 2, f"\nValidation dataset length: {len(val_ds) if val_ds is not None else 0}")
        unique_labels, counts = torch.unique(dataset.tensors[2][val_ds.indices], return_counts=True)
        label_counts_pytorch = dict(zip(unique_labels.tolist(), counts.tolist()))
        _vprint(verbose, 2, label_counts_pytorch)
        
        for idx_cal in cal_indices_list:
            _vprint(verbose, 2, f"\nCalibration set size: {idx_cal.size}")
            unique_labels, counts = torch.unique(dataset.tensors[2][idx_cal], return_counts=True)
            label_counts_pytorch = dict(zip(unique_labels.tolist(), counts.tolist()))
            _vprint(verbose, 2, label_counts_pytorch)
        
        return counts_tensor, train_loader, val_loader, cal_indices_list


def evaluate_latent_logreg(
    latent_np: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_labels: np.ndarray,
    cell_labels: np.ndarray,
    ref_to_targets_idx: dict[int, list[int]],
    cal_indices: np.ndarray,
    target_coverage: List[float] = [0.90],
    verbose: int | bool = 2,
) -> None:

    verbose = _normalize_verbose(verbose)
    latent_np = latent_np.astype(np.float32)

    if val_indices.size == 0:
        _vprint(verbose, 2, "Latent conformal evaluation skipped: validation set empty.")
        return

    alpha_list = np.array([1.0 - target_coverage_ for target_coverage_ in target_coverage]) 

    def run_scenario(
        clf: LogisticRegression,
        train_subset: np.ndarray,
        eval_subset: np.ndarray,
        cal_indices: np.ndarray,
        target_coverage: List[float],
    ) -> None:
        if eval_subset.size == 0 or train_subset.size == 0:
            _vprint(verbose, 2, "skipped: insufficient samples.")
            return

        clf.fit(latent_np[train_subset], cell_labels[train_subset])

        cal_probs = clf.predict_proba(latent_np[cal_indices])
        cal_true_cols = np.searchsorted(clf.classes_, cell_labels[cal_indices])
        cal_scores = 1.0 - cal_probs[np.arange(cal_indices.size), cal_true_cols]

        eval_probs = clf.predict_proba(latent_np[eval_subset])
        eval_true_cols = np.searchsorted(clf.classes_, cell_labels[eval_subset])

        cal_labels = cell_labels[cal_indices]
        idx_in_classes = np.searchsorted(clf.classes_, cal_labels)
        if (idx_in_classes >= len(clf.classes_)).any() or (clf.classes_[idx_in_classes] != cal_labels).any():
            raise ValueError("IndexError in evaluating conformal sets. Check class labels and predictions. Probably some cell-type missing not in the reference batch.")
           
            
        n_cal = cal_scores.size
        if n_cal == 0:
            q_hats = np.ones_like(alpha_list)
        else:
            cal_scores_ordered = np.sort(cal_scores)
            ranks_float = np.ceil((n_cal + 1) * (1 - alpha_list)) - 1
            ranks_int = np.clip(ranks_float.astype(int), 0, n_cal - 1)
            q_hats = cal_scores_ordered[ranks_int]

        thresholds = 1.0 - q_hats

        mask = eval_probs[None, :, :] >= thresholds[:, None, None]
        set_sizes_all_alphas = mask.sum(axis=2)
        size_list = set_sizes_all_alphas.mean(axis=1) 

        eval_true_probs = eval_probs[np.arange(eval_subset.size), eval_true_cols]
        contains_true_all_alphas = eval_true_probs[None, :] >= thresholds[:, None]
        cov_list = contains_true_all_alphas.mean(axis=1)

        for coverage, avg_size, target_coverage_ in zip(cov_list, size_list, target_coverage):

            _vprint(
                verbose,
                2,
                f"coverage: {coverage:.4f}, (target coverage {target_coverage_:.0%}. average set size: {avg_size:.2f})\n\n"
            )

        covgap = np.mean([t-c for c, t in zip(cov_list, target_coverage)])
        

        return covgap

    for idx,(reference_batch_idx, target_batch_idx_list) in enumerate(ref_to_targets_idx.items()):

        train_mask = batch_labels[train_indices] == reference_batch_idx
        eval_mask = np.isin(batch_labels[val_indices], target_batch_idx_list)
       
        cal_indices_aux = cal_indices[idx]
    

        if train_mask.any() and eval_mask.any():
            covgap = run_scenario(
                                    LogisticRegression(max_iter=10000),
                                    train_indices[train_mask],
                                    val_indices[eval_mask],
                                    cal_indices_aux,
                                    target_coverage
                                )
        else:
            _vprint(verbose, 2, "Latent conformal (train ref -> test other) skipped: insufficient samples.")

    return covgap



class Integrator(DataPreparation):

    def __init__(self,  
                seed_offset: int = 0,
                epochs: int = 3000,
                lr: float = 5e-4,
                kl_anneal_epochs: int = 1,
                batch_size: int = 64,
                calibration_fraction: float = 0.3,
                val_fraction: float = 0.2,
                device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                num_workers: Optional[int] = None,
                pin_memory: Optional[bool] = None,
                verbose: int | bool = 0,):

        self.verbose = _normalize_verbose(verbose)
        self.seed_offset = seed_offset
        self.epochs = epochs
        self.lr = lr
        self.kl_anneal_epochs = kl_anneal_epochs
        self.batch_size = batch_size
        self.calibration_fraction = calibration_fraction
        self.val_fraction = val_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if num_workers is None:
            try:
                self.num_workers = 2
            except Exception:
                self.num_workers = 0
        else:
            self.num_workers = max(0, num_workers)
        
        if pin_memory is None:
            self.pin_memory = bool(self.device.type == "cuda")
        else:
            self.pin_memory = bool(pin_memory)

        if self.num_workers > 0:
            _vprint(self.verbose, 2, f"Warning: Num. workers for data loading set to {self.num_workers}. If Jax installed, consider setting to 0 for security.")

    def train_cvae_on_counts(
        self,
        counts_array: np.ndarray,
        df_obs: pd.DataFrame,
        batch_key: str,
        cell_type_col: str,
        cell_types: list,
        ref_batch: dict,
        verbose: int | bool = 2,
        beta : float = None,
        epochs_CG_start : int = None,
        conf_T_init : float = None,
        conf_T_max_decay : float = None,
        lambda_size : float = None,
        gamma_tau_align : float = None,
    ):
        verbosity = _normalize_verbose(verbose)
        self.verbose = verbosity
        globals()["batch_key"] = batch_key
        globals()["cell_type_col"] = cell_type_col
        globals()["obs_df"] = df_obs
        globals()["seed"] = self.seed_offset
        reset_umap_basis()

        if sparse.issparse(counts_array):
            counts_array = counts_array.toarray()
        else:
            counts_array = np.asarray(counts_array)

        counts_array = counts_array.astype(np.float32, copy=False)

        self.data_loader_prep(
            obs_df=df_obs,
            batch_key=batch_key,
            cell_type_col=cell_type_col,
            reference_dictionary=ref_batch
        )
           
        _vprint(verbosity, 1, f"\nInitial batch size: {self.batch_size}")
        _vprint(verbosity, 1, "\nBuilding data tensors and adjusting batch size...")
        for ref_idx, target_idxs in self.ref_to_targets_idx.items():
            cal_samples_av = 0.0
            while cal_samples_av <= 90:   
                
                counts_tensor, train_loader, val_loader, cal_indices = self.build_data_tensors(
                    counts_array,
                    seed_offset=self.seed_offset,
                    batch_size=self.batch_size,
                    calibration_fraction=self.calibration_fraction,
                    val_fraction=self.val_fraction
                )
                _vprint(verbosity, 2, f"Evaluating average calibration samples for reference batch {ref_idx}...")
                cal_samples = []
                for _, bb, _ in train_loader:
                    mask = (bb == ref_idx)
                    
                    for t in target_idxs:
                        mask |= (bb == t)
                    
                    if not mask.any():
                        continue   
                    
                    b_s = bb[mask]
                    
                    cal_mask_s = (b_s == ref_idx)

                    all_idx = torch.arange(b_s.size(0), device=bb.device)
                
                    pool_idx = all_idx[cal_mask_s]
                        
                    cal_samples.append(pool_idx.numel())
                _vprint(verbosity, 2, cal_samples)
                cal_samples_av = np.sum(cal_samples)/len(cal_samples)  

                _vprint(verbosity, 1, f"Current batch size: {self.batch_size}, average cal samples for ref batch {ref_idx}: {cal_samples_av}")
                if cal_samples_av <= 90:
                    self.batch_size = int(self.batch_size * 2)

        _vprint(verbosity, 1, f"\nAdjusted batch size: {self.batch_size}\n")

        _vprint(verbosity, 2, "cal indices length:", len(cal_indices))
        _vprint(verbosity, 2, self.reference_batch_idx_list)
        _vprint(verbosity, 2, self.ref_to_targets_idx)

        
        batches = sorted(df_obs[batch_key].unique().tolist())
        
        cfg = VAEConfig(
            input_dim=counts_array.shape[1],
            latent_dim=vae_latent_dim,
            num_batches=len(batches),
            num_cell_types=len(cell_types),
            hidden=vae_hidden_layers,
            dropout=0.1,
        )

        

        model = ConditionalVAE(cfg).to(self.device)

        train_cfg = TrainConfig(
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            kl_anneal_epochs=self.kl_anneal_epochs,
        )



        _vprint(verbosity, 1, "\nTraining CVAE...")
        train_idx_np = np.asarray(train_loader.dataset.indices, dtype=int)
        val_idx_np = np.asarray(val_loader.dataset.indices, dtype=int) if val_loader is not None else None
        b_array_np = self.b_tensor.cpu().numpy()
        ct_array_np = self.ct_tensor.cpu().numpy()

        def latent_progress_callback(epoch_idx: int, model_snapshot: ConditionalVAE, final:bool) -> None:
            
            model_snapshot.eval()
            with torch.inference_mode():
                mu_epoch, _ = model_snapshot.encode(
                    counts_tensor,
                    self.b_tensor,
                    self.ct_tensor,
                )
            latent_epoch_np = mu_epoch.cpu().numpy()

            if verbosity >= 2:
                save_latent_plot(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}.png")

            covgap = float("nan")
            if val_loader is not None:

                target_coverage = [0.90]
                covgap = evaluate_latent_logreg(
                            latent_epoch_np,
                            train_idx_np,
                            val_idx_np,
                            b_array_np,
                            ct_array_np,
                            self.ref_to_targets_idx,
                            cal_indices,
                            target_coverage,
                            verbose=verbosity)  
            
            if int(epoch_idx) == 20 and verbosity >= 2:
                save_latent_umap(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}_UMAP.png")

            if int(epoch_idx) % 40 == 1 and verbosity >= 2:
                save_latent_umap(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}_UMAP.png")

            if final:
                if verbosity >= 2:
                    save_latent_umap(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}_UMAP.png")

                if val_loader is not None:

                    target_coverage = [0.90]
                    evaluate_latent_logreg(
                        latent_epoch_np,
                        train_idx_np,
                        val_idx_np,
                        b_array_np,
                        ct_array_np,
                        self.ref_to_targets_idx,
                        cal_indices,
                        target_coverage,
                        verbose=verbosity)
                    

            model_snapshot.train()


            return covgap

            
        self.b_tensor = self.b_tensor.to(self.device, non_blocking=True)
        self.ct_tensor = self.ct_tensor.to(self.device, non_blocking=True)
        counts_tensor = counts_tensor.to(self.device, non_blocking=True)

        train_cvae(
            model,
            train_loader,
            val_loader,
            train_cfg,
            reference_batch_dict=self.ref_to_targets_idx,
            cal_indices=cal_indices,
            cell_types=cell_types,
            epoch_callback=latent_progress_callback,
            device=self.device,
            beta = beta,
            epochs_CG_start  = epochs_CG_start,
            conf_T_init  = conf_T_init,
            conf_T_max_decay = conf_T_max_decay,
            lambda_size  = lambda_size,
            gamma_tau_align  = gamma_tau_align,
            verbose=verbosity,
        )

        plot_training_curves(model, plots_dir=plots_dir, verbose=verbosity)

        return model, counts_tensor
