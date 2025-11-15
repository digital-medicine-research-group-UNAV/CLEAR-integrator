import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchsort  # optional, for differentiable sorting
except Exception:
    torchsort = None

from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc
from scipy import sparse
import os
import shutil

import umap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.linear_model import LogisticRegression 


try:
    import seaborn as sns
    sns.set_context("talk")
except ModuleNotFoundError:
    sns = None

plt.style.use("seaborn-v0_8")

plots_dir = None
"""    """
plots_dir = "plots"
if os.path.exists(plots_dir):
    for filename in os.listdir(plots_dir):
        file_path = os.path.join(plots_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

else:
    os.makedirs(plots_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#seed = 123
#n_genes = 4000
#adata_path = "../data/diabetic-kidney-disease_processed"
#adata_read_path = adata_path + ".h5ad"
#adata_save_path = adata_path + "_integrated_crTr.h5ad"
#adata = sc.read_h5ad(adata_read_path) # sc.read_h5ad("../data/diabetic_processed.h5ad") #sc.read_h5ad("../data/MB_processed.h5ad") # sc.read_h5ad("../data/adata_tutorial.h5ad")

#cell_type_col = "cell_type" #"cell_type" # "cell_type_eval"
#use_layer = "counts"
#batch_key = "assay" #"batch" # "batch" # "system"

#reference_dictionary = {"healthy_6": ['healthy_5', 'healthy_4'],"control_3": ["control_1", "control_2"], 'diabetic_1': ['diabetic_2', 'diabetic_3','diabetic_4', 'diabetic_5']}


#reference_dictionary = {"healthy_6": ["control_1", "control_2", "control_3", "healthy_4", "healthy_5", 'diabetic_2', 'diabetic_3','diabetic_4', 'diabetic_5']}
#reference_dictionary = {"10x 5' v1": ["10x 3' v3"]}


#reference_batch_default =[]
#batches = []
#for key, val in reference_dictionary.items():
#    reference_batch_default.append(key)
#    batches.extend(val)
#    batches.append(key)


#reference_batch_default = "healthy_6" #"snATAC"  # "snATAC" "10X" #"1"
#batches = ["healthy_6", "control_3", "diabetic_1" ]# ["10X", "snATAC"] #["10X", "snATAC"] # ["0", "1"]



vae_latent_dim = 16
vae_hidden_layers = (564, 164)
umap_basis_model = None  # will hold the fitted UMAP for consistent coordinates

#train_weight_decay_default = 5e-4
#train_grad_clip_default = 1.0
#train_lr_gamma_default = 0.5
#train_lr_patience_default = 4
#train_min_lr_default = 4
#train_early_stopping_patience_default = 15



#EPS = 1e-8

#seed = 123
#np.random.seed(seed)
#torch.manual_seed(seed)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Running on {device} (GPU available: {torch.cuda.is_available()})")




#plots_dir = None




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

    # (Re)fit the basis if not present or if latent dim changed
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


def plot_training_curves(model, plots_dir=plots_dir, highlight_epoch: int = 21):
    """
    Genera:
      - training_curves_losses.png: SOLO total train/val, eje Y log, vline en highlight_epoch
      - training_curves_coverage.png: cobertura (global y por clase) + y=0.9 + vline en highlight_epoch
      - training_history.csv: histórico completo
    """
    import pandas as pd
    import numpy as np
    import os, matplotlib.pyplot as plt

    if not hasattr(model, "history") or len(model.history) == 0:
        print("No hay histórico en model.history; ¿ejecutaste el entrenamiento?")
        return

    hist = pd.DataFrame(model.history).sort_values("epoch").reset_index(drop=True)
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(plots_dir, "training_history.csv")
    hist.to_csv(csv_path, index=False)

    # ---------- Pérdidas: solo total train/val, eje Y log ----------
    fig, ax = plt.subplots(figsize=(8, 5))

    eps = 1e-8
    y_train = np.maximum(hist["train_total_loss"].to_numpy(dtype=float), eps)
    ax.plot(hist["epoch"], y_train, label="Train total loss", linewidth=2)  # línea continua

    if "val_total_loss" in hist.columns:
        y_val = hist["val_total_loss"].to_numpy(dtype=float)
        if np.isfinite(y_val).any():
            y_val = np.maximum(np.where(np.isfinite(y_val), y_val, np.nan), eps)
            ax.plot(hist["epoch"], y_val, linestyle="--", label="Val total loss", linewidth=2)  # discontinua

    # Línea vertical del cambio de loss
    ax.axvline(highlight_epoch, color="red", linestyle=":", linewidth=1.8, label=f"Nueva loss (época {highlight_epoch})")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Train vs. Validation (Total Loss: ELBO + ConfTr)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_curves_losses.png"), dpi=150)
    plt.close(fig)

    # ---------- Cobertura: overall y por clase + línea y=0.9 ----------
    fig, ax = plt.subplots(figsize=(8, 5))

    # Línea objetivo en 0.9 (negra discontinua)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="Objetivo 0.90")

    # Curva global
    if "coverage_overall" in hist.columns:
        ax.plot(hist["epoch"], hist["coverage_overall"], marker="o", linewidth=2, label="Coverage overall")

    # Curvas por clase (si existen)
    cov_cols = [c for c in hist.columns if c.startswith("coverage_") and c != "coverage_overall"]
    for c in sorted(cov_cols):
        ax.plot(hist["epoch"], hist[c], alpha=0.6, label=c.replace("coverage_", "cov_"))

    # Línea vertical del cambio de loss
    ax.axvline(highlight_epoch, color="red", linestyle=":", linewidth=1.8, label=f"Nueva loss (época {highlight_epoch})")

    ax.set_xlabel("Epochs"); ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1)
    ax.set_title("Coverage over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_curves_coverage.png"), dpi=150)
    plt.close(fig)

    print(f"Histórico guardado en {csv_path}")





# ---------------- Conformal Training (THR variant) ----------------

from typing import Optional

# robust SoftSort wrapper supporting torchsort.soft_sort and torchsort.softsort
def _softsort_1d(values: torch.Tensor, tau: float, ascending: bool = True):
    if 'torchsort' in globals() and torchsort is not None:
        if hasattr(torchsort, "soft_sort"):  # newer API
            return torchsort.soft_sort(values, regularization_strength=tau, descending=not ascending)
        if hasattr(torchsort, "softsort"):   # older API
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
        return torch.lerp(s[low], s[high], torch.tensor(w, device=values.device, dtype=s.dtype)) #linear interpolation between low and high 
    else:
        # STE fallback
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
    e_true = scores.gather(1, y.view(-1, 1)).squeeze(1)  # (B_cal,)
    n = max(1, e_true.numel())
    q = alpha * (1.0 + 1.0 / n)  # CP correction
    tau = _smooth_quantile_1d(e_true, q=q, tau=eps_sort)
    return tau  # scalar


def _smooth_pred_thr(scores: torch.Tensor, tau: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Soft confidence-set membership for THR: sigma((E - tau)/T)."""
    T = max(1e-6, float(temperature))
    return torch.sigmoid((scores - tau) / T)



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
    # Global fallback if a class has no calibration samples:
    tau_global = _smooth_calibrate_thr(scores, y, alpha=alpha, eps_sort=eps_sort)

    taus = []
    for k in range(K):
        mask_k = (y == k)
        if mask_k.any():
            e_k = scores[mask_k, k].reshape(-1) # NCscores for class k
            n_k = max(1, e_k.numel()) # number of cal samples in class k
            q_k = alpha * (1.0 + 1.0 / n_k)  # CP finite-sample correction
            tau_k = _smooth_quantile_1d(e_k, q=q_k, tau=eps_sort)
        else:
            tau_k = tau_global  # safe fallback
        taus.append(tau_k)
    return torch.stack(taus)

def _smooth_pred_thr_mondrian(scores: torch.Tensor,
                              tau_vec: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
    """Soft membership with per-class thresholds: sigma((scores - tau_k)/T)."""
    T = max(1e-6, float(temperature))
    return torch.sigmoid((scores - tau_vec.view(1, -1)) / T)


# ---- Per-batch Mondrian taus (class-conditional quantiles) ------------------
def _smooth_calibrate_thr_mondrian_by_group(
    scores: torch.Tensor,   # (N,K)
    y: torch.Tensor,        # (N,)
    g: torch.Tensor,        # (N,) group ids, e.g., batch in [0..G-1]
    K: int,
    alpha: float,
    eps_sort: float = 1e-2,
    min_n: int = 8,         # need a few samples to trust a group τ_k
):
    """
    Returns:
      taus_gk: (G,K) per-group per-class tau_k^g
      mask_gk: (G,K) bool -> True if class k had >= min_n samples in group g
    """
    G = int(g.max().item()) + 1
    taus = torch.zeros(G, K, device=scores.device)
    mask = torch.zeros(G, K, dtype=torch.bool, device=scores.device)

    # global fallback τ_k (using all groups)
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

# ---------------------------------------------------------------------------

    


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
        pool_idx = all_idx[cal_mask]   # cantidad de referencias en el minibatch que serán usados como calibración
        if pool_idx.numel() == 0:
            return torch.tensor([], device=device, dtype=torch.long), all_idx
            #pool_idx = all_idx  # fallback: no reference in minibatch

    cap_frac: float = 0.5   # fracción máxima de B_cal respecto al numero de muestras de calibración en el minibatch
    max_cal = max(1, int(cap_frac * pool_idx.numel()))

    if num_classes is None:
        K = int(y.max().item()) + 1
    else:
        K = num_classes

    # If no per-class requirement, just take up to cap from pool
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

            take_k = min(int(k_idx.numel()*0.5), min_per_class)  # num of samples to take from class k for calibration

            perm = torch.randperm(k_idx.numel(), generator=g, device=device)[:take_k]
            parts.append(k_idx[perm])

        if len(parts) == 0:
                                                    # fallback: random from pool in case of no class representation
            take = min(max_cal, pool_idx.numel())
            cal_idx = pool_idx[torch.randperm(pool_idx.numel(), generator=g, device=device)[:take]]

        else:
            cal_idx = torch.unique(torch.cat(parts)) # concatenate and unique
            # ensure minimum total
            if cal_idx.numel() < min_total:
                remaining = pool_idx[~torch.isin(pool_idx, cal_idx)] # remaining candidates
                need = min_total - cal_idx.numel()
                need = min(need, remaining.numel())
                if need > 0:
                    extra = remaining[torch.randperm(remaining.numel(), generator=g, device=device)[:need]] # random selection if neccesary
                    cal_idx = torch.unique(torch.cat([cal_idx, extra]))

            # cap to max_cal
            if cal_idx.numel() > max_cal:
                perm = torch.randperm(cal_idx.numel(), generator=g, device=device)[:max_cal]
                cal_idx = cal_idx[perm]

    pred_mask = torch.ones(N, dtype=torch.bool, device=device)
    pred_mask[cal_idx] = False
    pred_idx = all_idx[pred_mask] #idx not in calibration set

    # guard rails
    if pred_idx.numel() == 0:
        pred_idx = all_idx #cal_idx

    #if cal_idx.numel() == 0:
    #    cal_idx = pred_idx

    return cal_idx, pred_idx


def conftr_loss_thr_balanced(
    scores: torch.Tensor, y: torch.Tensor, alpha: float,
    *,
    cal_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    eps_sort: float = 1e-2,
    kappa: float = 1.0,
    lambda_size: float = 1.0,
    cover_weight: float = 1.0,
    other_weight: float = 0.5,
    min_cal_per_class: int = 0,
    min_cal_total: int = 16,
    num_classes: Optional[int] = None,
    rng: Optional[torch.Generator] = None
):
    """
    Conformal training (THR) with *balanced* B_cal construction.
    Pass `scores` as **log-probs** (recommended) or probabilities; keep it consistent.
    """
    device = scores.device
    cal_idx, pred_idx = _balanced_indices_for_calibration(
        y, cal_mask, min_per_class=min_cal_per_class, min_total=min_cal_total, num_classes=num_classes, rng=rng
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
            "cer": 0.0
        }
        return zero, diag
    

    # SmoothCal on B_cal
    tau = _smooth_calibrate_thr(scores[cal_idx], y[cal_idx], alpha=alpha, eps_sort=eps_sort)

    # SmoothPred on B_pred
    T = max(1e-6, float(temperature))
    C = torch.sigmoid((scores[pred_idx] - tau) / T)  # (B_pred, K)

    rows = torch.arange(pred_idx.numel(), device=device)
    include_true = C[rows, y[pred_idx]]
    include_others = (C.sum(dim=1) - include_true)

    L_class = cover_weight * (1.0 - include_true) + other_weight * include_others
    Omega = torch.relu(C.sum(dim=1) - float(kappa))
    inside = L_class + float(lambda_size) * Omega
    loss = torch.log(1e-6 + inside.mean())
    

    diag = {
        "tau": float(tau.detach().cpu()),
        "mean_set_size_pred": float(C.sum(dim=1).mean().detach().cpu()),
        "mean_L_class": float(L_class.mean().detach().cpu()),
        "mean_Omega": float(Omega.mean().detach().cpu()),
        "B_cal": int(cal_idx.numel()),
        "B_pred": int(pred_idx.numel()),
        "pred_idx": pred_idx.detach().cpu(),
        "cal_idx": cal_idx.detach().cpu(),
    }
    return loss, diag


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
    batch: Optional[torch.Tensor] = None,   # (N,) batch ids
    ref_batch_id: int = 0,
    gamma_tau_align: float = 0.6,  # weight for τ alignment (quantile matching)
    
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

    # Balanced split for calibration vs prediction inside the minibatch
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
            "cer": 0.0
        }
        return zero, diag

    # SmoothCal (Mondrian): per-class tau_k from the calibration subset
    tau_vec = _smooth_calibrate_thr_mondrian(
        scores[cal_idx], y[cal_idx], K=K, alpha=alpha, eps_sort=eps_sort
    )

    # SmoothPred with per-class thresholds (C is a matrix of soft memberships, shape (n_samples, n_classes)
    C = _smooth_pred_thr_mondrian(scores[pred_idx], tau_vec, temperature=temperature)

    # Class + size objectives 
    rows = torch.arange(pred_idx.numel(), device=device) # we create row indices for indexing
    include_true = C[rows, y[pred_idx]] # we extract the soft inclusion probabilities for the true classes
    include_others = (C.sum(dim=1) - include_true) # we compute the soft inclusion probabilities for the other classes (residual)

    L_class = cover_weight * (1.0 - include_true) + other_weight * include_others   # conformal loss 

    Omega = torch.relu(C.sum(dim=1) - float(kappa))                                 # size penalty beyond κ

    base_loss = torch.log(1e-6 + (L_class + float(lambda_size) * Omega).mean())    # Regularized conformal loss 

    # ---- Conformal Exchangeability Regularizer (CER) --------------------------
    cer = scores.new_tensor(0.0)
    info_align = {}

    if batch is not None and (gamma_tau_align > 0.0):

        # 1) τ-alignment across batches, per class
        taus_gk, mask_gk = _smooth_calibrate_thr_mondrian_by_group(
            scores=scores, y=y, g=batch, K=K, alpha=alpha, eps_sort=eps_sort)  # taus_gk: (G,K), mask_gk: (G,K)
        
        
        
        tau_ref = taus_gk[ref_batch_id]  # (K,)  # These are the class-wise thresholds for the reference batch

        if gamma_tau_align > 0.0:

            diff = torch.abs(taus_gk - tau_ref.unsqueeze(0))     # (G,K)
            
            valid = mask_gk.clone()

            if ref_batch_id < valid.size(0):
                valid[ref_batch_id, :] = False                   # don't compare ref to itself

            if valid.any():
                cer = cer + gamma_tau_align * diff[valid].mean()
                info_align["tau_align_mean_abs"] = float(diff[valid].mean().detach().cpu())
            else:
                info_align["tau_align_mean_abs"] = 0.0

    loss = base_loss + cer


    diag = {
        "tau_by_class": tau_vec.detach().cpu(),                     # (K,)
        "mean_set_size_pred": float(C.sum(dim=1).mean().detach().cpu()),
        "mean_L_class": float(L_class.mean().detach().cpu()),
        "mean_Omega": float(Omega.mean().detach().cpu()),
        "B_cal": int(cal_idx.numel()),
        "B_pred": int(pred_idx.numel()),
        "pred_idx": pred_idx.detach().cpu(),
        "cal_idx": cal_idx.detach().cpu(),
        "cer": float(cer.detach().cpu()),
        **info_align,
    }
    return loss, diag


# ---------------------------------------------------------------------------
# CVAE model
# ---------------------------------------------------------------------------


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

        # Conformal settings
        self.alpha = 0.1             # target miscoverage (alpha)
        
        # Epoch coverage accumulators (set/reset by trainer)
        self._cov_epoch_total = None
        self._cov_epoch_covered = None

        # Quick scalar logging
        self.current_batch_class_acc = 0.0 

        # ConfTr fixed parameters
        self.conf_eps = 1e-2                        # SmoothCal softness (scheduled) for differentiable quantile
        self.cover_weight: float = 1.0
        self.other_weight: float = 0.5
        self.min_cal_per_class: int = 10
        self.min_cal_total:int = 64
        self.num_classes = cfg.num_cell_types
        self.kappa: float = 1.5                    # size threshold for penalty (ligeramente > 1 para evitar 0-size sets)

        # ConfTr hyperparameters (to be tuned)
        self.conf_T: float = None                    # temperature (scheduled in training)
        self.lambda_size: float = None
        self.gamma_tau_align: float = None     


        
        # Classification head
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
        if self.classifier is not None and self.enable_classification:
            
            # Classifier on latent (use z to propagate gradients with ELBO; switch to mu_z if you prefer deterministic)
            logits = self.classifier(mu_z)#(z if self.training else z.detach())
            log_probs = F.log_softmax(logits, dim=1)

            per_task_losses = []
            per_task_sizes = []
            per_task_mean_set = []
            counted = torch.zeros_like(b, dtype=torch.bool) # to avoid double counting
            
            for ref_idx, target_idxs in reference_batch_dict.items():
                # mask for this task: only batches in {ref} ∪ targets
                mask = (b == ref_idx) # cal 
                
                for t in target_idxs:
                    mask |= (b == t)
                
                if not mask.any():
                    continue       
            
                scores_s = log_probs[mask]
                y_s = y[mask]
                b_s = b[mask]
                  
                # calibrate only on reference subset for this task
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
                    batch=b_s.long(),           # ok to use global ids
                    ref_batch_id=ref_idx,       # global ref id
                    gamma_tau_align=self.gamma_tau_align,
                )

                

                if int(diag_s.get("B_cal", 0)) == 0:
                    continue

                per_task_losses.append(loss_s)
                per_task_sizes.append(mask.sum())
                per_task_mean_set.append(float(diag_s.get("mean_set_size_pred", 0.0)))


                # --- Coverage tracker update (per-task tau), without double-counting ---
                if (self._cov_epoch_total is not None) and (self._cov_epoch_covered is not None):
                    with torch.no_grad():
                        new_samples_mask = mask & (~counted)
                        if new_samples_mask.any():
                            y_ns = y[new_samples_mask]
                            # true log-prob for true class for those uncounted samples
                            rows = torch.arange(new_samples_mask.sum().item(), device=y.device)
                            true_scores_ns = log_probs[new_samples_mask][rows, y_ns]
                            tau_vec = diag_s["tau_by_class"].to(y.device)  # (K,)
                            hard_cover_ns = (true_scores_ns >= tau_vec[y_ns]).to(torch.float32)

                            K = self.cfg.num_cell_types
                            total_add = torch.bincount(y_ns, minlength=K)
                            covered_add = torch.bincount(y_ns, weights=hard_cover_ns, minlength=K)

                            self._cov_epoch_total[:K] += total_add.to(self._cov_epoch_total.device)
                            self._cov_epoch_covered[:K] += covered_add.to(self._cov_epoch_covered.device)

                            counted = counted | new_samples_mask

            if len(per_task_losses) > 0:
                # size-weighted mixing across tasks in the minibatch
                w = torch.stack([s.to(mu_z).float() for s in per_task_sizes])
                w = w / w.sum().clamp_min(1.0)
                
                classification_loss = torch.stack(per_task_losses).mul(w).sum()

                # size-weighted mean set size (informational scalar)
                mean_set_tensor = torch.tensor(per_task_mean_set, device=mu_z.device, dtype=mu_z.dtype)
                classification_info = float((w * mean_set_tensor).sum().detach().item())

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
    min_lr: float = 4
    early_stopping_patience: int = 12


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
    ) -> None:
    
                
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg.lr_gamma,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
        verbose=True,
    )

    if not hasattr(model, "history"):
        model.history = []

    best_metric = float("inf")
    best_val_metric = float("inf")
    best_cov = 0.0
    best_state = None
    no_improve = 0
    val_no_improve = 0
    no_improve_cov_overall = 0
    best_callback_covgap = 0.0

    # Default hyperparameters if not provided
    beta = 1.0 if beta is None else beta
    epochs_CG_start = 20 if epochs_CG_start is None else epochs_CG_start
    conf_T_init = 1.5 if conf_T_init is None else conf_T_init
    conf_T_max_decay = 0.5 if conf_T_max_decay is None else conf_T_max_decay
    lambda_size = 1.0 if lambda_size is None else lambda_size
    gamma_tau_align = 1.5 if gamma_tau_align is None else gamma_tau_align


    #beta = 1.5
    # CG-specific settings
    #epochs_CG_start = epochs_CG_start
    epochs_CG_end =  epochs_CG_start + 10
    warm_T_epochs = int(epochs_CG_end - epochs_CG_start)

    model.conf_T = 1.00
    model.conf_T_init = conf_T_init
    model.conf_T_max_decay = conf_T_max_decay
    total_decay_amount = model.conf_T_init - model.conf_T_max_decay

    model.lambda_size = lambda_size
    model.gamma_tau_align = gamma_tau_align

    print("\nPARAMETERS:")

    print(f"  epochs_CG_start: {epochs_CG_start}")
    print(f"  epochs_CG_end: {epochs_CG_end}")
    print(f"  batch_size: {cfg.batch_size}")
    print("ConfTr parameters: ")
    print(f"  conf_T_init: {model.conf_T_init}")
    print(f"  conf_T_max_decay: {model.conf_T_max_decay}")
    print(f"  warm_T_epochs: {warm_T_epochs}")
    print(f"  kappa: {model.kappa}")
    print(f"  lambda_size: {model.lambda_size}")
    print(f"  cover_weight: {model.cover_weight}")
    print(f"  other_weight: {model.other_weight}")
    print(f"  gamma_tau_align: {model.gamma_tau_align}")

   
    #ref_cal_indices = np.asarray(cal_indices, dtype=np.int64)

    Flag_lr_scheduler_step = False
    start_best_val_metric_flag = False
    for epoch in range(cfg.epochs):
        model.train()
        
        # Reset epoch coverage accumulators
        K = model.cfg.num_cell_types
        model._cov_epoch_total = torch.zeros(K, dtype=torch.float32, device=device)
        model._cov_epoch_covered = torch.zeros(K, dtype=torch.float32, device=device)

         
        total_elbo = 0.0
        total_n = 0
        total_loss = 0.0
        total_elbo_loss = 0.0
        total_recon_nll = 0.0
        total_kl = 0.0
        total_conftr_loss = 0.0
        total_conftr_weighted_loss = 0.0
        total_conftr_scaled_loss = 0.0
        

        # Temperature schedule
        if epoch <= epochs_CG_start + 1:
            model.conf_T = model.conf_T_init
        else:
            effective_epoch = epoch - epochs_CG_start
            decay_progress = min(1.0, effective_epoch / warm_T_epochs)
            decay = total_decay_amount * decay_progress
            model.conf_T = float(model.conf_T_init - decay)
            #decay = 0.1 # effective_epoch  / warm_T_epochs # min(model.conf_T_max_decay, effective_epoch  / warm_T_epochs)
            #model.conf_T = max(model.conf_T_max_decay ,float(model.conf_T_init - decay))

        if epoch == epochs_CG_start + 1:
            print("\n--- Starting Conformal Training with SmoothCal loss ---\n")
            best_metric = float("inf")
            best_callback_covgap = 0.0 #float("inf")
            best_cov_overall = 0.0
            no_improve_cov_overall = 0
            #beta = 2.0 # strong regularization from here on for separability

        for xb, bb, yb in train_loader:
            xb = xb.to(device)
            bb = bb.to(device)
            yb = yb.to(device)
            
            recon_loglik, kl, batch_acc, batch_cls_loss = model(xb, bb, yb, reference_batch_dict=reference_batch_dict)
            elbo = recon_loglik - beta * kl # de aquí sale un vector de tamaño batch
            
            
            loss = -elbo.mean()   # objetivo a minimizar (cambio de signo). Se usan medias para evitar que el tamaño del batch afecte al gradiente
            

            if epoch > epochs_CG_start and batch_cls_loss is not None:
  
                batch_loss_weight = 1.5
                weighted_batch_loss = batch_loss_weight * batch_cls_loss

                cvae_magnitude = abs(loss.item())
                batch_magnitude = abs(weighted_batch_loss.detach().item())

                if batch_magnitude > 1e-3:  # Avoid division by very small numbers
                    target_ratio = 0.5  # Adjust this (0.1 to 0.5)
                    scaling_factor = (target_ratio * cvae_magnitude) / batch_magnitude
                    scaling_factor = min(scaling_factor, 5000.0)  # Cap to reasonable range

                    scaled_batch_loss = scaling_factor * weighted_batch_loss # recuerdese que esto es media sobre el batch
                    loss += scaled_batch_loss

                    total_conftr_weighted_loss += weighted_batch_loss.detach().item() * xb.size(0)  
                    total_conftr_scaled_loss += scaled_batch_loss.detach().item()* xb.size(0)
                    #total_conftr_loss += scaled_batch_loss.detach().item() if batch_cls_loss is not None else 0.0
                    
                #loss += weighted_batch_loss

            opt.zero_grad() 
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_elbo += elbo.detach().sum().item()        # acumular la suma de ELBOs en todos los batches de la epoch
            total_n += xb.size(0)                           # acumular el número total de muestras en la epoch
            total_loss += loss.detach().item() * xb.size(0) # el promedio del loss en la epoch * tamaño del batch    

            total_recon_nll += recon_loglik.detach().sum().item()
            total_kl += beta * kl.detach().sum().item()

        train_elbo = total_elbo / max(1, total_n)
        train_recon_nll = total_recon_nll / max(1, total_n)
        train_kl = total_kl / max(1, total_n)
        #train_conftr_loss = total_conftr_loss / max(1, total_n)
        train_conftr_weighted_loss = total_conftr_weighted_loss / max(1, total_n)
        train_conftr_scaled_loss = total_conftr_scaled_loss / max(1, total_n)
        train_loss = total_loss / max(1, total_n)
        
        

        val_elbo = None
        if val_loader is not None:

            model.eval()

            total_val_elbo = 0.0
            val_n = 0
            total_val_recon_nll = 0.0
            total_val_kl = 0.0
            total_val_conftr_weighted_loss = 0.0
            total_val_conftr_scaled_loss = 0.0
            total_val_loss = 0.0
            with torch.no_grad():
                for xb, bb, yb in val_loader:
                    xb = xb.to(device)
                    bb = bb.to(device)
                    yb = yb.to(device)
                    
                    
                    recon_loglik, kl, batch_acc, batch_cls_loss = model(xb, bb, yb, reference_batch_dict=reference_batch_dict)
                    elbo = recon_loglik - beta * kl

                    # Base objective is ELBO mean; subtract scaled batch penalty if available
                    #elbo_mean = elbo.mean().item()

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

                            scaled_batch_loss = scaling_factor * weighted_batch_loss # recuerdese que esto es media sobre el batch
                            loss_val += scaled_batch_loss

                            total_val_conftr_weighted_loss += weighted_batch_loss.detach().item() * xb.size(0)  
                            total_val_conftr_scaled_loss += scaled_batch_loss.detach().item()* xb.size(0)
                            #total_val_conftr_loss += scaled_batch_loss.detach().item() if batch_cls_loss is not None else 0.0
                            

                    total_val_elbo += elbo.detach().sum().item()
                    # accumulate as sum over samples to keep dataset average
                    val_n += xb.size(0)
                    total_val_loss += loss_val.detach().item() * xb.size(0)

                    total_val_recon_nll += recon_loglik.detach().sum().item()
                    total_val_kl += beta * kl.detach().sum().item()


            val_elbo = total_val_elbo / max(1, val_n)
            val_recon_nll = total_val_recon_nll / max(1, val_n)
            val_kl = total_val_kl / max(1, val_n)
            val_conftr_weighted_loss = total_val_conftr_weighted_loss / max(1, val_n)
            val_conftr_scaled_loss = total_val_conftr_scaled_loss / max(1, val_n)
            val_loss = total_val_loss / max(1, val_n)

            metric = val_loss

        else:
            metric = train_elbo

        # Per-class coverage (Mondrian) logging. (This is measured on)
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

        if (epoch + 1) % 1 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1:03d} | beta={beta:.3f} | LR={opt.param_groups[0]['lr']:.6f} | T={model.conf_T:.2f} |  gamma_tau={model.gamma_tau_align:.2f} | Cov(overall)={cov_overall:.3f}"
            print(msg)

            print(f"--Train Loss: ELBO={-train_elbo:.3f} | ConfTr={train_conftr_scaled_loss:.3f} | Total={train_loss:.3f} ")
            print(f"              Recon NLL: {train_recon_nll:.3f} | KL: {train_kl:.3f}") # -ELBO = -((NNLL) - KL) = KL - (NLL)
            #print(f"ConfTr: {train_conftr_weighted_loss:.3f}")
            #print(f"Scaled loss: {train_conftr_scaled_loss:.3f}")
            #print(f"Total loss: {train_loss:.3f}")
            if val_elbo is not None:
                print(f"--Val Loss: ELBO={-val_elbo:.3f} | ConfTr={val_conftr_scaled_loss:.3f} | Total={val_loss:.3f} ")
                print(f"            Recon NLL: {val_recon_nll:.3f} | KL: {val_kl:.3f}") # -ELBO = -((NNLL) - KL) = KL - (NLL)
            
            if per_class_cov:
                pcs = ", ".join([f"{k}:{v:.2f}" for k, v in per_class_cov.items()])
                print(f"--Per-class coverage: {pcs}\n")

        if Flag_lr_scheduler_step == False:
            scheduler.step(metric)

        if epoch_callback is not None and (epoch + 1) % 1 == 0:
            callback_covgap = epoch_callback(epoch + 1, model, False)
            #metric = callback_covgap
            print(f"callback_covgap: {callback_covgap}")
                    
        ### save epoch metrics in history
        entry = {
            "epoch": int(epoch + 1),
            "beta": float(beta),
            "conf_T": float(model.conf_T),
            # métricas de entrenamiento
            "train_total_loss": float(train_loss),
            "train_elbo_loss": float(-train_elbo),   # lo imprimes como -ELBO
            "train_recon_nll": float(train_recon_nll),
            "train_kl": float(train_kl),
            "train_conftr_scaled": float(train_conftr_scaled_loss),
            # métricas de validación (o NaN si no hay val_loader)
            "val_total_loss": float(val_loss) if 'val_loss' in locals() else float('nan'),
            "val_elbo_loss": float(-val_elbo) if 'val_elbo' in locals() and val_elbo is not None else float('nan'),
            "val_recon_nll": float(val_recon_nll) if 'val_recon_nll' in locals() else float('nan'),
            "val_kl": float(val_kl) if 'val_kl' in locals() else float('nan'),
            "val_conftr_scaled": float(val_conftr_scaled_loss) if 'val_conftr_scaled_loss' in locals() else float('nan'),
            # coverage (global y por clase)
            "coverage_overall": float(cov_overall),
        }
       
        for ct_name in cell_types:
            entry[f"coverage_{ct_name}"] = float(per_class_cov.get(ct_name, float('nan')))
        model.history.append(entry)

        ## ---------------------------------------------------------------------------

        # just for logging
        if cov_overall > best_callback_covgap:
            print(f"New best metric found: {cov_overall:.4f} (previous: { best_callback_covgap:.4f})")
            best_callback_covgap  = cov_overall
            
            no_improve = 0

            if callback_covgap <= 0.015:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print("\nSaved best model state based on callback coverage metric.\n")

        #else:
            #if epoch >= epochs_CG_start + 1:
                #no_improve += 1
                #print
                #if no_improve >= cfg.early_stopping_patience - cfg.early_stopping_patience//2:
                    #beta += 0.20
                    #model.gamma_tau_align += 0.10
                    #model.gamma_tau_align = min(model.gamma_tau_align, 2.0)
                    #model.other_weight += 0.04
                    #model.other_weight = min(model.other_weight, 1.0)
                    #print(f"Increasing gamma_tau_align to {model.gamma_tau_align} and to encourage coverage.")
                    #(f"Early stopping at epoch {epoch + 1}")
                    #break
                    #no_improve = 0
        
        # we are closer to the target coverage, we reduce lr to refine and avoid the nets learning too only to increase coverage
        if ( callback_covgap <= 0.01 and epoch > epochs_CG_start ) or (start_best_val_metric_flag == True): # and (cov_overall >= 1.0 - model.alpha - 0.025):
            
            start_best_val_metric_flag = True

            if Flag_lr_scheduler_step == False:
                print(f"Coverage target nearly reached, reducing LR at epoch {epoch + 1}")
                opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] * 0.8
                Flag_lr_scheduler_step = True

            if cov_overall >= 1.0 - model.alpha - 0.01:
                print(f"Desired coverage reached. Stopping at epoch {epoch + 1}")
                break
            
            if cov_overall > best_cov_overall:
                best_cov_overall = cov_overall
                no_improve_cov_overall = 0
            else:
                no_improve_cov_overall += 1
            
            if no_improve_cov_overall >= cfg.early_stopping_patience:
                print(f"Coverage overall did not improve for {cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1} to avoid neurons learn how to overcome coverage.")
                break
            

            #if Flag_lr_scheduler_step == False:
            #    print(f"Coverage target nearly reached, reducing LR at epoch {epoch + 1}")
            #    opt.param_groups[0]['lr'] = 1e-5 #opt.param_groups[0]['lr'] * 0.1
            #    Flag_lr_scheduler_step = True

            #if callback_avg_size < 1.0:
            #    no_improve += 1
            #    if no_improve >= cfg.early_stopping_patience:
            #        print(f"Average set size did not improve for {cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1} to avoid neurons learn how to overcome coverage.")
            #        break
            #else:

            #print(f"Desired coverage reached. Stopping at epoch {epoch + 1}")
            #break
                

            
            
            
            #if callback_coverage > best_cov:
             #   best_cov = callback_coverage
            #else:
            #    no_improve += 1
            
            #if cov_overall >= 1.0 - model.alpha - 0.01:
            #    print(f"Desired coverage reached. Stopping at epoch {epoch + 1}")
            #    break

            #if no_improve >= cfg.early_stopping_patience:
                
            #    print(f"Coverage did not improve for {cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1} to avoid neurons learn how to overcome coverage.")
            #    if epoch_callback is not None:
            #        epoch_callback(epoch + 1, model,True)

            #    break

        #if start_best_val_metric_flag:
        #    print( np.log(metric) + 1e-2, best_val_metric)
        #    print(val_no_improve)
        #    if np.log(metric) + 1e-2 > best_val_metric:
        #        val_no_improve += 1
        #        if val_no_improve >= cfg.early_stopping_patience:
        #            print(f"Validation metric did not improve for {cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1}.")
        #            break

            #else:
            #    val_no_improve = 0
            #    best_val_metric = np.log(metric)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model state based on callback coverage metric.")

    if epoch_callback is not None:
        epoch_callback(epoch + 1, model,True)    
                
    


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

class DataPreparation():

    def data_loader_prep(
        self,
        obs_df: pd.DataFrame,
        batch_key: str,
        cell_type_col: str,
        reference_dictionary: dict,
        seed: int = 0):
    
        
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


        print("\nBatch to index mapping:", batch_to_idx)
        print("Cell type to index mapping:", cell_type_to_idx)
        print("Reference to targets index mapping:", self.ref_to_targets_idx)

        batch_codes = obs_df[batch_key].map(batch_to_idx).to_numpy()
        cell_type_codes = obs_df[cell_type_col].map(cell_type_to_idx).to_numpy()

        self.b_tensor = torch.from_numpy(batch_codes).long()
        self.ct_tensor = torch.from_numpy(cell_type_codes).long()

        # Extract non-reference batch data indices
        self.reference_batch_idx_list = []
        for ref in reference_batch_default:

            reference_batch_idx = batch_to_idx[ref]
            self.reference_batch_idx_list.append(reference_batch_idx)

            print(f"Reference batch '{ref}' has index {reference_batch_idx}")



    def build_data_tensors(
        self,
        counts_array: np.ndarray,
        seed_offset: int = 0,
        val_fraction: float = 0.2,
        batch_size: int = 256,
        calibration_fraction: float = 0.3,
    ):  
        
        counts_tensor = torch.from_numpy(counts_array).float()
        if counts_tensor.shape[0] != self.b_tensor.shape[0] or counts_tensor.shape[0] != self.ct_tensor.shape[0]:
            raise ValueError("Counts tensor must align with batch and cell type annotations")
        dataset = TensorDataset(counts_tensor, self.b_tensor, self.ct_tensor)

        # Determine calibration indices from reference batch only (recommended)
        # Hold out a fixed fraction for calibration and exclude from train/val

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
                    # we implement stratified sampling by cell type here

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

                    # Asegura el tamaño total repartiendo los sobrantes con mayor parte decimal.
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

                    #cal_indices = rng_local.choice(ref_all, size=cal_size, replace=False)
                else:
                    cal_indices = np.asarray([], dtype=np.int64)

                cal_indices_list.append(cal_indices)
        else:
            cal_indices = np.asarray([], dtype=np.int64)

        cal_index_flatten = np.concatenate(cal_indices_list) if calibration_fraction > 0.0 else np.asarray([], dtype=np.int64)
        
        # Pool for training/validation excludes calibration indices
        if cal_index_flatten.size > 0:
            pool_indices = np.setdiff1d(all_indices, cal_index_flatten, assume_unique=False)
        else:
            pool_indices = all_indices

        # Build train/val subsets from pool
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


        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
        
        print(f"Train dataset length: {len(train_ds)}")
        unique_labels, counts = torch.unique(dataset.tensors[2][train_ds.indices], return_counts=True)
        label_counts_pytorch = dict(zip(unique_labels.tolist(), counts.tolist()))
        print(label_counts_pytorch)

        print(f"\nValidation dataset length: {len(val_ds) if val_ds is not None else 0}")
        unique_labels, counts = torch.unique(dataset.tensors[2][val_ds.indices], return_counts=True)
        label_counts_pytorch = dict(zip(unique_labels.tolist(), counts.tolist()))
        print(label_counts_pytorch)
        
        for idx_cal in cal_indices_list:
            print(f"\nCalibration set size: {idx_cal.size}")
            unique_labels, counts = torch.unique(dataset.tensors[2][idx_cal], return_counts=True)
            label_counts_pytorch = dict(zip(unique_labels.tolist(), counts.tolist()))
            print(label_counts_pytorch)
        
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
) -> None:


    if val_indices.size == 0:
        print("Latent conformal evaluation skipped: validation set empty.")
        return

    alpha_list = [1.0 - target_coverage_ for target_coverage_ in target_coverage] 


    def compute_q_hat(nonconformity: np.ndarray, alpha:float) -> float:
        n_cal = nonconformity.size
        if n_cal == 0:
            return 1.0
        ordered = np.sort(nonconformity)
        
        rank = int(np.ceil((n_cal + 1) * (1 - alpha))) - 1
        rank = max(0, min(rank, n_cal - 1))
        return ordered[rank]

    def run_scenario(
        clf: LogisticRegression,
        train_subset: np.ndarray,
        eval_subset: np.ndarray,
        cal_indices: np.ndarray,
        target_coverage: List[float],
    ) -> None:
        if eval_subset.size == 0 or train_subset.size == 0:
            print(f"skipped: insufficient samples.")
            return

        clf.fit(latent_np[train_subset], cell_labels[train_subset])

        cal_probs = clf.predict_proba(latent_np[cal_indices])
        cal_true_cols = np.searchsorted(clf.classes_, cell_labels[cal_indices])
        cal_scores = 1.0 - cal_probs[np.arange(cal_indices.size), cal_true_cols]

        eval_probs = clf.predict_proba(latent_np[eval_subset])
        eval_true_cols = np.searchsorted(clf.classes_, cell_labels[eval_subset])

        cov_list = []
        size_list = []
        for alpha in alpha_list:

            q_hat = compute_q_hat(cal_scores, alpha)
            threshold = 1.0 - q_hat
            
            try:
                contains_true = eval_probs[np.arange(eval_subset.size), eval_true_cols] >= threshold
            except IndexError as e:
                print(e)
                print("IndexError in evaluating conformal sets. Check class labels and predictions. Probably some cell-type missing not in the reference batch.")
                exit(1)


            set_sizes = (eval_probs >= threshold).sum(axis=1)

            coverage = float(contains_true.mean())
            avg_size = float(set_sizes.mean())

            cov_list.append(coverage)
            size_list.append(avg_size)

        for coverage, avg_size, target_coverage_ in zip(cov_list, size_list, target_coverage):

            print(
                f"coverage: {coverage:.4f}, (target coverage {target_coverage_:.0%}. average set size: {avg_size:.2f})\n\n"
            )

        covgap = np.mean([abs(c - t) for c, t in zip(cov_list, target_coverage)])

        return covgap

    # Scenario 1: train on all batches → validate on all batches
    #run_scenario(
    #    LogisticRegression(max_iter=1000),
    #    train_indices,
    #    val_indices,
    #    "Latent conformal (train all batches -> test all batches) (allways achieved target coverage):\n",
    #)

    # Scenario 2: train only on reference batch → validate on remaining batches
    for idx,(reference_batch_idx, target_batch_idx_list) in enumerate(ref_to_targets_idx.items()):

        train_mask = batch_labels[train_indices] == reference_batch_idx
        eval_mask = np.isin(batch_labels[val_indices], target_batch_idx_list)

        #if len(target_batch_idx_list) == 1:
        #    eval_mask = batch_labels[val_indices] == target_batch_idx_list[0]  # solo uno
        #else:
        #   eval_mask = batch_labels[val_indices] == target_batch_idx_list[0]
        #   for t in target_batch_idx_list:
        #       eval_mask = eval_mask | (batch_labels[val_indices] == t)
               #eval_mask = batch_labels[val_indices] == target_batch_idx_list #aqui es donde hay que hacer que sean solo de los que queremos
       
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
            print("Latent conformal (train ref -> test other) skipped: insufficient samples.")

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
                device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        
        self.seed_offset = seed_offset
        self.epochs = epochs
        self.lr = lr
        self.kl_anneal_epochs = kl_anneal_epochs
        self.batch_size = batch_size
        self.calibration_fraction = calibration_fraction
        self.val_fraction = val_fraction
        self.device = device# torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def train_cvae_on_counts(
        self,
        counts_array: np.ndarray,
        df_obs: pd.DataFrame,
        batch_key: str,
        cell_type_col: str,
        cell_types: list,
        ref_batch: dict,
        verbose: bool = True,
        beta : float = None,
        epochs_CG_start : int = None,
        conf_T_init : float = None,
        conf_T_max_decay : float = None,
        lambda_size : float = None,
        gamma_tau_align : float = None,
    ):
        
        # lets print the original UMAP before training:
        reset_umap_basis()
        #save_latent_umap(counts_array, f"Umap counts", f"UMAP_original.png")

        #check if counts_array is sparse:
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
           
        
        print("\nBuilding data tensors and adjusting batch size...")
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

                cal_samples = []
                for xb, bb, yb in train_loader:
                    xb = xb.to(self.device)
                    bb = bb.to(self.device)
                    yb = yb.to(self.device)
                    
                    # mask for this task: only batches in {ref} ∪ targets
                    mask = (bb == ref_idx) # cal 
                    
                    for t in target_idxs:
                        mask |= (bb == t)
                    
                    if not mask.any():
                        continue   
                    
                    y_s = yb[mask]
                    b_s = bb[mask]
                    
                    cal_mask_s = (b_s == ref_idx)

                    all_idx = torch.arange(b_s.size(0), device=self.device)
                
                    pool_idx = all_idx[cal_mask_s]
                        
                    cal_samples.append(pool_idx.numel())

                cal_samples_av = np.sum(cal_samples)/len(cal_samples)  

                print(f"Current batch size: {self.batch_size}, average cal samples for ref batch {ref_idx}: {cal_samples_av}")
                if cal_samples_av <= 90:
                    self.batch_size = int(self.batch_size * 2)

        print(f"\nAdjusted batch size: {self.batch_size}\n")

        print("cal indices length:", len(cal_indices))
        print(self.reference_batch_idx_list)
        print(self.ref_to_targets_idx)

        
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



        print("\nTraining CVAE...")
        def latent_progress_callback(epoch_idx: int, model_snapshot: ConditionalVAE, final:bool) -> None:
            
            model_snapshot.eval()
            with torch.no_grad():
                mu_epoch, _ = model_snapshot.encode(
                    counts_tensor.to(self.device),
                    self.b_tensor.to(self.device),
                    self.ct_tensor.to(self.device),
                )
            latent_epoch_np = mu_epoch.cpu().numpy()

            if verbose:
                save_latent_plot(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}.png")

            if val_loader is not None:

                train_idx = np.asarray(train_loader.dataset.indices, dtype=int)
                val_idx = np.asarray(val_loader.dataset.indices, dtype=int)
                target_coverage = [0.90]
                covgap = evaluate_latent_logreg(
                            latent_epoch_np,
                            train_idx,
                            val_idx,
                            self.b_tensor.cpu().numpy(),
                            self.ct_tensor.cpu().numpy(),
                            self.ref_to_targets_idx,
                            cal_indices,  # cal_indices already filtered to only cointain reference batch samples
                            target_coverage)  


            #if int(epoch_idx) == 0:
                #reset_umap_basis()
            
            if int(epoch_idx) == 20 and verbose:
                save_latent_umap(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}_UMAP.png")

            if int(epoch_idx) % 40 == 1 and verbose:
                save_latent_umap(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}_UMAP.png")

            if final:
                if verbose:
                    save_latent_umap(latent_epoch_np, f"epoch {epoch_idx:03d}", f"latent_space_epoch_{epoch_idx:03d}_UMAP.png")

                if val_loader is not None:

                    train_idx = np.asarray(train_loader.dataset.indices, dtype=int)
                    val_idx = np.asarray(val_loader.dataset.indices, dtype=int)
                    target_coverage = [0.90]
                    evaluate_latent_logreg(
                        latent_epoch_np,
                        train_idx,
                        val_idx,
                        self.b_tensor.cpu().numpy(),
                        self.ct_tensor.cpu().numpy(),
                        self.ref_to_targets_idx,
                        cal_indices,
                        target_coverage)  # cal_indices already filtered to only cointain reference batch samples
                    

            model_snapshot.train()


            return covgap

            


        train_cvae(
            model,
            train_loader,
            val_loader,
            train_cfg,
            reference_batch_dict=self.ref_to_targets_idx, #reference_batch_idx,
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
        )


        return model, counts_tensor


# ---------------------------------------------------------------------------
# MAAIN SCRIPT
# --------------------------------------------------------------------------


if __name__ == "__main__":
   
    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------

    EPS = 1e-8
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} (GPU available: {torch.cuda.is_available()})")

    n_genes = 4000
    adata_path = "../../data/diabetic-kidney-disease_processed" #"../data/gexV9_processed" #"../data/diabetic-kidney-disease_processed"
    adata_read_path = adata_path + ".h5ad"
    adata_save_path = adata_path + "_integrated_crTr.h5ad"
    adata = sc.read_h5ad(adata_read_path) # sc.read_h5ad("../data/diabetic_processed.h5ad") #sc.read_h5ad("../data/MB_processed.h5ad") # sc.read_h5ad("../data/adata_tutorial.h5ad")

    cell_type_col = "cell_type" #"cell_type" # "cell_type_eval"
    use_layer = "counts"
    batch_key = "batch" #"assay" #"batch" # "batch" # "system"
    vae_decoder_type = "gaussian"

    #reference_dictionary = {"healthy_6": ['healthy_5', 'healthy_4'],"control_3": ["control_1", "control_2"]}


    reference_dictionary = {"healthy_6": ["control_1", "control_2", "control_3", "healthy_4", "healthy_5", 'diabetic_2', 'diabetic_3','diabetic_4', 'diabetic_5']}

    #reference_dictionary = {"GTEX-1HSMQ": [ 'GTEX-13N11', 'GTEX-1ICG6', 'GTEX-15RIE', 'GTEX-145ME', 'GTEX-1I1GU', 'GTEX-16BQI', 'GTEX-144GM', 'GTEX-15CHR', 'GTEX-15SB6', 'GTEX-12BJ1', 'GTEX-1R9PN', 'GTEX-1CAMS', 'GTEX-1MCC2', 'GTEX-15EOM' ]}

    #reference_dictionary = {"10x 5' v1": ["10x 3' v3"]}


    reference_batch_default =[]
    batches = []
    for key, val in reference_dictionary.items():
        reference_batch_default.append(key)
        batches.extend(val)
        batches.append(key)


    # ---------------------------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------------------------

    if adata.layers.get(use_layer) is None:
        adata.layers[use_layer] = adata.X.astype(int)
    #adata.layers["counts"] = adata.X.astype(int)

    adata.obs[batch_key] = adata.obs[batch_key].astype(str)
    if batches:
        adata = adata[adata.obs[batch_key].isin(batches)].copy()
        batches = sorted(adata.obs[batch_key].unique())
    else:
        batches = sorted(adata.obs[batch_key].unique())

    adata = adata[adata.obs[cell_type_col].value_counts()[adata.obs[cell_type_col]].values >= 400].copy()

    if n_genes is not None:
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_genes,
                flavor="seurat_v3",
                batch_key=batch_key,
                subset=True,
                layer=use_layer,
            )
        except Exception as e:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_genes,
                flavor="seurat_v3",
                batch_key=batch_key,
                subset=True,
                span=0.7,
                layer=use_layer,
            )

    counts_matrix = (
        adata.layers[use_layer].toarray()
        if sparse.issparse(adata.layers[use_layer])
        else np.asarray(adata.layers[use_layer], dtype=np.float32)
    )

    counts_matrix = counts_matrix.astype(np.float32)
    obs_df = pd.DataFrame(adata.obs)

    if vae_decoder_type == "gaussian":
        if "lognorm_gaussian" not in adata.layers:
            
            temp_adata = sc.AnnData(adata.layers[use_layer])
            sc.pp.normalize_total(temp_adata, target_sum=1e4)
            sc.pp.log1p(temp_adata)
            
            adata.layers["lognorm_gaussian"] = temp_adata.X.copy()

        model_matrix = adata.layers["lognorm_gaussian"].astype(np.float32, copy=True)
        #print(model_matrix.shape, model_matrix.dtype, model_matrix[0,:5])

    else:
        model_matrix = counts_matrix.copy()

    cell_types = adata.obs[cell_type_col].astype(str).unique().tolist()


    # Tensor view of the full training matrix for callbacks/calibration
    #full_counts_tensor = torch.from_numpy(model_matrix).float()

    print("Dataset shape:", model_matrix.shape)
    print("Cell types:", cell_types)
    print("Batches:", batches)


    # ---------------------------------------------------------------------------
    # Initial plotting UMAP and PCA before integration:
    # ---------------------------------------------------------------------------
    
    umap_basis_model = None  # will hold the fitted UMAP for consistent coordinates
    sc.pp.pca(adata, n_comps=vae_latent_dim, svd_solver='arpack',key_added = "X_pca", layer="lognorm_gaussian")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=vae_latent_dim , use_rep="X_pca")
    sc.tl.umap(adata, n_components=2, key_added="X_umap")

    cols_to_visualize = [batch_key, cell_type_col]

    fig, axs = plt.subplots(1, len(cols_to_visualize), figsize=(12, 5))

    for i, (col, ax) in enumerate(zip(cols_to_visualize, axs)):
        sc.pl.embedding(
            adata,
            layer = "lognorm_gaussian",
            basis="X_umap",  # Specify which embedding to use
            color=col,       # The metadata column to color by
            s=5,            # Size of the points in the scatter plot
            ax=ax,           # The matplotlib axes object to plot on
            show=False,      # Do not display the plot immediately
            sort_order=False,# Do not sort points by color value
            frameon=False,   # Remove the plot frame
            legend_loc='on data' if col == cell_type_col else 'right margin', # Custom legend location
            legend_fontsize=8,
            title=col.replace('_', ' ').title() # Create a clean title (e.g., 'cell_type' -> 'Cell Type')
        )
        # Customize axis labels for clarity
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)

    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout()
    output_filename = os.path.join(plots_dir, "umap_preintegration.png") 
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')

    pca_raw = PCA(n_components=2)
    pca_input = model_matrix if vae_decoder_type == "gaussian" else np.log1p(counts_matrix)
    coords_raw = pca_raw.fit_transform(pca_input)

    plot_df = pd.DataFrame(
        {
            "PC1": coords_raw[:, 0],
            "PC2": coords_raw[:, 1],
            batch_key: obs_df[batch_key].values,
            cell_type_col: obs_df[cell_type_col].values,
        }
    )

    if len(batches) == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        if sns:
            sns.scatterplot(
                data=plot_df,
                x="PC1",
                y="PC2",
                hue=cell_type_col,
                ax=ax,
                s=5,
                alpha=0.7,
            )
            ax.legend(loc="best", fontsize=9)
        else:
            for label in plot_df[cell_type_col].unique():
                mask = plot_df[cell_type_col] == label
                ax.scatter(
                    plot_df.loc[mask, "PC1"],
                    plot_df.loc[mask, "PC2"],
                    s=35,
                    alpha=0.7,
                    label=label,
                )
            ax.legend(loc="best", fontsize=9)
        ax.set_title(f"Batch {batches[0]} by cell type")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, hue, title in zip(
            axes,
            [cell_type_col, batch_key],
            ["Original data by cell type", "Original data by batch"],
        ):
            if sns:
                sns.scatterplot(
                    data=plot_df,
                    x="PC1",
                    y="PC2",
                    hue=hue,
                    ax=ax,
                    s=5,
                    alpha=0.7,
                )
                ax.legend(loc="best", fontsize=9)
            else:
                for label in plot_df[hue].unique():
                    mask = plot_df[hue] == label
                    ax.scatter(
                        plot_df.loc[mask, "PC1"],
                        plot_df.loc[mask, "PC2"],
                        s=35,
                        alpha=0.7,
                        label=label,
                    )
                ax.legend(loc="best", fontsize=9)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(title)
        plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pca_original.png"), dpi=150)
    plt.close()




    train_cvae_epochs_default =  100
    train_cvae_lr_default = 5e-4
    train_cvae_kl_anneal_epochs_default = 1
    train_cvae_batch_size_default = 32
    calibration_fraction_default = 0.3  # Set >0 to hold out a calibration set (e.g., 0.4)
    build_data_val_fraction_default = 0.2

    integrador = Integrator(
        seed_offset=0,
        epochs=train_cvae_epochs_default,
        lr=train_cvae_lr_default,
        kl_anneal_epochs=train_cvae_kl_anneal_epochs_default,
        batch_size=train_cvae_batch_size_default,
        calibration_fraction=calibration_fraction_default,
        val_fraction=build_data_val_fraction_default,
    )

    

    model, counts_tensor = integrador.train_cvae_on_counts(
            model_matrix,
            df_obs=obs_df,
            batch_key=batch_key,
            cell_type_col=cell_type_col,
            cell_types=cell_types,
            ref_batch=reference_dictionary,
        )


    print("\nModel integrated!")

    plot_training_curves(model, plots_dir=plots_dir)

    model.eval()
    with torch.no_grad():
        mu_z, logvar_z = model.encode(
            counts_tensor.to(device),
            integrador.b_tensor.to(device),
            integrador.ct_tensor.to(device),
        )
        latent_np = mu_z.cpu().numpy()
        recon_mu, _ = model.decode(mu_z, integrador.b_tensor.to(device), integrador.ct_tensor.to(device))
        recon_np = recon_mu.cpu().numpy()
        logvar_np = logvar_z.cpu().numpy()



    print("Adding CVAE embedding to adata....")
    adata.obsm['X_ConfTr'] = latent_np
    if 'umap_basis_model' in globals() and umap_basis_model is not None:
        adata.obsm['X_ConfTr_umap'] = umap_basis_model.transform(latent_np)

    print(f"Saving AnnData object to '{adata_save_path}'...")
    adata.write_h5ad(adata_save_path)









# ---------------------------------------------------------------------------
# Plot latent space
# ---------------------------------------------------------------------------

#save_latent_plot(latent_np, "final", "latent_space.png")


# ---------------------------------------------------------------------------
# Plot reconstructed space
# ---------------------------------------------------------------------------

    pca_recon = PCA(n_components=2)
    coords_recon = pca_recon.fit_transform(recon_np)

    recon_df = pd.DataFrame(
        {
            "PC1": coords_recon[:, 0],
            "PC2": coords_recon[:, 1],
            batch_key: obs_df[batch_key].values,
            cell_type_col: obs_df[cell_type_col].values,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, hue_col, title in zip(
        axes,
        [batch_key, cell_type_col],
        ["Reconstruction by batch", "Reconstruction by cell type"],
    ):
        if sns:
            sns.scatterplot(
                data=recon_df,
                x="PC1",
                y="PC2",
                hue=hue_col,
                ax=ax,
                s=35,
                alpha=0.7,
            )
            ax.legend(loc="best", fontsize=8)
        else:
            for label in recon_df[hue_col].unique():
                mask = recon_df[hue_col] == label
                ax.scatter(
                    recon_df.loc[mask, "PC1"],
                    recon_df.loc[mask, "PC2"],
                    s=35,
                    alpha=0.7,
                    label=label,
                )
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reconstructed_space.png"), dpi=150)
    plt.close()


    print(f"Plots saved to '{plots_dir}'.")
