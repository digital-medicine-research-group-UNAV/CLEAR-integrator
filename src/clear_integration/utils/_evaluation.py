"""Evaluation helpers used by training diagnostics and paper metrics."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression


def evaluate_latent_logreg(
    latent_np: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_labels: np.ndarray,
    cell_labels: np.ndarray,
    ref_to_targets_idx: dict[int, list[int]],
    cal_indices: np.ndarray,
    target_coverage: List[float] = [0.90],
    gap_objective: str = "mean",
    return_details: bool = False,
) -> None:

    latent_np = latent_np.astype(np.float32)

    if val_indices.size == 0:
        print("Latent conformal evaluation skipped: validation set empty.")
        return {"covgap": np.nan, "coverage": []} if return_details else None

    alpha_list = np.array([1.0 - target_coverage_ for target_coverage_ in target_coverage]) 


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

        """ 
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
        """
        
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

        mask = eval_probs[None, :, :] >= thresholds[:, None, None]   # (n_alpha, n_eval, n_classes)
        set_sizes_all_alphas = mask.sum(axis=2)                      # sumamos sobre clases → (n_alpha, n_eval)
        size_list = set_sizes_all_alphas.mean(axis=1) 

        # Calcular la cobertura para todos los alfas de una vez
        eval_true_probs = eval_probs[np.arange(eval_subset.size), eval_true_cols]
        contains_true_all_alphas = eval_true_probs[None, :] >= thresholds[:, None]
        cov_list = contains_true_all_alphas.mean(axis=1)

        for coverage, avg_size, target_coverage_ in zip(cov_list, size_list, target_coverage):

            print(
                f"coverage: {coverage:.4f}, (target coverage {target_coverage_:.0%}. average set size: {avg_size:.2f})\n\n"
            )

        gaps = np.array([t - c for c, t in zip(cov_list, target_coverage)], dtype=np.float32)
        mean_gap = float(np.mean(gaps))
        max_gap = float(np.max(gaps))

        details = []
        for coverage, avg_size, target_coverage_, alpha_, gap in zip(
            cov_list,
            size_list,
            target_coverage,
            alpha_list,
            gaps,
        ):
            details.append({
                "target_coverage": float(target_coverage_),
                "alpha": float(alpha_),
                "empirical_coverage": float(coverage),
                "coverage_gap": float(gap),
                "avg_set_size": float(avg_size),
                "n_train": int(train_subset.size),
                "n_cal": int(cal_indices.size),
                "n_eval": int(eval_subset.size),
            })

        covgap = max_gap if gap_objective == "max" else mean_gap
        return covgap, details

    
    # : train only on reference batch → validate on remaining batches
    covgap = float("nan")
    coverage_details = []
    for idx,(reference_batch_idx, target_batch_idx_list) in enumerate(ref_to_targets_idx.items()):

        train_mask = batch_labels[train_indices] == reference_batch_idx
        eval_mask = np.isin(batch_labels[val_indices], target_batch_idx_list)
       
        cal_indices_aux = cal_indices[idx]
    

        if train_mask.any() and eval_mask.any():
            covgap, scenario_details = run_scenario(
                                    LogisticRegression(max_iter=10000),
                                    train_indices[train_mask],
                                    val_indices[eval_mask],
                                    cal_indices_aux,
                                    target_coverage
                                )
            for detail in scenario_details:
                detail["reference_batch_idx"] = int(reference_batch_idx)
                detail["target_batch_idx_list"] = ",".join(str(x) for x in target_batch_idx_list)
            coverage_details.extend(scenario_details)
        else:
            print("Latent conformal (train ref -> test other) skipped: insufficient samples.")

    if return_details:
        return {"covgap": covgap, "coverage": coverage_details}
    return covgap

__all__ = ["evaluate_latent_logreg"]
