"""
validation.py — Regulatory Validation and Cluster Reporting (RTX A5000)
========================================================================
Post-processing functions for:
  1. Tail-risk cluster Precision / Recall against ground-truth TARGET column
  2. Per-cluster PD/CAR profile table (regulatory report format)
  3. Convergence plot (matplotlib — optional, skipped if unavailable)

The tail-risk validation hook corresponds to Section V-D of the paper:
a cluster with individual CAR ≈ 6.8% is retained by the portfolio-level
Lagrangian while aggregate portfolio CAR ≥ 8%.

RTX A5000 note: all GPU ops use the same CuPy API as the A800 port.
No architecture-specific changes are required in this module.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import cupy as cp


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Tail-Risk Precision / Recall  (Section V-D validation hook)
# ─────────────────────────────────────────────────────────────────────────────

def tail_risk_precision_recall(
        labels_gpu:           cp.ndarray,       # (n,) cluster assignments on GPU
        pd_col_gpu:           Optional[cp.ndarray],
        true_target:          np.ndarray,        # (n,) binary 0/1 (Kaggle TARGET)
        k:                    int,
        tier1:                float,
        lgd:                  float,
        target_car_threshold: float = 0.068,    # 6.8% paper tail cluster
        verbose:              bool  = True
) -> dict:
    """
    Identify the cluster whose individual CAR is closest to `target_car_threshold`
    (6.8% — the high-risk tail cluster the paper retains via portfolio-level CAR),
    then evaluate Precision, Recall, F1 against the ground-truth TARGET column.

    Parameters
    ----------
    labels_gpu    : (n,) int32 cluster assignments (GPU)
    pd_col_gpu    : (n,) float64 PD values (GPU); None → uses 5% default
    true_target   : (n,) binary 0/1 CPU array — Kaggle HOME CREDIT TARGET
    k             : number of clusters
    tier1         : Tier 1 capital (£/$ nominal)
    lgd           : Loss Given Default
    target_car_threshold : individual cluster CAR target (6.8% in paper)
    verbose       : print detailed report

    Returns
    -------
    dict with precision, recall, f1, tail_cluster_id, tail_cluster_car
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    labels_cpu = cp.asnumpy(labels_gpu)
    n          = len(labels_cpu)

    # ── Per-cluster individual CAR ────────────────────────────────────────────
    cluster_cars  = []
    cluster_sizes = []
    cluster_pds   = []

    for j in range(k):
        mask_j = labels_cpu == j
        n_j    = int(mask_j.sum())
        cluster_sizes.append(n_j)

        if n_j == 0:
            cluster_cars.append(1.0)
            cluster_pds.append(0.0)
            continue

        if pd_col_gpu is not None:
            pd_j = float(pd_col_gpu[cp.array(mask_j)].mean())
        else:
            pd_j = 0.05

        pd_j  = float(np.clip(pd_j, 0.001, 0.999))
        rw_j  = 12.5 * pd_j * lgd
        car_j = float(tier1 / (n_j * rw_j)) if n_j > 0 else 1.0  # IRB: RWA = EAD × RW
        cluster_cars.append(car_j)
        cluster_pds.append(pd_j)

    # ── Tail cluster: closest individual CAR to threshold ────────────────────
    tail_cluster = int(
        np.argmin([abs(c - target_car_threshold) for c in cluster_cars])
    )

    pred_binary = (labels_cpu == tail_cluster).astype(np.int32)
    true_binary = (true_target > 0).astype(np.int32)

    prec   = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1     = f1_score(true_binary, pred_binary, zero_division=0)

    result = {
        "tail_cluster_id":        tail_cluster,
        "tail_cluster_car":       f"{cluster_cars[tail_cluster]:.2%}",
        "tail_cluster_size":      cluster_sizes[tail_cluster],
        "tail_cluster_pct":       f"{cluster_sizes[tail_cluster] / n:.1%}",
        "tail_cluster_pd_mean":   f"{cluster_pds[tail_cluster]:.3f}",
        "target_car_threshold":   f"{target_car_threshold:.1%}",
        "precision":              round(prec, 4),
        "recall":                 round(recall, 4),
        "f1_score":               round(f1, 4),
    }

    if verbose:
        print("\n── Tail-Risk Validation (Section V-D) — RTX A5000 ─────────────────────")
        for key, val in result.items():
            print(f"  {key:<35s}: {val}")
        print("─────────────────────────────────────────────────────────────────────────")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-Cluster Risk Profile Table
# ─────────────────────────────────────────────────────────────────────────────

def cluster_risk_profile(
        labels_gpu:  cp.ndarray,
        pd_col_gpu:  Optional[cp.ndarray],
        k:           int,
        tier1:       float,
        lgd:         float,
        car_min:     float = 0.08
) -> list[dict]:
    """
    Print and return a per-cluster risk profile table matching the paper's
    regulatory audit output format (Section V-D, Table IV notes).

    Columns: Cluster | Size | Size% | Mean PD | Risk Weight | Indiv. CAR | Basel III
    """
    labels_cpu = cp.asnumpy(labels_gpu)
    n          = len(labels_cpu)
    rows       = []

    print("\n── Per-Cluster Risk Profile (RTX A5000) ─────────────────────────────────────")
    print(f"  {'Cluster':>7}  {'Size':>8}  {'Size%':>6}  {'Mean PD':>8}  "
          f"{'RiskWt':>8}  {'Indiv CAR':>10}  {'Basel':>6}")
    print("  " + "─" * 72)

    for j in range(k):
        mask_j = labels_cpu == j
        n_j    = int(mask_j.sum())
        if n_j == 0:
            continue

        if pd_col_gpu is not None:
            pd_j = float(pd_col_gpu[cp.array(mask_j)].mean())
        else:
            pd_j = 0.05

        pd_j  = float(np.clip(pd_j, 0.001, 0.999))
        rw_j  = 12.5 * pd_j * lgd
        car_j = float(tier1 / (n_j * rw_j)) if n_j > 0 else 1.0  # IRB: RWA = EAD × RW
        ok    = "✓" if car_j >= car_min else "✗ tail"

        print(f"  {j:>7}  {n_j:>8,}  {n_j/n:>6.1%}  {pd_j:>8.4f}  "
              f"{rw_j:>8.4f}  {car_j:>10.2%}  {ok:>6}")

        rows.append({
            "cluster": j, "size": n_j, "size_pct": n_j / n,
            "mean_pd": pd_j, "risk_weight": rw_j,
            "individual_car": car_j, "basel_compliant": car_j >= car_min,
        })

    print("─────────────────────────────────────────────────────────────────────────────")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Convergence Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
        history:     dict,
        output_path: str = "results/convergence.png"
) -> None:
    """
    Plot fitness, J_M, and portfolio CAR vs. iteration.
    Saves to output_path. Skips silently if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[validation] matplotlib not available; skipping convergence plot.")
        return

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    iters   = list(range(1, len(history["fitness"]) + 1))
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(iters, history["fitness"], color="#1f77b4", lw=1.5)
    ax[0].set_ylabel("Fitness F (eq 9)")
    ax[0].set_title("D-CUDA-FA Convergence — RTX A5000 sm_86")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(iters, history["jm"], color="#d62728", lw=1.5)
    ax[1].set_ylabel("J_M Mahalanobis TICV (eq 4)")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(iters, [c * 100 for c in history["car"]],
               color="#2ca02c", lw=1.5)
    ax[2].axhline(8.0, color="k", ls="--", lw=1.0, label="Basel III 8% floor")
    ax[2].set_ylabel("Portfolio CAR (%)")
    ax[2].set_xlabel("Iteration")
    ax[2].legend(fontsize=9)
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[validation] Convergence plot saved to {output_path}")
