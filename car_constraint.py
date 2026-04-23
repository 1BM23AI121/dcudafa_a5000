"""
car_constraint.py — Basel III CAR Constraint & EWM Weighting (RTX A5000)
=========================================================================
Implements:
  • Entropy Weight Method (EWM)                 — eq (5)
  • Portfolio-level CAR                          — eq (6)
  • Portfolio-level Lagrangian penalty ΨBasel   — eq (7)
  • Dynamic VIX-linked multiplier λ(t)          — eq (8)
  • Complete multi-objective fitness F           — eq (9)

All heavy array ops are vectorised CuPy (GPU); only scalar results
cross the PCIe bus.

Key design choice (paper, Section III-D):
  ΨBasel = 0  when  CARport ≥ 8 %,  regardless of any individual cluster's
  CAR.  This preserves genuine tail-risk clusters (e.g. a 6.8 %-CAR cluster
  offset by two low-risk clusters at 16.2 % and 14.9 %).  K-Means-style
  cluster-level penalties would fragment this structure.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import cupy as cp

from config import DCUDAFAConfig


# ─────────────────────────────────────────────────────────────────────────────
# Entropy Weight Method  (Section III-C, eq 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ewm_weights(Xp: cp.ndarray) -> cp.ndarray:
    """
    Eq (5): Shannon entropy per standardised feature → EWM weight vector.

    For feature j:
      pᵢⱼ = x'ᵢⱼ / Σᵢ x'ᵢⱼ          (proportional distribution)
      eⱼ  = −(ln n)⁻¹ Σᵢ pᵢⱼ ln pᵢⱼ  (normalised Shannon entropy)
      wⱼ  = (1−eⱼ) / Σⱼ(1−eⱼ)        (diversity weight)

    High-variation, high-discriminatory-power features receive larger wⱼ.
    Weights are computed once on GPU from training data and logged to the
    regulatory audit trail (Basel III model validation / IFRS 9 governance).

    Parameters
    ----------
    Xp : (n, d) standardised feature matrix on GPU

    Returns
    -------
    w  : (d,) EWM weight vector, float64, sums to 1.0
    """
    n, d = Xp.shape

    # Shift to strictly positive domain before normalising
    Xpos    = Xp - Xp.min(axis=0) + 1e-9          # (n, d)
    col_sum = Xpos.sum(axis=0)                     # (d,)

    P       = Xpos / col_sum                       # (n, d) proportional dist.
    log_P   = cp.where(P > 1e-15, cp.log(P), 0.0) # safe log
    e       = -(1.0 / float(cp.log(float(n)))) * (P * log_P).sum(axis=0)  # (d,)

    div = 1.0 - e                                  # (d,)
    w   = div / div.sum()                          # normalise
    return w.astype(cp.float64)


def pad_ewm_weights(w: cp.ndarray, min_len: int = 4) -> cp.ndarray:
    """Ensure the weight vector has at least min_len entries (pad with ε)."""
    if len(w) < min_len:
        pad = cp.full(min_len - len(w), 1e-9, dtype=cp.float64)
        w   = cp.concatenate([w, pad])
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic VIX-linked multiplier  (eq 8)
# ─────────────────────────────────────────────────────────────────────────────

def dynamic_lambda(cfg: DCUDAFAConfig) -> float:
    """
    Eq (8): λ(t) = λ₀ (1 + δ · VIX(t) / VIXref)

    Stressed regimes (VIX(t) > VIXref) drive conservative clustering;
    calm regimes allow diverse risk geometries.
    δ = 0.04, VIXref = 20 (Table I).
    """
    return cfg.lambda0 * (1.0 + cfg.delta * cfg.vix_current / cfg.vix_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio CAR  (eq 6)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_car(
        labels:    cp.ndarray,           # (n,) int32 cluster assignments
        pd_col:    Optional[cp.ndarray], # (n,) PD values or None
        k:         int,
        tier1:     float,
        lgd:       float,
) -> Tuple[float, cp.ndarray]:
    """
    Eq (6): CARport = Tier1 / Σⱼ |Cⱼ| · RWⱼ · EADⱼ

    All operations vectorised on GPU; only the scalar CARport is returned
    to Python.  EADⱼ = cluster size |Cⱼ|; RWⱼ = 12.5 · PD̄ⱼ · LGD (IRB proxy).

    Returns
    -------
    car_scalar : float   portfolio-level CAR
    rw         : (k,)    risk weights per cluster  (used for RCR and ΦCARs)
    """
    k_range = cp.arange(k, dtype=cp.int32)

    # One-hot membership matrix (n, k) — vectorised, no Python loop
    one_hot = (labels[:, None] == k_range[None, :])          # (n, k) bool
    counts  = one_hot.sum(axis=0).astype(cp.float64)         # (k,)

    if pd_col is not None:
        pd_sum = (one_hot.astype(cp.float64) * pd_col[:, None]).sum(axis=0)
        pd_j   = cp.where(counts > 0, pd_sum / cp.maximum(counts, 1.0), 0.05)
    else:
        pd_j = cp.full(k, 0.05, dtype=cp.float64)

    pd_j = cp.clip(pd_j, 0.001, 0.999)
    rw   = 12.5 * pd_j * lgd                                 # (k,)
    ead  = counts                                             # EAD ≈ cluster size

    # Basel III IRB: RWA_j = EAD_j × RW_j  (NOT EAD²×RW — that was a bug)
    # CAR = Tier1 / Σ_j RWA_j
    denom = float((ead * rw).sum())
    car   = (tier1 / denom) if denom > 0.0 else 1.0
    return car, rw


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level Lagrangian penalty  (eq 7)
# ─────────────────────────────────────────────────────────────────────────────

def psi_basel(car: float, lam: float, car_min: float) -> float:
    """
    Eq (7): ΨBasel = λ(t) · max(0, CARmin − CARport)²

    cp.where semantics (branchless predicated execution on GPU).
    Returns 0.0 when portfolio CAR ≥ 8 %, regardless of any single cluster's
    CAR — key regulatory property that permits tail-risk cluster retention.
    """
    shortfall = float(cp.where(
        cp.array(car_min - car) > 0.0,
        cp.array(car_min - car),
        cp.array(0.0)
    ))
    return lam * shortfall ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Complete multi-objective fitness  (eq 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fitness_vector(
        jm_vec:       cp.ndarray,          # (N,) J_M per firefly
        phi_pd_vec:   cp.ndarray,          # (N,) Φ_PD per firefly
        phi_dti_vec:  cp.ndarray,          # (N,) Φ_DTI per firefly
        phi_car_vec:  cp.ndarray,          # (N,) Φ_CAR (soft cluster-level) per firefly
        psi_vec:      cp.ndarray,          # (N,) ΨBasel per firefly
        ewm_w:        cp.ndarray,          # (≥4,) EWM weights
) -> cp.ndarray:
    """
    Eq (9): F = −[w₁·JM + w₂·ΦPD + w₃·ΦDTI + w₄·ΦCAR + ΨBasel]

    Maximising F minimises Mahalanobis TICV + within-cluster PD/DTI variance
    while penalising Basel III shortfall.

    All {wⱼ} are EWM-calibrated (eq 5); ΨBasel is portfolio-level (not
    cluster-level), preserving diverse risk topology.

    Returns
    -------
    fitness : (N,) float64 — higher is better (maximisation problem)
    """
    w = ewm_w[:4] / ewm_w[:4].sum()      # normalise first 4 weights
    weighted = (w[0] * jm_vec
                + w[1] * phi_pd_vec
                + w[2] * phi_dti_vec
                + w[3] * phi_car_vec
                + psi_vec)
    return -weighted                       # negate: solver maximises


# ─────────────────────────────────────────────────────────────────────────────
# Per-firefly CAR batch computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_car_batch(
        Xp:           cp.ndarray,          # (n, d) standardised data (VRAM)
        population:   cp.ndarray,          # (N, k, d) firefly centroids (VRAM)
        sigma_inv:    cp.ndarray,          # (d, d) precision matrix
        pd_col:       Optional[cp.ndarray],
        cfg:          DCUDAFAConfig,
        assign_fn,                         # callable: assign_clusters_gpu
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute ΨBasel and ΦCARs for all N fireflies.

    This is the only Python-level loop remaining in the hot path; it iterates
    over N=256 fireflies, each calling the fast CUDA cluster_assign_kernel.
    Total cost: N × O(n·k·d / 256 threads) — dominated by n=500K, k=5, d=30.

    Returns
    -------
    psi_vec     : (N,) ΨBasel values
    phi_car_vec : (N,) risk-weight variance per firefly (Φ_CAR soft term)
    """
    N   = population.shape[0]
    lam = dynamic_lambda(cfg)

    psi_vec     = cp.empty(N, dtype=cp.float64)
    phi_car_vec = cp.empty(N, dtype=cp.float64)

    for f in range(N):
        labels_f        = assign_fn(Xp, population[f], sigma_inv)
        car_f, rw_f     = portfolio_car(labels_f, pd_col, cfg.k, cfg.tier1, cfg.lgd)
        psi_vec[f]      = psi_basel(car_f, lam, cfg.car_min)
        phi_car_vec[f]  = float(cp.var(rw_f))

    return psi_vec, phi_car_vec


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised portfolio CAR for N fireflies simultaneously  (replaces N-loop)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_car_batch(
        all_labels: cp.ndarray,           # (N, n) int32 — all firefly assignments
        pd_col:     Optional[cp.ndarray], # (n,) PD values
        k:          int,
        tier1:      float,
        lgd:        float,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Vectorised portfolio CAR for all N fireflies in a single CuPy pass.

    Memory: (N, n, k) bool one-hot ≈ 640 MB at N=256, n=500K, k=5 (1 byte/elem).
    As float64: ~5.1 GB peak — comfortably within the A5000's 24 GB GDDR6.

    Returns
    -------
    car_vec : (N,) float64  portfolio-level CAR per firefly
    rw_mat  : (N, k) float64 risk weights per firefly per cluster
    """
    N, n    = all_labels.shape
    k_range = cp.arange(k, dtype=cp.int32)

    # One-hot membership: (N, n, k)  bool → keeps memory at 1 B/elem
    one_hot = (all_labels[:, :, None] == k_range[None, None, :])  # (N, n, k)
    counts  = one_hot.sum(axis=1).astype(cp.float64)               # (N, k)

    if pd_col is not None:
        # Weighted PD sum over borrowers for each (firefly, cluster)
        pd_sum = (one_hot.astype(cp.float64) * pd_col[None, :, None]).sum(axis=1)  # (N, k)
        pd_mat = cp.where(counts > 0, pd_sum / cp.maximum(counts, 1.0), 0.05)      # (N, k)
    else:
        pd_mat = cp.full((N, k), 0.05, dtype=cp.float64)

    pd_mat = cp.clip(pd_mat, 0.001, 0.999)
    rw_mat = 12.5 * pd_mat * lgd                   # (N, k)
    ead    = counts                                  # EAD ≈ cluster size

    # Basel III IRB: CAR = Tier1 / Σⱼ (EADⱼ × RWⱼ)
    denom   = (ead * rw_mat).sum(axis=1)            # (N,)
    car_vec = cp.where(denom > 0.0, tier1 / denom, 1.0)  # (N,)

    return car_vec, rw_mat


def _assign_clusters_cupy_batch(
        Xp:          cp.ndarray,   # (n, d)
        population:  cp.ndarray,   # (N, k, d)
        sigma_inv:   cp.ndarray,   # (d, d)
        batch_size:  int = 8,
) -> cp.ndarray:
    """
    Mini-batched Mahalanobis cluster assignment for all N fireflies using CuPy.

    Processes `batch_size` fireflies at a time to bound peak VRAM:
      memory per batch ≈ n × B × k × d × 8 B
      B=8: 500K × 8 × 5 × 30 × 8 B ≈ 4.8 GB  (safe on 24 GB A5000)

    Avoids the N=256 Python kernel-launch loop in the original compute_car_batch.

    Returns
    -------
    all_labels : (N, n) int32  cluster assignment per firefly per borrower
    """
    N, k, d = population.shape
    n        = Xp.shape[0]
    all_labels = cp.empty((N, n), dtype=cp.int32)

    for b_start in range(0, N, batch_size):
        b_end = min(b_start + batch_size, N)
        B     = b_end - b_start

        # centroids_b: (B, k, d)
        centroids_b = population[b_start:b_end]

        # diff: (n, B, k, d)  —  broadcast without materialising full float64 yet
        diff = (Xp[:, None, None, :] - centroids_b[None, :, :, :])   # (n, B, k, d)

        # Mahalanobis: (x-μ)ᵀ Σ⁻¹ (x-μ)  =  (diff @ sigma_inv) * diff  summed over d
        # (n,B,k,d) @ (d,d) → (n,B,k,d)
        tmp  = diff @ sigma_inv                                         # (n, B, k, d)
        dist = (tmp * diff).sum(axis=-1)                               # (n, B, k)

        # argmin over k → (n, B), transpose → (B, n)
        all_labels[b_start:b_end] = cp.argmin(dist, axis=-1).T.astype(cp.int32)

        # Explicitly free large intermediates to reclaim VRAM immediately
        del diff, tmp, dist

    return all_labels   # (N, n)


def compute_car_batch_vectorized(
        Xp:          cp.ndarray,           # (n, d) standardised data (VRAM)
        population:  cp.ndarray,           # (N, k, d) firefly centroids (VRAM)
        sigma_inv:   cp.ndarray,           # (d, d) precision matrix
        pd_col:      Optional[cp.ndarray],
        cfg:         DCUDAFAConfig,
        assign_batch_size: int = 8,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Fully vectorised replacement for compute_car_batch.

    Instead of N=256 Python iterations each launching a CUDA kernel,
    this function:
      1. Assigns all N×n borrower-cluster memberships via CuPy batched matmul
         (batch_size fireflies at a time → ~32 Python iterations).
      2. Computes portfolio CAR for all N fireflies in a single CuPy reduction.
      3. Assembles Ψ_Basel and Φ_CAR fully on GPU — no D→H scalar transfers
         until the final assignment.

    Speedup over original: ~8-16× fewer Python iterations; larger, more
    cache-friendly CuPy ops better exploit the A5000's 768 GB/s bandwidth.

    Returns
    -------
    psi_vec     : (N,) ΨBasel values
    phi_car_vec : (N,) risk-weight variance per firefly (Φ_CAR soft term)
    """
    lam = dynamic_lambda(cfg)

    # Step 1: batched cluster assignments (N, n) — ~32 Python iters instead of 256
    all_labels = _assign_clusters_cupy_batch(
        Xp, population, sigma_inv, assign_batch_size)

    # Step 2: vectorised CAR for all N fireflies — single CuPy pass
    car_vec, rw_mat = portfolio_car_batch(
        all_labels, pd_col, cfg.k, cfg.tier1, cfg.lgd)

    # Step 3: Ψ_Basel — branchless, fully vectorised (eq 7)
    shortfall = cp.maximum(cfg.car_min - car_vec, 0.0)   # (N,)
    psi_vec   = lam * shortfall ** 2                       # (N,)

    # Step 4: Φ_CAR — risk-weight variance across clusters per firefly (N,)
    phi_car_vec = cp.var(rw_mat, axis=1)

    return psi_vec, phi_car_vec
