"""
fitness.py — Complete Multi-Objective Fitness (Section III-E, Equation 9)
=========================================================================
.. deprecated::
   Standalone reference copy not used by the production solver.
   The active code path is ``solver.DCUDAFA._evaluate()`` which calls
   ``car_constraint.compute_car_batch_vectorized()`` (no Python N-loop).
   Do not import from this module in new code.

Equation (9):
    F = −[w₁·J_M + w₂·Φ_PD + w₃·Φ_DTI + w₄·Φ_CAR + Ψ_Basel]

where:
  J_M     — Mahalanobis TICV (eq 4)         ← from raw CUDA kernel
  Φ_PD    — within-cluster PD variance
  Φ_DTI   — within-cluster DTI variance
  Φ_CAR   — soft cluster-level CAR variance  (interpretability)
  Ψ_Basel — portfolio-level CAR Lagrangian   (regulatory hard constraint)
  w₁..w₄  — EWM-calibrated weights (eq 5)

All ops are pure CuPy (GPU); CAR constraint leverages car_constraint.py.

RTX A5000 note:
  Identical sm_86 compute capability to the RTX 3070 paper baseline.
  Higher GDDR6 bandwidth (768 GB/s) accelerates the memory-bound kernel;
  double-buffered streams overlap fitness (stream_compute) and position
  update (stream_transfer) as in the A800 port.
"""

from __future__ import annotations
import warnings
warnings.warn(
    "fitness.py is a legacy reference module; use solver.DCUDAFA._evaluate() instead.",
    DeprecationWarning, stacklevel=2,
)
from typing import Optional

import numpy as np
import cupy as cp

from kernel import get_kernels
from config import smem_bytes, THREADS_PER_BLOCK_FITNESS, THREADS_PER_BLOCK_ASSIGN
from car_constraint import (
    compute_car_batch,
    compute_fitness_vector,
    portfolio_car,
    dynamic_lambda,
    psi_basel,
)


def launch_mahal_fitness(
        Xp_gpu:         cp.ndarray,        # (n, d) — stays in VRAM
        population_gpu: cp.ndarray,        # (N, k, d) — all firefly centroids
        sigma_inv_gpu:  cp.ndarray,        # (d, d) precision matrix
        pd_col_gpu:     Optional[cp.ndarray],
        stream_compute: cp.cuda.Stream,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Launch mahal_fitness_kernel for all N fireflies.

    Grid  : (N,) blocks × 512 threads per block
    Smem  : TILE×D_PAD + k×D_PAD = 9,472 B per block (< 48 KB/SM limit)

    Returns (jm_vec, phi_pd_vec, phi_dti_vec) each of shape (N,).
    """
    mahal_fitness_kernel, _ = get_kernels()
    n, d    = Xp_gpu.shape
    N, k, _ = population_gpu.shape

    jm_vec      = cp.zeros(N, dtype=cp.float64)
    phi_pd_vec  = cp.zeros(N, dtype=cp.float64)
    phi_dti_vec = cp.zeros(N, dtype=cp.float64)

    pd_ptr = pd_col_gpu if pd_col_gpu is not None else cp.zeros(n, dtype=cp.float64)

    with stream_compute:
        mahal_fitness_kernel(
            (N,), (THREADS_PER_BLOCK_FITNESS,),
            (Xp_gpu, population_gpu, sigma_inv_gpu, pd_ptr,
             jm_vec, phi_pd_vec, phi_dti_vec,
             np.int32(n), np.int32(d), np.int32(k)),
            shared_mem=smem_bytes(k),
        )

    return jm_vec, phi_pd_vec, phi_dti_vec


def launch_cluster_assign(
        Xp_gpu:    cp.ndarray,   # (n, d)
        centroids: cp.ndarray,   # (k, d) single firefly's centroid set
        sigma_inv: cp.ndarray,   # (d, d)
) -> cp.ndarray:
    """
    Dispatch cluster_assign_kernel; returns (n,) int32 label array on GPU.
    Grid: ceil(n/256) blocks × 256 threads.
    """
    _, cluster_assign_kernel = get_kernels()
    n, d = Xp_gpu.shape
    k    = centroids.shape[0]

    labels = cp.empty(n, dtype=cp.int32)
    blocks = (n + THREADS_PER_BLOCK_ASSIGN - 1) // THREADS_PER_BLOCK_ASSIGN

    cluster_assign_kernel(
        (blocks,), (THREADS_PER_BLOCK_ASSIGN,),
        (Xp_gpu, centroids, sigma_inv, labels,
         np.int32(n), np.int32(d), np.int32(k)),
    )
    cp.cuda.runtime.deviceSynchronize()
    return labels


def evaluate_population(
        Xp_gpu:          cp.ndarray,        # (n, d) — stays in VRAM
        population_gpu:  cp.ndarray,        # (N, k, d) — all firefly centroids
        sigma_inv_gpu:   cp.ndarray,        # (d, d) precision matrix
        ewm_w_gpu:       cp.ndarray,        # (≥4,) EWM weight vector
        pd_col_gpu:      Optional[cp.ndarray],
        cfg,                                # DCUDAFAConfig
        stream_compute:  cp.cuda.Stream,
        stream_transfer: cp.cuda.Stream
) -> tuple[cp.ndarray, list[float]]:
    """
    Evaluate multi-objective fitness (eq 9) for all N fireflies in parallel.

    Returns
    -------
    fitness_gpu : (N,) float64 on GPU  — higher = better partition
    cars        : list[float]          — per-firefly portfolio CAR (for logging)
    """
    N = population_gpu.shape[0]
    k = cfg.k

    # ── Raw kernel: J_M, partial Φ_PD, Φ_DTI for all N fireflies ────────────
    out_jm, out_phi_pd, out_phi_dti = launch_mahal_fitness(
        Xp_gpu, population_gpu, sigma_inv_gpu, pd_col_gpu, stream_compute
    )
    stream_compute.synchronize()

    # ── Per-firefly CAR constraint (Ψ_Basel) + Φ_CAR ─────────────────────────
    psi_vec, phi_car_vec = compute_car_batch(
        Xp_gpu, population_gpu, sigma_inv_gpu,
        pd_col_gpu, cfg, launch_cluster_assign
    )

    # Collect per-firefly CARs for logging
    cars = []
    for f in range(N):
        labels_f    = launch_cluster_assign(Xp_gpu, population_gpu[f], sigma_inv_gpu)
        car_f, _    = portfolio_car(labels_f, pd_col_gpu, k, cfg.tier1, cfg.lgd)
        cars.append(car_f)

    # ── Eq (9) assembly ───────────────────────────────────────────────────────
    fitness_gpu = compute_fitness_vector(
        out_jm, out_phi_pd, out_phi_dti, phi_car_vec, psi_vec, ewm_w_gpu
    )
    return fitness_gpu, cars
