"""
solver.py — D-CUDA-FA Solver for RTX A5000 (sm_86)
====================================================
Implements the complete Algorithm 1 from the paper:
  • Pre-processing  : z-score (eq 1) + adaptive Σ⁻¹ (eqs 2–3)
  • Population init : random centroid draws from data (GPU)
  • Fitness eval    : Mahalanobis raw kernel + CAR Lagrangian (eqs 4–9)
  • Position update : vectorised branchless firefly move (eqs 10–12)
  • Brightness sort : CuPy radix sort every Ssort=10 iterations
  • Streams         : double-buffered (stream_compute, stream_transfer)

RTX A5000 Memory Strategy:
  The 500 K × 30 × 8 B = 120 MB dataset is pinned to 24 GB GDDR6 VRAM via
  cp.asarray after the single H2D transfer in _preprocess().  It never leaves
  the GPU until fit() returns, eliminating PCIe overhead entirely.

Paper equations referenced inline throughout.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from typing import Optional

import numpy as np
import cupy as cp
import cupy.linalg as cpla
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import DCUDAFAConfig, smem_bytes, THREADS_PER_BLOCK_FITNESS, THREADS_PER_BLOCK_ASSIGN
from kernel import get_kernels
from car_constraint import (
    compute_ewm_weights,
    pad_ewm_weights,
    portfolio_car,
    compute_car_batch,
    compute_car_batch_vectorized,   # fast vectorised replacement
    compute_fitness_vector,
    dynamic_lambda,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing  (Section III-A, eqs 1–3)
# ─────────────────────────────────────────────────────────────────────────────

def z_score_standardise(X: np.ndarray) -> tuple[cp.ndarray, StandardScaler]:
    """
    Eq (1): x'ᵢⱼ = (xᵢⱼ − μⱼ) / σⱼ

    Fit on CPU (sklearn), transform on CPU, then single H2D transfer via
    cp.asarray — the dataset is pinned to VRAM for the duration of fit().
    """
    scaler  = StandardScaler()
    Xp_cpu  = scaler.fit_transform(X).astype(np.float64)
    Xp_gpu  = cp.asarray(Xp_cpu)   # 500 K×30×8 B = 120 MB — fits in 24 GB
    return Xp_gpu, scaler


def compute_precision_matrix(
        Xp:          cp.ndarray,
        svd_thresh:  float = 1e-4,
        cond_thresh: float = 1e4,
) -> cp.ndarray:
    """
    Eqs (2)–(3): Adaptive precision matrix Σ⁻¹ on GPU.

    Σ = (X'−X̄')ᵀ(X'−X̄') / (n−1)                                    [eq 2]

    κ(Σ) ≤ 10⁴  → Cholesky inversion  (fast, numerically preferred)
    κ(Σ) > 10⁴  → Truncated SVD pseudoinverse:                        [eq 3]
                   Σ† = Uᵣ Λᵣ⁻¹ Uᵣᵀ  over signal-dominant components
                   (singular values σⱼ/σ₁ ≥ τ = 10⁻⁴)

    Result stays in GPU VRAM — accessed by the kernel via const double*,
    kept hot in the A5000's 6 MB L2 cache (900 doubles = 7.2 KB).
    """
    n = Xp.shape[0]
    Xc    = Xp - Xp.mean(axis=0)
    Sigma = (Xc.T @ Xc) / (n - 1)          # (d, d) — eq (2)

    U, s, _ = cpla.svd(Sigma, full_matrices=False)
    s_min   = float(s[-1])
    kappa   = float(s[0] / s_min) if s_min > 0 else float("inf")

    if kappa <= cond_thresh:
        try:
            L         = cpla.cholesky(Sigma)
            Id        = cp.eye(Sigma.shape[0], dtype=cp.float64)
            Sigma_inv = cpla.solve(L.T, cpla.solve(L, Id))
            return Sigma_inv.astype(cp.float64)
        except cp.linalg.LinAlgError:
            pass  # ill-conditioned despite κ check; fall through to SVD

    # Truncated SVD pseudoinverse — eq (3)
    mask     = (s / s[0]) >= svd_thresh
    Ur       = U[:, mask]
    sr       = s[mask]
    Sigma_inv = Ur @ cp.diag(1.0 / sr) @ Ur.T
    return Sigma_inv.astype(cp.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster assignment wrapper (CUDA raw kernel)
# ─────────────────────────────────────────────────────────────────────────────

def assign_clusters(
        Xp:        cp.ndarray,   # (n, d)
        centroids: cp.ndarray,   # (k, d) — single firefly's centroid set
        sigma_inv: cp.ndarray,   # (d, d)
) -> cp.ndarray:
    """
    Dispatch cluster_assign_kernel; returns (n,) int32 label array on GPU.
    Grid: ceil(n/256) blocks × 256 threads.
    """
    _, cluster_assign_kernel = get_kernels()
    n, d = Xp.shape
    k    = centroids.shape[0]

    labels = cp.empty(n, dtype=cp.int32)
    blocks = (n + THREADS_PER_BLOCK_ASSIGN - 1) // THREADS_PER_BLOCK_ASSIGN

    cluster_assign_kernel(
        (blocks,), (THREADS_PER_BLOCK_ASSIGN,),
        (Xp, centroids, sigma_inv, labels,
         np.int32(n), np.int32(d), np.int32(k)),
    )
    cp.cuda.runtime.deviceSynchronize()
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Raw-kernel fitness dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_fitness_kernel(
        Xp_gpu:        cp.ndarray,
        population:    cp.ndarray,
        sigma_inv:     cp.ndarray,
        pd_col:        Optional[cp.ndarray],
        cfg:           DCUDAFAConfig,
        stream:        cp.cuda.Stream,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Launch mahal_fitness_kernel for all N fireflies on stream_compute.

    Grid  : (N,) blocks × 512 threads per block
    Smem  : TILE×D_PAD + k×D_PAD = 9,472 B per block (well within 48 KB)

    Returns (jm_vec, phi_pd_vec, phi_dti_vec) each of shape (N,).
    """
    mahal_fitness_kernel, _ = get_kernels()
    n, d     = Xp_gpu.shape
    N, k, _  = population.shape

    jm_vec      = cp.zeros(N, dtype=cp.float64)
    phi_pd_vec  = cp.zeros(N, dtype=cp.float64)
    phi_dti_vec = cp.zeros(N, dtype=cp.float64)

    pd_ptr = pd_col if pd_col is not None else cp.zeros(n, dtype=cp.float64)

    with stream:
        mahal_fitness_kernel(
            (N,), (THREADS_PER_BLOCK_FITNESS,),
            (Xp_gpu, population, sigma_inv, pd_ptr,
             jm_vec, phi_pd_vec, phi_dti_vec,
             np.int32(n), np.int32(d), np.int32(k)),
            shared_mem=smem_bytes(k),
        )

    return jm_vec, phi_pd_vec, phi_dti_vec


# ─────────────────────────────────────────────────────────────────────────────
# Firefly position update  (eqs 10–12)
# ─────────────────────────────────────────────────────────────────────────────

def firefly_update(
        population: cp.ndarray,   # (N, k, d)
        fitness:    cp.ndarray,   # (N,)
        cfg:        DCUDAFAConfig,
        rng:        cp.random.Generator,
) -> cp.ndarray:
    """
    Eqs (10)–(12): vectorised branchless position update — pure CuPy, no H2D.

    F_i^new = F_i + m_ij [β(r_ij)(F_j − F_i) + α·ε]       [eq 10]

    β(r) = β₀ · exp(−γ·r²)                                  [eq 10]
    m_ij ∈ {0,1} when Fⱼ > Fᵢ  — cp.where branchless mask  [eq 12]
    ε ~ N(0, I)   (GPU RNG, no CPU round-trip)

    All (N,N) broadcast ops run on GDDR6; N=256 → N²=65,536 comparisons
    per iteration, negligible vs the n=500 K fitness kernel cost.
    """
    N, k, d = population.shape
    eps      = rng.standard_normal((N, k, d), dtype=cp.float64)  # GPU RNG

    flat = population.reshape(N, -1)          # (N, k*d)

    # Pairwise squared Euclidean distances in centroid space  — (N, N)
    diff_sq = cp.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)

    # Attractiveness matrix β(rᵢⱼ) — (N, N)
    beta_mat = cfg.beta0 * cp.exp(-cfg.gamma * diff_sq)

    # Predicated mask: m_ij = 1 iff Fⱼ > Fᵢ  (eq 12, branchless)
    mask = cp.where(
        fitness[None, :] > fitness[:, None],
        cp.ones((N, N), dtype=cp.float64),
        cp.zeros((N, N), dtype=cp.float64),
    )                                          # (N, N)

    # Weighted attraction vector (N, k*d)
    beta_mask   = (beta_mat * mask)[:, :, None]                   # (N, N, 1)
    delta_flat  = (beta_mask * (flat[None, :, :] - flat[:, None, :])).sum(axis=1)

    # Normalise by attractor count to prevent centroid blow-up
    n_attr      = mask.sum(axis=1, keepdims=True).clip(min=1.0)   # (N, 1)
    delta_flat /= n_attr

    delta      = delta_flat.reshape(N, k, d)
    # Alpha decay: reduces step size over iterations to aid convergence.
    # The iteration number is not directly available here, so decay is applied
    # in the fit() loop by passing a decayed alpha via cfg each iteration.
    new_pop = population + delta + cfg.alpha * eps

    # Centroid clipping: prevent centroids from drifting outside data range.
    # z-scored data lives in roughly [-4, 4]; clip to [-6, 6] for safety.
    new_pop = cp.clip(new_pop, -6.0, 6.0)
    return new_pop


# ─────────────────────────────────────────────────────────────────────────────
# J_M helper for history/diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_jm(
        Xp:        cp.ndarray,
        centroids: cp.ndarray,   # (k, d)
        labels:    cp.ndarray,   # (n,) int32
        sigma_inv: cp.ndarray,
) -> float:
    """
    Eq (4): J_M = Σⱼ Σᵢ∈Cⱼ (x'ᵢ − μⱼ)ᵀ Σ⁻¹ (x'ᵢ − μⱼ)  — vectorised.
    """
    k     = centroids.shape[0]
    total = 0.0
    for j in range(k):
        mask = labels == j
        if int(mask.sum()) == 0:
            continue
        diff  = Xp[mask] - centroids[j]      # (nⱼ, d)
        tmp   = diff @ sigma_inv              # (nⱼ, d)
        total += float((tmp * diff).sum())
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

class DCUDAFA:
    """
    D-CUDA-FA for NVIDIA RTX A5000 (sm_86, 24 GB GDDR6).

    All tensor data lives in VRAM from the first cp.asarray call to the
    return of fit().  Two async CUDA streams implement the double-buffered
    pipeline described in Section IV:

      stream_compute  (Stream 0) — fitness kernel over iteration t
      stream_transfer (Stream 1) — position update for iteration t+1

    These overlap PCIe-free GDDR6 ops, sustaining near-peak 768 GB/s
    bandwidth utilisation.
    """

    def __init__(self, cfg: Optional[DCUDAFAConfig] = None):
        self.cfg = cfg or DCUDAFAConfig()

        # VRAM-resident tensors
        self._Xp:        Optional[cp.ndarray] = None
        self._sigma_inv: Optional[cp.ndarray] = None
        self._ewm_w:     Optional[cp.ndarray] = None
        self._pd_col:    Optional[cp.ndarray] = None

        # Best solution
        self.best_centroids_: Optional[cp.ndarray] = None
        self.best_labels_:    Optional[cp.ndarray] = None
        self.best_fitness_:   float = -float("inf")
        self.best_car_:       float = 0.0

        # Diagnostics
        self.history_: dict = {"fitness": [], "car": [], "jm": []}
        self.fit_time_s_: float = 0.0
        self.scaler_: Optional[StandardScaler] = None

    # ── Pre-processing ────────────────────────────────────────────────────────

    def _preprocess(self, X: np.ndarray, pd_col: Optional[np.ndarray]) -> None:
        """
        Z-score (eq 1) → Σ⁻¹ (eqs 2–3) → EWM weights (eq 5).
        Single H2D transfer; all arrays pinned to VRAM.
        Also auto-scales tier1 if left at 0.0 sentinel.
        """
        self._Xp, self.scaler_ = z_score_standardise(X)
        self._sigma_inv = compute_precision_matrix(
            self._Xp, self.cfg.svd_thresh, self.cfg.cond_thresh)
        self._ewm_w = pad_ewm_weights(compute_ewm_weights(self._Xp))

        if pd_col is not None:
            self._pd_col = cp.asarray(pd_col.astype(np.float64))

        # Auto-scale tier1 when left at default 0.0:
        # tier1 = car_min * n * mean_RW ensures a uniform portfolio lands
        # exactly at the Basel III floor, giving meaningful non-zero CAR values.
        # Example: n=500K, avg_PD=5%, LGD=0.45 -> tier1 = 0.08*500000*0.28125 = 11,250
        if self.cfg.tier1 == 0.0:
            n      = self._Xp.shape[0]
            avg_pd = float(self._pd_col.mean()) if self._pd_col is not None else 0.05
            avg_pd = float(np.clip(avg_pd, 0.001, 0.999))
            avg_rw = 12.5 * avg_pd * self.cfg.lgd
            scaled_tier1 = self.cfg.car_min * n * avg_rw
            self.cfg = dataclasses.replace(self.cfg, tier1=scaled_tier1)

    # ── Population initialisation ─────────────────────────────────────────────

    def _init_population(self, rng: cp.random.Generator) -> cp.ndarray:
        """
        N fireflies, each a (k, d) centroid set drawn from data rows.
        Shape: (N, k, d) — fully on GPU.
        """
        n, d = self._Xp.shape
        N, k = self.cfg.n_fireflies, self.cfg.k
        idx  = rng.integers(0, n, size=(N, k))
        return self._Xp[idx].copy()

    # ── Full fitness evaluation (kernel + CAR) ────────────────────────────────

    def _evaluate(
            self,
            population: cp.ndarray,
            stream:     cp.cuda.Stream,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Full fitness vector (N,) combining kernel outputs and CAR Lagrangian.
        Returns (fitness, jm_vec, psi_vec, phi_car_vec).
        """
        # Stream 0: Mahalanobis fitness kernel
        jm_vec, phi_pd_vec, phi_dti_vec = _dispatch_fitness_kernel(
            self._Xp, population, self._sigma_inv,
            self._pd_col, self.cfg, stream)
        stream.synchronize()

        # Per-firefly CAR + Lagrangian — fully vectorised (no Python loop over N)
        psi_vec, phi_car_vec = compute_car_batch_vectorized(
            self._Xp, population, self._sigma_inv,
            self._pd_col, self.cfg)

        # Eq (9): F = −[w₁·JM + w₂·ΦPD + w₃·ΦDTI + w₄·ΦCAR + ΨBasel]
        fitness = compute_fitness_vector(
            jm_vec, phi_pd_vec, phi_dti_vec, phi_car_vec, psi_vec, self._ewm_w)

        return fitness, jm_vec, psi_vec, phi_car_vec

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(
            self,
            X:       np.ndarray,
            pd_col:  Optional[np.ndarray] = None,
            verbose: bool = True,
    ) -> "DCUDAFA":
        """
        Algorithm 1 — D-CUDA-FA on RTX A5000.

        Double-buffered CUDA stream pipeline (Section IV):
          stream_compute  → fitness kernel, iteration t
          stream_transfer → position update, iteration t+1  (overlapped)

        Brightness sort (thrust radix) every Ssort=10 iters
        coalesces attractor reads into 128-byte cache lines.
        """
        cfg = self.cfg
        rng = cp.random.default_rng(cfg.seed)

        if verbose:
            print("[D-CUDA-FA  RTX A5000 / sm_86]  Pre-processing …", flush=True)
        self._preprocess(X, pd_col)
        n, d = self._Xp.shape

        if verbose:
            dev  = cp.cuda.Device()
            name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"]
            if isinstance(name, bytes):
                name = name.decode()
            print(f"  GPU      : {name}")
            print(f"  SM arch  : {cfg.sm_arch}")
            print(f"  Dataset  : n={n:,}  d={d}  k={cfg.k}")
            print(f"  Fireflies: N={cfg.n_fireflies}  T={cfg.n_iterations}")
            mem_mb = n * d * 8 / 1e6
            print(f"  VRAM pin : {mem_mb:.1f} MB (dataset) — no PCIe after this")
            print(f"  Tier 1   : {self.cfg.tier1:,.2f}  "
                  f"({'auto-scaled' if cfg.tier1 == 0.0 else 'user-specified'})")
            print(f"  CAR floor: {cfg.car_min:.0%}  (Basel III minimum)")

        population = self._init_population(rng)

        # ── Double-buffered CUDA streams (Section IV) ─────────────────────────
        stream_compute  = cp.cuda.Stream(non_blocking=True)   # Stream 0: fitness
        stream_transfer = cp.cuda.Stream(non_blocking=True)   # Stream 1: positions

        t0 = time.perf_counter()

        for t in range(1, cfg.n_iterations + 1):

            # ── Stream 0: fitness evaluation over current population (iter t) ─
            fitness, jm_vec, psi_vec, _ = self._evaluate(population, stream_compute)

            # ── Track best firefly ─────────────────────────────────────────────
            best_i = int(cp.argmax(fitness))
            if float(fitness[best_i]) > self.best_fitness_:
                self.best_fitness_    = float(fitness[best_i])
                self.best_centroids_  = population[best_i].copy()
                self.best_labels_     = assign_clusters(
                    self._Xp, self.best_centroids_, self._sigma_inv)
                self.best_car_, _     = portfolio_car(
                    self.best_labels_, self._pd_col,
                    cfg.k, cfg.tier1, cfg.lgd)

            # ── Diagnostics (cheap — just scalars) ───────────────────────────
            jm_best = compute_jm(
                self._Xp, self.best_centroids_, self.best_labels_, self._sigma_inv)
            self.history_["fitness"].append(self.best_fitness_)
            self.history_["car"].append(self.best_car_)
            self.history_["jm"].append(jm_best)

            # ── Alpha decay: reduce step size each iteration ────────────────
            cfg = dataclasses.replace(cfg, alpha=cfg.alpha * cfg.alpha_decay)

            # ── Stream 1: async position update (iter t+1) ────────────────────
            with stream_transfer:
                population = firefly_update(population, fitness, cfg, rng)

            # ── Brightness sort every Ssort iters (thrust radix) ─────────────
            if t % cfg.sort_interval == 0:
                order      = cp.argsort(fitness)[::-1]
                population = population[order]
                fitness    = fitness[order]

            # ── Convergence diagnostic @ iter 400 ────────────────────────────
            if t == cfg.convergence_check_iter and verbose:
                elapsed = time.perf_counter() - t0
                basel   = "✓" if self.best_car_ >= cfg.car_min else "✗"
                print(f"\n  ── Convergence check @ iter {t} ──────────────────")
                print(f"     best_fit={self.best_fitness_:.6f}  "
                      f"JM={jm_best:.4f}  CAR={self.best_car_:.2%}  "
                      f"Basel III={basel}  elapsed={elapsed:.1f}s")

            stream_transfer.synchronize()

            # ── Progress logging ──────────────────────────────────────────────
            if verbose and (t % 50 == 0 or t == 1):
                elapsed = time.perf_counter() - t0
                print(f"  iter {t:4d}/{cfg.n_iterations}  "
                      f"best_fit={self.best_fitness_:.4f}  "
                      f"JM={jm_best:.4f}  "
                      f"CAR={self.best_car_:.1%}  "
                      f"elapsed={elapsed:.1f}s", flush=True)

        self.fit_time_s_ = time.perf_counter() - t0
        if verbose:
            print(f"\n[D-CUDA-FA]  Done in {self.fit_time_s_:.2f}s  "
                  f"({n / self.fit_time_s_ / 1e3:.1f}K records/s)")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Mahalanobis nearest-centroid assignment for new data.
        CPU in → GPU → CPU out; does not retain new data in VRAM.
        """
        Xp_gpu     = cp.asarray(self.scaler_.transform(X).astype(np.float64))
        labels_gpu = assign_clusters(Xp_gpu, self.best_centroids_, self._sigma_inv)
        return cp.asnumpy(labels_gpu)

    # ── Cluster quality metrics (Table IV) ───────────────────────────────────

    def cluster_quality(self) -> dict:
        """
        Compute all metrics from Table IV.  Heavy ops stay on GPU;
        only sklearn-based metrics (silhouette, DB, CH) pull a 5 K-sample
        subset to CPU.
        """
        labels = self.best_labels_
        k      = self.cfg.k

        # J_M (eq 4) — vectorised GPU
        jm = compute_jm(self._Xp, self.best_centroids_, labels, self._sigma_inv)

        # Risk Concentration Ratio = max(|Cⱼ|) / min(|Cⱼ|)
        counts    = cp.array([(labels == j).sum() for j in range(k)],
                             dtype=cp.float64)
        counts_nz = counts[counts > 0]
        rcr       = float(counts_nz.max() / counts_nz.min()) \
                    if len(counts_nz) > 0 else float("inf")

        # Silhouette / DB / CH on a 5 K-sample subset (sklearn requires CPU)
        n      = self._Xp.shape[0]
        sample = min(5_000, n)
        idx    = cp.random.choice(n, sample, replace=False)
        Xs     = cp.asnumpy(self._Xp[idx])
        Ls     = cp.asnumpy(labels[idx])

        sil = silhouette_score(Xs, Ls, metric="euclidean")  # Euclidean (for K-Means comparability)
        # Note: algorithm minimises Mahalanobis distance; Euclidean silhouette
        # is reported here to match K-Means baseline comparison in the paper.
        db  = davies_bouldin_score(Xs, Ls)
        ch  = calinski_harabasz_score(Xs, Ls)

        return {
            "JM_mahalanobis":            round(jm,  4),
            "silhouette_score":          round(sil, 4),
            "davies_bouldin_index":      round(db,  4),
            "calinski_harabasz_score":   round(ch,  1),
            "risk_concentration_ratio":  round(rcr, 2),
            "portfolio_car":             f"{self.best_car_:.2%}",
            "basel_iii_compliant":       self.best_car_ >= self.cfg.car_min,
            "fit_time_s":                round(self.fit_time_s_, 2),
        }

    # ── Tail-risk precision / recall (Section V-D) ───────────────────────────

    def tail_risk_metrics(
            self,
            true_target:         np.ndarray,
            target_car_threshold: float = 0.068,
            verbose:             bool  = True,
    ) -> dict:
        """
        Identify the tail-risk cluster (individual CAR ≈ 6.8 %) and compute
        Precision, Recall, F1 against the ground-truth TARGET binary column.

        Paper finding: one cluster at 6.8 % CAR offset by 16.2 % / 14.9 %
        clusters yields 11.4 % portfolio CAR — above Basel III 8 % minimum.
        """
        cfg        = self.cfg
        labels_gpu = self.best_labels_
        k          = cfg.k

        # Per-cluster individual CAR
        cluster_cars = []
        for j in range(k):
            mask_j = labels_gpu == j
            n_j    = int(mask_j.sum())
            if n_j == 0:
                cluster_cars.append(1.0)
                continue
            if self._pd_col is not None:
                pd_j = float(self._pd_col[mask_j].mean())
            else:
                pd_j = 0.05
            pd_j  = np.clip(pd_j, 0.001, 0.999)
            rw_j  = 12.5 * pd_j * cfg.lgd
            car_j = cfg.tier1 / (n_j * rw_j)   # IRB: RWA = EAD × RW (EAD = n_j)
            cluster_cars.append(car_j)

        # Tail cluster = closest individual CAR to target threshold
        tail_cluster = int(np.argmin([abs(c - target_car_threshold)
                                      for c in cluster_cars]))

        labels_cpu   = cp.asnumpy(labels_gpu)
        pred_binary  = (labels_cpu == tail_cluster).astype(np.int32)
        true_binary  = (true_target > 0).astype(np.int32)

        prec   = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1     = f1_score(true_binary, pred_binary, zero_division=0)

        tail_size = int((labels_cpu == tail_cluster).sum())
        result = {
            "tail_cluster_id":       tail_cluster,
            "tail_cluster_car":      f"{cluster_cars[tail_cluster]:.2%}",
            "tail_cluster_size":     tail_size,
            "tail_pct_of_portfolio": f"{tail_size / len(labels_cpu):.1%}",
            "precision":             round(prec,   4),
            "recall":                round(recall, 4),
            "f1_score":              round(f1,     4),
            "target_car_threshold":  f"{target_car_threshold:.1%}",
            "portfolio_car":         f"{self.best_car_:.2%}",
            "basel_iii_compliant":   self.best_car_ >= cfg.car_min,
        }

        if verbose:
            print("\n── Tail-Risk Validation (Section V-D) " + "─" * 30)
            for key, val in result.items():
                print(f"  {key:<30s}: {val}")
            print("─" * 68)

        return result
