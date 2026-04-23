"""
config.py — D-CUDA-FA Configuration for NVIDIA RTX A5000 (sm_86)
=================================================================
Target hardware: NVIDIA RTX A5000
  • Architecture : Ampere GA102
  • CUDA cores   : 8,192
  • VRAM         : 24 GB GDDR6  (768 GB/s)
  • Peak FP32    : 27.8 TFLOP/s
  • SM count     : 64
  • Warp size     : 32
  • L2 cache      : 6 MB
  • PCIe          : Gen 4 x16

Paper reference: Table I (hyperparameters) + Section IV (kernel config).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Compute Target ────────────────────────────────────────────────────────────
SM_ARCH = "sm_86"           # RTX A5000  — Ampere GA102, compute capability 8.6
CUDA_STD = "c++17"

# ── Kernel launch geometry (Section IV) ──────────────────────────────────────
# One block per firefly (N=256); 512 threads per block = 16 warps.
# Register cap: -maxrregcount=64 → 65,536 / (512 × 64) = 2 resident blocks/SM
# → 50% SM occupancy; hides 28-cycle GDDR6 latency via warp switching.
THREADS_PER_BLOCK_FITNESS  = 512     # Mahalanobis fitness kernel
THREADS_PER_BLOCK_ASSIGN   = 256     # cluster-assignment kernel
TILE_ROWS   = 32                     # shared-memory data tile height
D_PAD       = 32                     # row padding: d=30 → 32 (warp width),
                                     # sequential threads → distinct banks → 0 conflicts
MAX_REG_COUNT = 64                   # register cap for occupancy target

# ── Shared-memory budget per block ───────────────────────────────────────────
# smem = (TILE_ROWS × D_PAD + k × D_PAD) × 8 bytes (float64)
# k=5 → (32*32 + 5*32) × 8 = 1184 × 8 = 9,472 B  — well within 48 KB/SM limit
def smem_bytes(k: int) -> int:
    return (TILE_ROWS * D_PAD + k * D_PAD) * 8


# ── Brightness sort interval ──────────────────────────────────────────────────
# thrust::sort_by_key radix sort every Ssort=10 iters coalesces attractor reads
# into 128-byte cache lines; raises L1 hit rate 41 % → 78 %.
SORT_INTERVAL = 10


@dataclass
class DCUDAFAConfig:
    """
    D-CUDA-FA Hyperparameter Configuration — RTX A5000 / sm_86 edition.

    All values match Table I of the paper unless explicitly noted.
    A5000-specific overrides:
      • SM_ARCH      = sm_86  (vs sm_90 for A800, sm_86 for RTX 3070)
      • VRAM budget  = 24 GB GDDR6  → 500 K × 30 × 8 B = 120 MB (fits easily)
      • tier1 scaled to match RTX 3070 paper baseline
    """
    # ── Firefly Algorithm (Table I) ───────────────────────────────────────────
    n_fireflies:   int   = 256      # N — one thread block per firefly
    n_iterations:  int   = 500      # T — convergence verified by iter ~400
    beta0:         float = 1.0      # β₀ — initial attractiveness
    gamma:         float = 1.0      # γ  — light absorption coefficient
    alpha:         float = 0.01     # Gaussian step-size (was 0.2: caused centroid drift)
    alpha_decay:   float = 0.995    # multiplicative decay per iter: alpha*=alpha_decay each step
    sort_interval: int   = SORT_INTERVAL   # Ssort — brightness sort frequency

    # ── Cluster geometry (Table I) ────────────────────────────────────────────
    k:             int   = 5        # clusters — Basel III IRB buckets

    # ── Numerical stability (Table I) ─────────────────────────────────────────
    svd_thresh:    float = 1e-4     # τ  — SVD truncation threshold
    cond_thresh:   float = 1e4      # κ(Σ) threshold: Cholesky vs pseudoinverse

    # ── Regulatory / Basel III ────────────────────────────────────────────────
    car_min:       float = 0.08     # 8 % Basel III minimum
    tier1:         float = 0.0      # Tier 1 capital (eq 6); 0.0 = auto-scale in fit()
    lgd:           float = 0.45     # LGD — IRB standard

    # ── VIX-linked dynamic Lagrangian (eq 8) ─────────────────────────────────
    lambda0:       float = 0.5      # λ₀
    delta:         float = 0.04     # δ — VIX sensitivity
    vix_ref:       float = 20.0     # VIXref — normalisation baseline
    vix_current:   float = 20.0     # VIX(t) — override for stress scenarios

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed:          int   = 42

    # ── Convergence diagnostics ───────────────────────────────────────────────
    convergence_check_iter: int = 400   # paper: convergence by iter 400

    # ── EWM weights (populated during preprocessing) ─────────────────────────
    ewm_weights: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Hardware ──────────────────────────────────────────────────────────────
    sm_arch: str = SM_ARCH
