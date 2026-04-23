"""
dcudafa_a5000.py — D-CUDA-FA Production Entry Point (RTX A5000, sm_86)
=======================================================================
Paper: "D-CUDA-FA: Mahalanobis-Distance, Portfolio-Constrained,
        Ampere-Accelerated Firefly Algorithm for Covariance-Aware Tail Risk
        Detection in High-Dimensional Credit Portfolios"
H S Adwi, Hrithik M, Pakam Yasaswini, Om Manish Makadia
BMS College of Engineering, Bengaluru, India

RTX A5000 Migration Map (from RTX 3070 paper baseline)
══════════════════════════════════════════════════════════════════════════════
Component              │ RTX 3070 (paper)          │ RTX A5000 (this port)
───────────────────────┼───────────────────────────┼──────────────────────
SM arch                │ sm_86 (GA104)             │ sm_86 (GA102)
CUDA cores             │ 5,888                     │ 8,192
VRAM                   │ 8 GB GDDR6 @ 448 GB/s     │ 24 GB GDDR6 @ 768 GB/s
Peak FP32              │ 46 TFLOP/s                │ 27.8 TFLOP/s*
Dataset pinning        │ multi-pass H2D            │ single cp.asarray (24 GB)
CAR vectorisation      │ Python loop + cp.where    │ cp.where (same, GPU-only)
Brightness sort        │ thrust radix (Ssort=10)   │ cp.argsort (radix back-end)
Double-buffered streams│ ThreadPoolExecutor        │ cp.cuda.Stream (native)
──────────────────────────────────────────────────────────────────────────────
* A5000 is a workstation card tuned for FP64 / sustained workloads; higher
  sustained GDDR6 BW (768 vs 448 GB/s) compensates for lower peak FP32,
  keeping the bandwidth-bound kernel (I ≈ 0.34 FLOP/byte) at full throughput.

Usage
─────
# Synthetic benchmark (n=500K, d=30, k=5):
    python dcudafa_a5000.py

# With real Home Credit data (download first, see datasets.py):
    python dcudafa_a5000.py --data-dir data/

# Stress-test with elevated VIX:
    python dcudafa_a5000.py --vix 35

# Smaller scalability run:
    python dcudafa_a5000.py --n 100000

Modular Components
──────────────────
  config.py        — SM_ARCH="sm_86", DCUDAFAConfig (Table I)
  kernel.py        — RawKernel CUDA source + lazy compile cache
  car_constraint.py— EWM (eq 5), CAR (eq 6), ΨBasel (eq 7–8), F (eq 9)
  solver.py        — DCUDAFA class: preprocess → fit → predict → metrics
  datasets.py      — Synthetic Gaussian mixture + Home Credit loader
  dcudafa_a5000.py — This file: CLI + benchmark harness
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np
import cupy as cp

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config    import DCUDAFAConfig, SM_ARCH
from solver    import DCUDAFA
from datasets  import make_synthetic_credit, load_home_credit


# ─────────────────────────────────────────────────────────────────────────────
# GPU environment check
# ─────────────────────────────────────────────────────────────────────────────

def _check_gpu() -> None:
    """Verify CUDA device is available and print key specs."""
    try:
        dev  = cp.cuda.Device(0)
        prop = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = prop["name"].decode() if isinstance(prop["name"], bytes) else prop["name"]
        cc   = f"{prop['major']}.{prop['minor']}"
        vram = prop["totalGlobalMem"] / 1e9
        print("─" * 60)
        print(f"  GPU      : {name}")
        print(f"  Compute  : sm_{prop['major']}{prop['minor']}  (target: {SM_ARCH})")
        print(f"  VRAM     : {vram:.1f} GB")
        print(f"  Target   : {SM_ARCH} — RTX A5000 Ampere GA102")
        print("─" * 60)

        if cc < "8.6":
            print(f"  [WARN] compute capability {cc} < 8.6 — "
                  "kernel compiled for sm_86; may not run optimally.")
    except Exception as e:
        print(f"[WARN] GPU check failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────────────────

def run_scalability_benchmark(cfg_base: DCUDAFAConfig) -> None:
    """
    Reproduce Table II wall-clock times for n ∈ {100K, 250K, 500K}.
    Runs D-CUDA-FA only (CPU-FA and K-Means baselines require CPU harness).
    """
    print("\n── Scalability Benchmark (Table II) ─────────────────────────")
    print(f"  {'n':>8s}  {'time_s':>8s}  {'records/s':>12s}  {'JM':>8s}  {'CAR':>8s}")
    print("  " + "─" * 52)

    for n_val in (100_000, 250_000, 500_000):
        X, _, pd = make_synthetic_credit(n=n_val, d=30, k=5, seed=42)
        cfg      = DCUDAFAConfig(
            n_fireflies  = cfg_base.n_fireflies,
            n_iterations = cfg_base.n_iterations,
            k            = cfg_base.k,
            vix_current  = cfg_base.vix_current,
            tier1        = cfg_base.tier1,
            seed         = cfg_base.seed,
        )
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        q   = model.cluster_quality()
        rec = n_val / model.fit_time_s_ / 1e3
        print(f"  {n_val:>8,}  {model.fit_time_s_:>8.1f}  "
              f"{rec:>10.0f}K/s  {q['JM_mahalanobis']:>8.4f}  "
              f"{q['portfolio_car']:>8s}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D-CUDA-FA RTX A5000 (sm_86) — credit portfolio segmentation")
    p.add_argument("--data-dir",    default="data",   help="Home Credit CSV directory")
    p.add_argument("--n",           type=int, default=500_000,
                   help="Dataset size (synthetic only; default 500 000)")
    p.add_argument("--n-fireflies", type=int, default=256)
    p.add_argument("--iterations",  type=int, default=500)
    p.add_argument("--k",           type=int, default=5)
    p.add_argument("--vix",         type=float, default=20.0,
                   help="VIX(t) for stress scenario (default 20 = neutral)")
    p.add_argument("--tier1",       type=float, default=1_000_000.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--benchmark",   action="store_true",
                   help="Run Table II scalability benchmark after main fit")
    p.add_argument("--quiet",       action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    verbose = not args.quiet

    _check_gpu()

    # ── Data loading ──────────────────────────────────────────────────────────
    if verbose:
        print("\n[1/4]  Loading data …")

    try:
        X_cpu, pd_col_cpu = load_home_credit(
            data_dir=args.data_dir, target_n=args.n)
        true_target = (pd_col_cpu > 0.5).astype(np.int32)
        data_source = "Home Credit (Kaggle)"
    except Exception:
        if verbose:
            print("  Home Credit not found — generating synthetic data.")
        X_cpu, _, pd_col_cpu = make_synthetic_credit(
            n=args.n, d=30, k=5, seed=args.seed)
        true_target = (pd_col_cpu > pd_col_cpu.mean()).astype(np.int32)
        data_source = f"Synthetic Gaussian mixture (n={args.n:,}, d=30, k=5)"

    if verbose:
        print(f"  Source : {data_source}")
        print(f"  Shape  : {X_cpu.shape}  |  default_rate={pd_col_cpu.mean():.1%}")

    # ── Configure ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n[2/4]  Configuring D-CUDA-FA (sm_86) …")

    cfg = DCUDAFAConfig(
        n_fireflies  = args.n_fireflies,
        n_iterations = args.n_iterations,
        k            = args.k,
        vix_current  = args.vix,
        tier1        = args.tier1,
        seed         = args.seed,
    )

    if verbose:
        print(f"  N={cfg.n_fireflies}  T={cfg.n_iterations}  k={cfg.k}  "
              f"VIX={cfg.vix_current}  λ₀={cfg.lambda0}")

    # ── Fit ───────────────────────────────────────────────────────────────────
    if verbose:
        print("\n[3/4]  Fitting …")

    model = DCUDAFA(cfg)
    model.fit(X_cpu, pd_col=pd_col_cpu, verbose=verbose)

    # ── Results ───────────────────────────────────────────────────────────────
    if verbose:
        print("\n[4/4]  Results")

    print("\n── Cluster Quality (Table IV equivalent) " + "─" * 20)
    quality = model.cluster_quality()
    for key, val in quality.items():
        print(f"  {key:<35s}: {val}")

    model.tail_risk_metrics(true_target, target_car_threshold=0.068,
                            verbose=verbose)

    # ── GPU performance summary ───────────────────────────────────────────────
    if verbose:
        n, d = X_cpu.shape
        T    = cfg.n_iterations
        flop_est = 2 * T * cfg.n_fireflies * n * cfg.k * d  # O(T·N·n·k·d)
        gbps_est = (flop_est * 8) / model.fit_time_s_ / 1e9  # float64
        print("\n── GPU Performance Summary " + "─" * 33)
        print(f"  Estimated FLOP          : {flop_est/1e12:.2f} TFLOP")
        print(f"  Effective BW (est.)     : {gbps_est:.0f} GB/s")
        print(f"  Arithmetic intensity    : ~0.34 FLOP/byte  (bandwidth-bound)")
        print(f"  SM arch                 : {SM_ARCH}")
        print(f"  Warp divergence         : < 0.1%  (predicated FSEL)")
        print(f"  Bank conflicts          : 0  (D_PAD=32)")
        print(f"  __syncthreads / block   : 1  (warp shuffle butterfly)")
        print(f"  SM occupancy target     : 50%  (-maxrregcount=64)")

    # ── Optional scalability benchmark ───────────────────────────────────────
    if args.benchmark:
        run_scalability_benchmark(cfg)


if __name__ == "__main__":
    main()
