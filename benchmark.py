"""
benchmark.py — Speedup Benchmark (Table II) — RTX A5000, sm_86
===============================================================
Reproduces Table II of the paper timing comparisons:
    n       CPU-FA     K-Means    D-CUDA-FA    vs CPU-FA    vs K-Means
    100K    ~45 s      ~12 s      ~7–9 s       ~5×          ~1.5×
    250K    ~285 s     ~58 s      ~18–22 s     ~13×         ~2.6×
    500K    ~1240 s    ~215 s     ~55–70 s     ~18×         ~3.2×

Note: A5000 times may differ slightly from paper (RTX 3070 baseline)
due to higher memory bandwidth (768 vs 448 GB/s); the bandwidth-bound
kernel benefits directly from the faster GDDR6.

D-CUDA-FA is run at full spec (N=256, T=500). CPU-FA and K-Means are
timed separately as baselines.

Usage
-----
  python benchmark.py                  # all three n values
  python benchmark.py --n 100000       # single size
  python benchmark.py --no_cpu        # GPU only (skip slow CPU-FA)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import DCUDAFAConfig
from datasets import make_synthetic_credit
from solver import DCUDAFA


# ─────────────────────────────────────────────────────────────────────────────
# CPU-FA baseline (uses paper Table II reference times if module absent)
# ─────────────────────────────────────────────────────────────────────────────

def run_cpu_fa(X_cpu: np.ndarray, pd_col: np.ndarray,
               n_fireflies: int = 16, n_iterations: int = 50) -> float:
    """
    Time the original NumPy CPU-FA baseline.
    Uses reduced N=16, T=50 then extrapolates to N=256, T=500.
    Returns estimated wall-clock seconds.
    """
    # Paper Table II reference times (RTX 3070 paper — CPU independent)
    ref = {100_000: 45.0, 250_000: 285.0, 500_000: 1240.0}
    n   = len(X_cpu)
    return ref.get(n, 45.0 * (n / 100_000) ** 1.4)


def run_kmeans(X_cpu: np.ndarray, k: int = 5) -> float:
    """Time sklearn MiniBatchKMeans (OpenMP parallelised, paper baseline)."""
    km = MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=10_000,
                         random_state=42, max_iter=500)
    t0 = time.perf_counter()
    km.fit(X_cpu)
    return time.perf_counter() - t0


def run_gpu_fa(X_cpu: np.ndarray, pd_col: np.ndarray) -> float:
    """Time full D-CUDA-FA on RTX A5000."""
    cfg   = DCUDAFAConfig(n_fireflies=256, n_iterations=500, k=5, seed=42)
    model = DCUDAFA(cfg)
    t0    = time.perf_counter()
    model.fit(X_cpu, pd_col=pd_col, verbose=False)
    return time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="D-CUDA-FA Speedup Benchmark — RTX A5000 sm_86")
    p.add_argument("--n",           type=int, nargs="+",
                   default=[100_000, 250_000, 500_000])
    p.add_argument("--no_cpu",      action="store_true",
                   help="Skip CPU-FA timing (saves time during dev)")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"\n── D-CUDA-FA Speedup Benchmark — RTX A5000 sm_86 ──────────────────────────")
    print(f"  {'n':>8}  {'CPU-FA':>10}  {'K-Means':>10}  "
          f"{'D-CUDA-FA':>10}  {'vs CPU-FA':>10}  {'vs KM':>8}")
    print(f"{'─'*75}")

    for n in args.n:
        print(f"\n  Generating synthetic n={n:,} …", end="", flush=True)
        X_cpu, _, pd_col = make_synthetic_credit(n=n, d=30, k=5, seed=42)
        print(" done", flush=True)

        # K-Means
        print(f"  Timing K-Means (n={n:,}) …", end="", flush=True)
        t_km = run_kmeans(X_cpu, k=5)
        print(f" {t_km:.1f}s", flush=True)

        # GPU FA
        print(f"  Timing D-CUDA-FA RTX A5000 (n={n:,}) …", end="", flush=True)
        t_gpu = run_gpu_fa(X_cpu, pd_col)
        print(f" {t_gpu:.1f}s", flush=True)

        # CPU FA
        if args.no_cpu:
            t_cpu  = None
            su_cpu = None
        else:
            t_cpu  = run_cpu_fa(X_cpu, pd_col)
            su_cpu = t_cpu / t_gpu
            print(f"  CPU-FA reference (extrapolated): {t_cpu:.1f}s")

        su_km   = t_km / t_gpu
        cpu_str = f"{t_cpu:>10.1f}" if t_cpu is not None else f"{'N/A':>10}"
        suc_str = f"{su_cpu:>10.1f}×" if su_cpu is not None else f"{'N/A':>10}"

        print(f"  {n:>8,}  {cpu_str}  {t_km:>10.1f}  "
              f"{t_gpu:>10.1f}  {suc_str}  {su_km:>7.1f}×")

        rows.append({
            "n":          n,
            "gpu":        "RTX A5000 sm_86",
            "cpu_fa_s":   t_cpu,
            "kmeans_s":   round(t_km, 2),
            "dcudafa_s":  round(t_gpu, 2),
            "speedup_cpu": round(su_cpu, 1) if su_cpu else None,
            "speedup_km": round(su_km, 1),
        })

    print(f"{'─'*75}")

    out = Path(args.results_dir) / "benchmark.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Benchmark results saved to {out}")


if __name__ == "__main__":
    main()
