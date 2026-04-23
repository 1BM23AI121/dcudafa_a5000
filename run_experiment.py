"""
run_experiment.py — End-to-End Experiment Runner (RTX A5000, sm_86)
====================================================================
Reproduces the results in Tables II–IV of the paper on RTX A5000 sm_86.

Usage
-----
  python run_experiment.py                    # paper-spec run (n=500K)
  python run_experiment.py --synthetic        # force synthetic data
  python run_experiment.py --vix 35.0        # stress scenario
  python run_experiment.py --n 100000        # scalability sweep
  python run_experiment.py --runs 5          # multiple independent runs
  python run_experiment.py --data_dir data/  # Home Credit CSV path
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import DCUDAFAConfig
from datasets import load_home_credit, make_synthetic_credit
from solver import DCUDAFA
from validation import (tail_risk_precision_recall,
                        cluster_risk_profile,
                        plot_convergence)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D-CUDA-FA RTX A5000 Experiment Runner")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic data (ignore Kaggle CSV)")
    p.add_argument("--data_dir",     default="data",
                   help="Directory containing application_train.csv")
    p.add_argument("--n",            type=int,   default=500_000,
                   help="Number of borrower records (paper: 500000)")
    p.add_argument("--n_fireflies",  type=int,   default=256)
    p.add_argument("--n_iterations", type=int,   default=500)
    p.add_argument("--k",            type=int,   default=5)
    p.add_argument("--vix",          type=float, default=20.0,
                   help="VIX(t) override (20=neutral, 35=stress)")
    p.add_argument("--runs",         type=int,   default=1,
                   help="Number of independent runs (paper uses 5)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--no_plot",      action="store_true")
    return p.parse_args()


def run_once(
        X_cpu:       np.ndarray,
        pd_col:      np.ndarray,
        cfg:         DCUDAFAConfig,
        run_id:      int,
        results_dir: Path,
        no_plot:     bool = False,
) -> dict:
    """Run one complete D-CUDA-FA fit and return result dict."""
    import cupy as cp

    print(f"\n{'='*70}")
    print(f"  RUN {run_id}  —  n={len(X_cpu):,}  k={cfg.k}  "
          f"N={cfg.n_fireflies}  T={cfg.n_iterations}  VIX={cfg.vix_current}")
    print(f"  Hardware: RTX A5000 sm_86 / 24 GB GDDR6")
    print(f"{'='*70}")

    model = DCUDAFA(cfg)
    model.fit(X_cpu, pd_col=pd_col, verbose=True)

    # ── Table IV metrics ──────────────────────────────────────────────────────
    quality = model.cluster_quality()
    print("\n── Cluster Quality (Table IV) ───────────────────────────────────")
    for key, val in quality.items():
        print(f"  {key:<35s}: {val}")

    # ── Per-cluster risk profile ──────────────────────────────────────────────
    cluster_risk_profile(
        model.best_labels_, model._pd_col,
        cfg.k, cfg.tier1, cfg.lgd, cfg.car_min
    )

    # ── Tail-risk validation hook ─────────────────────────────────────────────
    true_target = (pd_col > 0.5).astype(np.int32)   # binary for synthetic
    val_result  = tail_risk_precision_recall(
        model.best_labels_, model._pd_col,
        true_target, cfg.k, cfg.tier1, cfg.lgd,
        target_car_threshold=0.068,
    )

    # ── Convergence plot ──────────────────────────────────────────────────────
    if not no_plot:
        plot_path = str(results_dir / f"convergence_run{run_id}.png")
        plot_convergence(model.history_, plot_path)

    result = {
        "run_id":     run_id,
        "n":          len(X_cpu),
        "k":          cfg.k,
        "N":          cfg.n_fireflies,
        "T":          cfg.n_iterations,
        "vix":        cfg.vix_current,
        "fit_time_s": model.fit_time_s_,
        "gpu":        "RTX A5000 sm_86",
        **quality,
        **{f"val_{kk}": v for kk, v in val_result.items()},
    }

    return result


def main() -> None:
    args        = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        print(f"[main] Generating synthetic data n={args.n:,} …")
        X_cpu, _, pd_col = make_synthetic_credit(n=args.n, d=30, k=5, seed=args.seed)
    else:
        X_cpu, pd_col = load_home_credit(data_dir=args.data_dir, target_n=args.n)

    # ── Config ────────────────────────────────────────────────────────────────
    base_cfg = DCUDAFAConfig(
        n_fireflies  = args.n_fireflies,
        n_iterations = args.n_iterations,
        k            = args.k,
        vix_current  = args.vix,
        seed         = args.seed,
    )

    # ── Multiple runs ─────────────────────────────────────────────────────────
    all_results = []
    for run_id in range(1, args.runs + 1):
        import dataclasses
        cfg = dataclasses.replace(
            base_cfg,
            seed        = args.seed + run_id - 1,
            ewm_weights = None,
        )
        result = run_once(X_cpu, pd_col, cfg, run_id, results_dir, args.no_plot)
        all_results.append(result)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {args.runs} run(s) on RTX A5000 sm_86")
    print(f"{'='*70}")

    jm_vals   = [r["JM_mahalanobis"]          for r in all_results if "JM_mahalanobis" in r]
    car_vals  = [float(r["portfolio_car"].strip("%")) / 100 for r in all_results]
    rcr_vals  = [r["risk_concentration_ratio"] for r in all_results]
    time_vals = [r["fit_time_s"]               for r in all_results]

    if jm_vals:
        print(f"  JM      mean={np.mean(jm_vals):.4f}  std={np.std(jm_vals):.4f}")
    print(f"  CAR     mean={np.mean(car_vals):.2%}  std={np.std(car_vals):.3%}")
    print(f"  RCR     mean={np.mean(rcr_vals):.2f}   std={np.std(rcr_vals):.2f}")
    print(f"  Time(s) mean={np.mean(time_vals):.1f}   std={np.std(time_vals):.1f}")
    compliant = sum(1 for r in all_results if r.get("basel_iii_compliant"))
    print(f"  Basel III: {compliant}/{args.runs} runs compliant")

    # ── Save JSON results ─────────────────────────────────────────────────────
    out_path = results_dir / f"experiment_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
