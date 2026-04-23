"""
datasets.py — Data Loading Utilities for D-CUDA-FA (RTX A5000, sm_86)
======================================================================
Provides two data sources matching the paper's experimental setup:

  1. make_synthetic_credit  — Gaussian mixture synthetic credit portfolio
                              Used for Table II (scalability) and quick tests
  2. load_home_credit       — Kaggle Home Credit Default Risk dataset
                              Loads application_train.csv from data/ directory
                              Falls back to synthetic if CSV is absent

Usage
-----
  from datasets import make_synthetic_credit, load_home_credit

  # Synthetic (n=500K, d=30, k=5 underlying clusters):
  X, labels_true, pd_col = make_synthetic_credit(n=500_000, d=30, k=5)

  # Home Credit (or synthetic fallback):
  X, pd_col = load_home_credit(data_dir="data/", target_n=500_000)

Paper reference: Section V-A (dataset description) and Table III.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Credit Portfolio  (Section V-A)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_credit(
        n:     int   = 500_000,
        d:     int   = 30,
        k:     int   = 5,
        seed:  int   = 42,
        vix:   float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic credit portfolio of n borrowers with d features and k
    underlying risk clusters — matches the paper's Section V-A Gaussian mixture.

    Risk structure (paper Table III):
      • Cluster 0 (tail risk) : PD ≈ 50%,  30% of portfolio
      • Cluster 1 (mid risk)  : PD ≈ 25%,  25% of portfolio
      • Cluster 2 (low risk)  : PD ≈ 10%,  20% of portfolio
      • Cluster 3 (prime)     : PD ≈  3%,  15% of portfolio
      • Cluster 4 (prime)     : PD ≈  1%,  10% of portfolio

    Features are drawn from shifted Gaussian distributions; feature 0 is the
    credit score proxy (negative correlation with PD), feature 1 is the DTI
    ratio proxy (positive correlation with PD).

    Parameters
    ----------
    n    : number of borrowers
    d    : feature dimensionality (paper uses d=30)
    k    : number of underlying risk clusters (paper uses k=5)
    seed : RNG seed for reproducibility
    vix  : VIX value used to modulate tail-cluster size (stress testing)

    Returns
    -------
    X           : (n, d)  float64 feature matrix  — raw, un-standardised
    labels_true : (n,)    int32   true cluster labels  (for evaluation)
    pd_col      : (n,)    float64 probability-of-default column
    """
    rng = np.random.default_rng(seed)

    # ── Cluster definitions ───────────────────────────────────────────────────
    # VIX stress: at VIX=35, tail cluster grows by +5% at expense of primes
    stress_extra = max(0.0, (vix - 20.0) / 100.0)  # 0 at VIX=20, 0.15 at VIX=35

    weights = np.array([0.30 + stress_extra,
                        0.25,
                        0.20,
                        0.15 - stress_extra / 2,
                        0.10 - stress_extra / 2])
    weights = np.clip(weights, 0.01, 1.0)
    weights /= weights.sum()

    pd_means   = np.array([0.50, 0.25, 0.10, 0.03, 0.01])
    pd_stds    = np.array([0.08, 0.06, 0.04, 0.01, 0.005])

    # Cluster centroids in feature space (d=30)
    # Feature 0: credit score proxy    (high score → low PD)
    # Feature 1: DTI ratio proxy       (high DTI   → high PD)
    # Features 2–29: macroeconomic / demographic covariates
    base_centers = np.zeros((k, d))
    for j in range(k):
        # Credit score: inversely correlated with PD cluster index
        base_centers[j, 0] = -3.0 * j + 6.0          # 6, 3, 0, -3, -6
        # DTI ratio: positively correlated
        base_centers[j, 1] =  2.5 * j - 5.0           # -5, -2.5, 0, 2.5, 5
        # Remaining features: random cluster offsets
        base_centers[j, 2:] = rng.normal(0, 1.5, d - 2)

    # Per-cluster covariance: diagonal + small off-diagonal coupling
    cov_base  = np.eye(d) * 2.0
    cov_base += rng.uniform(0, 0.3, (d, d))
    cov_base  = (cov_base + cov_base.T) / 2 + np.eye(d) * 0.5  # ensure PSD

    # ── Sample borrowers ──────────────────────────────────────────────────────
    counts       = rng.multinomial(n, weights)
    X_parts      = []
    labels_parts = []
    pd_parts     = []

    for j, n_j in enumerate(counts):
        if n_j == 0:
            continue
        # Feature vectors
        X_j = rng.multivariate_normal(base_centers[j], cov_base, size=n_j)
        X_parts.append(X_j)
        labels_parts.append(np.full(n_j, j, dtype=np.int32))

        # PD values ~ Beta(α, β) shaped to target mean
        pd_j_raw = rng.normal(pd_means[j], pd_stds[j], n_j)
        pd_j     = np.clip(pd_j_raw, 0.001, 0.999)
        pd_parts.append(pd_j)

    X           = np.vstack(X_parts).astype(np.float64)
    labels_true = np.concatenate(labels_parts).astype(np.int32)
    pd_col      = np.concatenate(pd_parts).astype(np.float64)

    # Shuffle to avoid any ordering artefacts
    perm = rng.permutation(n)
    return X[perm], labels_true[perm], pd_col[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Home Credit Default Risk Dataset  (Section V-A, Kaggle)
# ─────────────────────────────────────────────────────────────────────────────

# Features selected from application_train.csv — 30 credit-relevant columns
# (matching paper's d=30 after PCA / feature selection described in Section V-A)
_HOME_CREDIT_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
]

_CSV_NAME = "application_train.csv"


def load_home_credit(
        data_dir:  str = "data",
        target_n:  int = 500_000,
        seed:      int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Kaggle Home Credit Default Risk dataset.

    Expects ``application_train.csv`` in ``data_dir``.  If the file is absent,
    falls back to :func:`make_synthetic_credit` with a warning so that the
    experiment runner and benchmarks work out-of-the-box without Kaggle data.

    Preprocessing
    -------------
    • Select the 30 credit-relevant columns matching the paper's feature set.
    • Fill missing values with column medians (robust to extreme outliers).
    • Upsample / downsample to ``target_n`` rows using repeating random draws.
    • Return raw (un-standardised) feature matrix — solver.py z-scores on GPU.

    Parameters
    ----------
    data_dir : directory that contains application_train.csv
    target_n : desired number of borrower records (paper: 500 000)
    seed     : RNG seed for row sampling

    Returns
    -------
    X      : (target_n, 30) float64 feature matrix  (raw, un-standardised)
    pd_col : (target_n,)    float64 probability-of-default column (TARGET column)
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load Home Credit data. "
            "Install with: pip install pandas"
        ) from exc

    csv_path = Path(data_dir) / _CSV_NAME

    if not csv_path.exists():
        warnings.warn(
            f"[datasets] {csv_path} not found — falling back to synthetic data.\n"
            f"  To use the real dataset, download application_train.csv from\n"
            f"  https://www.kaggle.com/c/home-credit-default-risk/data and place\n"
            f"  it in '{data_dir}/'.",
            UserWarning,
            stacklevel=2,
        )
        X, _, pd_col = make_synthetic_credit(n=target_n, d=30, k=5, seed=seed)
        return X, pd_col

    print(f"[datasets] Loading {csv_path} …", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)

    # ── TARGET column (binary default label → used as PD proxy) ──────────────
    if "TARGET" in df.columns:
        target_raw = df["TARGET"].values.astype(np.float64)
    else:
        warnings.warn("[datasets] TARGET column not found; using PD=0.05 for all rows.")
        target_raw = np.full(len(df), 0.05)

    # ── Feature selection ─────────────────────────────────────────────────────
    available = [f for f in _HOME_CREDIT_FEATURES if f in df.columns]
    missing   = [f for f in _HOME_CREDIT_FEATURES if f not in df.columns]
    if missing:
        warnings.warn(
            f"[datasets] {len(missing)} features not in CSV, padding with zeros: "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
        )

    df_feat = df[available].copy()

    # Pad absent columns with 0.0
    for col in missing:
        df_feat[col] = 0.0

    # Reorder to match paper order
    df_feat = df_feat[_HOME_CREDIT_FEATURES]

    # ── Missing value imputation (column medians) ─────────────────────────────
    for col in df_feat.columns:
        if df_feat[col].isna().any():
            median = df_feat[col].median()
            df_feat[col].fillna(median if not np.isnan(median) else 0.0, inplace=True)

    X_raw  = df_feat.values.astype(np.float64)
    pd_raw = target_raw

    n_orig = len(X_raw)
    print(f"[datasets] Loaded {n_orig:,} rows × {X_raw.shape[1]} features.", flush=True)

    # ── Resample to target_n ──────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    if n_orig >= target_n:
        idx = rng.choice(n_orig, size=target_n, replace=False)
    else:
        # Upsample with replacement (augment with small Gaussian noise)
        idx = rng.choice(n_orig, size=target_n, replace=True)
        noise = rng.normal(0, 0.01, (target_n, X_raw.shape[1]))
        X_raw  = X_raw[idx] + noise
        pd_raw = pd_raw[idx]
        print(f"[datasets] Upsampled {n_orig:,} → {target_n:,} with noise augmentation.",
              flush=True)
        return X_raw, pd_raw

    X_out  = X_raw[idx]
    pd_out = pd_raw[idx]
    print(f"[datasets] Sampled {target_n:,} rows from {n_orig:,}.", flush=True)
    return X_out, pd_out
