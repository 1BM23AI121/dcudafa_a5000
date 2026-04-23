"""
ewm.py — Entropy Weight Method (Section III-C, Equation 5) — RTX A5000
=======================================================================
.. deprecated::
   Standalone reference copy.  Production implementation is
   ``car_constraint.compute_ewm_weights()`` which is called by
   ``solver.DCUDAFA._preprocess()``.  Do not import from this module in new code.

Parameterfree, information-theoretic feature weight calibration.
Computed once from training data; logged to regulatory audit trail
(satisfies Basel III model validation + IFRS 9 governance).
"""

from __future__ import annotations
import warnings
warnings.warn(
    "ewm.py is a legacy reference module; use car_constraint.compute_ewm_weights() instead.",
    DeprecationWarning, stacklevel=2,
)
import cupy as cp



def compute_ewm_weights_gpu(Xp_gpu: cp.ndarray) -> cp.ndarray:
    """
    Eq (5): Compute EWM weight vector w of shape (d,) on GPU.

    Assigns higher weight to features with large entropy divergence
    (1 − eⱼ), i.e. features that discriminate most across borrowers.

    Parameters
    ----------
    Xp_gpu : (n, d) standardised feature matrix on VRAM

    Returns
    -------
    w_gpu : (d,) float64 EWM weight vector, Σw = 1
    """
    n, d = Xp_gpu.shape

    # Shift to strictly positive before proportional distribution
    Xpos    = Xp_gpu - Xp_gpu.min(axis=0) + 1e-9   # (n, d) ≥ 1e-9
    col_sum = Xpos.sum(axis=0)                       # (d,)
    P       = Xpos / col_sum                         # (n, d) proportional dist.

    # Shannon entropy per feature (vectorised, no Python loop)
    log_P = cp.where(P > 0.0, cp.log(P), cp.zeros_like(P))
    e     = -(1.0 / cp.log(float(n))) * (P * log_P).sum(axis=0)  # (d,)

    # EWM divergence and normalised weights
    div   = 1.0 - e
    w     = div / div.sum()

    # Pad to at least 4 weights (required by fitness function for w1..w4)
    if w.shape[0] < 4:
        pad = cp.full(4 - w.shape[0], 1e-9, dtype=cp.float64)
        w   = cp.concatenate([w, pad])
        w   = w / w.sum()

    return w.astype(cp.float64)


def log_ewm_weights(w_gpu: cp.ndarray, feature_names: list[str] | None = None) -> None:
    """
    Print EWM weight table to stdout for regulatory audit trail.
    Feature names default to f0..f{d-1} if not provided.
    """
    w_cpu = w_gpu.get()
    d     = len(w_cpu)
    names = feature_names or [f"f{i:02d}" for i in range(d)]
    order = w_cpu.argsort()[::-1]

    print("\n── EWM Feature Weights (Regulatory Audit Log — RTX A5000) ──────────────────")
    print(f"  {'Feature':<35s}  {'Weight':>10s}  {'Rank':>6s}")
    print("  " + "─" * 55)
    for rank, idx in enumerate(order[:d], start=1):
        name = names[idx] if idx < len(names) else f"f{idx:02d}"
        print(f"  {name:<35s}  {w_cpu[idx]:>10.6f}  {rank:>6d}")
    print("──────────────────────────────────────────────────────────────")
