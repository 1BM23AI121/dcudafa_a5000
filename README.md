# D-CUDA-FA — RTX A5000 CuPy Port (`sm_86`)

> **Paper:** *D-CUDA-FA: Mahalanobis-Distance, Portfolio-Constrained, Ampere-Accelerated Firefly Algorithm for Covariance-Aware Tail Risk Detection in High-Dimensional Credit Portfolios*  
> H S Adwi, Hrithik M, Pakam Yasaswini, Om Manish Makadia — BMSCE, Bengaluru

---

## Hardware Target

| Property | RTX 3070 (paper) | RTX A5000 (this port) |
|---|---|---|
| Architecture | Ampere GA104 | Ampere GA102 |
| `SM_ARCH` | `sm_86` | **`sm_86`** |
| CUDA cores | 5,888 | 8,192 |
| VRAM | 8 GB GDDR6 @ 448 GB/s | **24 GB GDDR6 @ 768 GB/s** |
| L2 cache | 4 MB | 6 MB |
| SM count | 46 | 64 |

Both cards share **sm_86** compute capability, so the CUDA PTX is identical. The A5000's larger VRAM eliminates all H2D re-transfer at n=500K (120 MB dataset), and the higher memory bandwidth sustains the bandwidth-bound kernel at higher throughput.

---

## Module Map

```
dcudafa_a5000/
├── config.py          # SM_ARCH="sm_86", DCUDAFAConfig (Table I)
├── kernel.py          # RawKernel CUDA source + lazy compile cache
├── car_constraint.py  # EWM (eq 5), CAR (eq 6), ΨBasel (eqs 7–8), F (eq 9)
├── solver.py          # DCUDAFA class: preprocess → fit → predict → metrics
├── datasets.py        # Synthetic Gaussian mixture + Home Credit loader
└── dcudafa_a5000.py   # Entry point / benchmark harness
```

---

## Installation

```bash
pip install cupy-cuda12x numpy scipy scikit-learn pandas
```

Requires CUDA Toolkit ≥ 12.1 and an Ampere GPU (sm_86).

---

## Quick Start

```bash
# Synthetic benchmark (n=500K, d=30, k=5):
python dcudafa_a5000.py

# With real Home Credit data:
python dcudafa_a5000.py --data-dir data/

# Stress scenario (elevated VIX):
python dcudafa_a5000.py --vix 35

# Table II scalability benchmark:
python dcudafa_a5000.py --benchmark

# Smaller run for testing:
python dcudafa_a5000.py --n 100000 --iterations 100
```

---

## Programmatic API

```python
from config   import DCUDAFAConfig
from solver   import DCUDAFA
from datasets import make_synthetic_credit

X, _, pd_col = make_synthetic_credit(n=500_000, d=30, k=5)

cfg   = DCUDAFAConfig(n_fireflies=256, n_iterations=500, k=5, vix_current=20.0)
model = DCUDAFA(cfg)
model.fit(X, pd_col=pd_col, verbose=True)

print(model.cluster_quality())
labels = model.predict(X_new)
```

---

## Kernel Engineering (Section IV)

### Shared Memory Tiling with Zero Bank Conflicts
```
TILE_ROWS = 32   # rows per pass
D_PAD     = 32   # d=30 padded to 32 (warp width)
smem      = (32×32 + k×32) × 8 B = 9,472 B per block  (< 48 KB/SM limit)
```
Sequential thread indices map to distinct 8-byte banks → **zero bank conflicts**.

### Warp Shuffle Butterfly Reduction (eq 11)
```c
for (int offset = 16; offset >= 1; offset >>= 1) {
    acc_jm += __shfl_down_sync(0xFFFFFFFFu, acc_jm, offset);
}
```
Reduces 32 lanes to lane 0 in 5 steps; cuts `__syncthreads()` calls from 9 → **1 per block**.

### Register Pressure & SM Occupancy
```
-maxrregcount=64  →  65,536 / (512 × 64) = 2 resident blocks/SM
                  →  50% SM occupancy  →  hides 28-cycle GDDR6 latency
```

### Double-Buffered CUDA Streams
```python
stream_compute  = cp.cuda.Stream(non_blocking=True)  # fitness kernel,  iter t
stream_transfer = cp.cuda.Stream(non_blocking=True)  # position update, iter t+1
```
Overlaps computation and data movement; PCIe overhead < 2% after initial pin.

### Predicated Execution (eq 12)
```c
const int better = (d2 < best_d2);   // branchless FSEL on Ampere compiler
best_d2 = better ? d2 : best_d2;
```
Warp divergence < 0.1%.

---

## Equations Implemented

| Eq | Description | Module |
|---|---|---|
| (1) | Z-score standardisation | `solver.py` |
| (2) | Sample covariance Σ | `solver.py` |
| (3) | Truncated SVD pseudoinverse Σ† | `solver.py` |
| (4) | Mahalanobis TICV J_M | `kernel.py` |
| (5) | EWM weight vector | `car_constraint.py` |
| (6) | Portfolio CAR | `car_constraint.py` |
| (7) | Basel III Lagrangian ΨBasel | `car_constraint.py` |
| (8) | Dynamic VIX multiplier λ(t) | `car_constraint.py` |
| (9) | Complete multi-objective fitness F | `car_constraint.py` |
| (10)| Firefly attractiveness β(r) | `solver.py` |
| (11)| Warp shuffle butterfly sum | `kernel.py` (CUDA) |
| (12)| Predicated branchless position update | `kernel.py` (CUDA) |

---

## Expected Results (Table IV equivalent)

| Metric | K-Means | D-CUDA-FA |
|---|---|---|
| J_M (Mahalanobis TICV) | 1.42 | ~1.29 |
| Silhouette Score | 0.64 | ~0.70 |
| Risk Concentration Ratio | 2.3 | ~1.9 |
| Portfolio CAR ≥ 8%? | ✗ (7.9%) | ✓ (~11.4%) |

---

## Roofline Profile

```
Arithmetic intensity  I  = 0.34 FLOP/byte  (bandwidth-bound)
Ridge point       Iridge  ≈ 36 FLOP/byte  (RTX A5000 GDDR6)
Attainable peak    P     ≈ 262 GFLOP/s
GDDR6 utilisation        ≈ 73% of 768 GB/s peak
```

Since I ≪ Iridge, the kernel is **memory-bandwidth bound** — consistent with the 15% FP32 utilisation reported in the paper for the RTX 3070.
