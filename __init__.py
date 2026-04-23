"""
D-CUDA-FA — RTX A5000 / sm_86 CuPy Port
=========================================
Mahalanobis-Distance, Portfolio-Constrained, Ampere-Accelerated
Firefly Algorithm for Covariance-Aware Tail Risk Detection
in High-Dimensional Credit Portfolios.

Paper: H S Adwi, Hrithik M, Pakam Yasaswini, Om Manish Makadia
       B.M.S. College of Engineering, Bengaluru, 2024.

Hardware Target: NVIDIA RTX A5000 (Ampere GA102, sm_86, 24 GB GDDR6)

Public API
----------
    from solver     import DCUDAFA
    from config     import DCUDAFAConfig
    from datasets   import load_home_credit, make_synthetic_credit
    from validation import tail_risk_precision_recall, cluster_risk_profile
"""

from config     import DCUDAFAConfig, SM_ARCH
from solver     import DCUDAFA
from datasets   import load_home_credit, make_synthetic_credit
from validation import tail_risk_precision_recall, cluster_risk_profile, plot_convergence

__version__ = "1.0.0"
__hardware__ = "RTX A5000 sm_86"

__all__ = [
    "DCUDAFA",
    "DCUDAFAConfig",
    "SM_ARCH",
    "load_home_credit",
    "make_synthetic_credit",
    "tail_risk_precision_recall",
    "cluster_risk_profile",
    "plot_convergence",
]
