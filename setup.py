"""
setup.py — Editable install for D-CUDA-FA RTX A5000

Install (editable, for development):
    pip install -e .

Install (production):
    pip install .
"""

from setuptools import setup, find_packages

setup(
    name            = "dcudafa-a5000",
    version         = "1.0.0",
    description     = "D-CUDA-FA: CuPy RTX A5000 Port — Mahalanobis-Distance "
                      "Portfolio-Constrained Firefly Algorithm for Credit Risk "
                      "(sm_86, 24 GB GDDR6)",
    author          = "H S Adwi, Hrithik M, Pakam Yasaswini, Om Manish Makadia",
    author_email    = "hsadwi.ai23@bmsce.ac.in",
    python_requires = ">=3.10",
    packages        = find_packages(exclude=["tests*", "data*", "results*", "logs*"]),
    install_requires = [
        "cupy-cuda12x>=13.0.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
    ],
    extras_require = {
        "plot": ["matplotlib>=3.8.0", "seaborn>=0.13.0"],
        "dev":  ["pytest>=8.0.0"],
    },
    entry_points = {
        "console_scripts": [
            "dcudafa-run       = run_experiment:main",
            "dcudafa-benchmark = benchmark:main",
            "dcudafa-a5000     = dcudafa_a5000:main",
        ]
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
)
