"""
tests/test_dcudafa.py — D-CUDA-FA Test Suite (RTX A5000, sm_86)
===============================================================
Comprehensive pytest test suite covering:

  Unit tests  (no GPU required — cpu-only / mock):
    • test_config        : DCUDAFAConfig defaults and validation
    • test_datasets      : make_synthetic_credit shape/dtype/range checks
    • test_datasets_home : load_home_credit fallback to synthetic when CSV absent
    • test_ewm           : compute_ewm_weights shape, sum-to-one, non-negative
    • test_dynamic_lambda: λ(t) formula correctness at neutral/stress VIX
    • test_psi_basel     : ΨBasel = 0 when CAR ≥ 8%, positive when below
    • test_portfolio_car : CAR formula on a simple 2-cluster scenario (CPU mock)
    • test_preprocessing : z_score_standardise CPU path + covariance matrix

  GPU tests  (require CUDA GPU — decorated @pytest.mark.gpu):
    • test_kernel_compile      : CUDA kernel compiles without error on sm_86
    • test_assign_clusters     : cluster_assign_kernel produces valid int32 labels
    • test_fitness_kernel_shape: mahal_fitness_kernel returns correct (N,) shapes
    • test_solver_fit_small    : DCUDAFA.fit() on small synthetic n=1000
    • test_solver_predict      : predict() returns correct shape & valid labels
    • test_solver_cluster_quality: cluster_quality() dict has all required keys
    • test_firefly_update      : firefly_update returns correct shape, finite values
    • test_brightness_sort     : sorted fitness is monotonically non-increasing
    • test_ewm_gpu             : compute_ewm_weights_gpu shape, sum, GPU dtype
    • test_tail_risk_metrics   : tail_risk_precision_recall returns valid P/R/F1
    • test_convergence_history : fit() populates history with T entries

Run all:        pytest
Run CPU only:   pytest -m "not gpu"
Run GPU only:   pytest -m gpu -v
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

# Allow imports from project root (for direct pytest invocation)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    """Return True if a CUDA GPU is accessible via CuPy."""
    try:
        import cupy as cp
        cp.array([1.0])            # triggers device init
        return True
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()
skip_no_gpu   = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="No CUDA GPU available — skipping GPU test"
)


# ─────────────────────────────────────────────────────────────────────────────
# ── CPU / Pure-Python Unit Tests ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    """DCUDAFAConfig defaults and field types."""

    def test_defaults(self):
        from config import DCUDAFAConfig
        cfg = DCUDAFAConfig()
        assert cfg.n_fireflies  == 256
        assert cfg.n_iterations == 500
        assert cfg.k            == 5
        assert cfg.sm_arch      == "sm_86"
        assert cfg.car_min      == 0.08
        # Bug-fix assertions: verify corrected values
        assert cfg.gamma        == 1.0,   f"gamma should be 1.0 not {cfg.gamma}"
        assert cfg.alpha        == 0.01,  f"alpha should be 0.01 not {cfg.alpha}"
        assert cfg.alpha_decay  == 0.995, f"alpha_decay should be 0.995"
        assert cfg.tier1        == 0.0,   f"tier1 should be 0.0 (auto-scale)"

    def test_custom_values(self):
        from config import DCUDAFAConfig
        cfg = DCUDAFAConfig(n_fireflies=64, n_iterations=100, k=3, vix_current=35.0)
        assert cfg.n_fireflies  == 64
        assert cfg.n_iterations == 100
        assert cfg.k            == 3
        assert cfg.vix_current  == 35.0

    def test_smem_bytes(self):
        from config import smem_bytes
        # k=5: (32*32 + 5*32) * 8 = 9472
        assert smem_bytes(5) == 9_472
        # k=3: (32*32 + 3*32) * 8 = 8960
        assert smem_bytes(3) == 8_960

    def test_smem_within_limit(self):
        """Shared memory must stay within 48 KB per SM limit."""
        from config import smem_bytes
        for k in range(1, 16):
            assert smem_bytes(k) < 48 * 1024, \
                f"smem_bytes({k})={smem_bytes(k)} exceeds 48 KB SM limit"


class TestDatasets:
    """make_synthetic_credit correctness."""

    def test_shape(self):
        from datasets import make_synthetic_credit
        X, labels, pd = make_synthetic_credit(n=1000, d=30, k=5, seed=0)
        assert X.shape      == (1000, 30)
        assert labels.shape == (1000,)
        assert pd.shape     == (1000,)

    def test_dtypes(self):
        from datasets import make_synthetic_credit
        X, labels, pd = make_synthetic_credit(n=500, d=30, k=5, seed=1)
        assert X.dtype      == np.float64
        assert labels.dtype == np.int32
        assert pd.dtype     == np.float64

    def test_pd_range(self):
        from datasets import make_synthetic_credit
        _, _, pd = make_synthetic_credit(n=2000, d=30, k=5, seed=2)
        assert float(pd.min()) >= 0.001
        assert float(pd.max()) <= 0.999

    def test_labels_range(self):
        from datasets import make_synthetic_credit
        _, labels, _ = make_synthetic_credit(n=2000, d=30, k=5, seed=3)
        assert int(labels.min()) >= 0
        assert int(labels.max()) <= 4

    def test_all_clusters_represented(self):
        """All k clusters should appear in labels for large enough n."""
        from datasets import make_synthetic_credit
        _, labels, _ = make_synthetic_credit(n=5000, d=30, k=5, seed=4)
        assert len(np.unique(labels)) == 5

    def test_reproducibility(self):
        from datasets import make_synthetic_credit
        X1, l1, p1 = make_synthetic_credit(n=500, d=30, k=5, seed=99)
        X2, l2, p2 = make_synthetic_credit(n=500, d=30, k=5, seed=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(p1, p2)

    def test_no_nans(self):
        from datasets import make_synthetic_credit
        X, _, pd = make_synthetic_credit(n=1000, d=30, k=5, seed=5)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(pd))

    def test_stress_vix(self):
        """VIX=35 should produce more borrowers in the tail cluster."""
        from datasets import make_synthetic_credit
        _, labels_neutral, _ = make_synthetic_credit(n=10000, d=30, k=5, vix=20.0, seed=6)
        _, labels_stress,  _ = make_synthetic_credit(n=10000, d=30, k=5, vix=35.0, seed=6)
        # Cluster 0 is the tail cluster; stress should have more of them
        n_tail_neutral = int((labels_neutral == 0).sum())
        n_tail_stress  = int((labels_stress  == 0).sum())
        assert n_tail_stress >= n_tail_neutral


class TestDatasetsHomeCredit:
    """load_home_credit fallback behaviour when CSV is absent."""

    def test_fallback_to_synthetic(self, tmp_path):
        from datasets import load_home_credit
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X, pd = load_home_credit(data_dir=str(tmp_path), target_n=500)
        assert X.shape    == (500, 30)
        assert pd.shape   == (500,)
        assert X.dtype    == np.float64
        assert pd.dtype   == np.float64
        # Should have warned about fallback
        assert any("not found" in str(warning.message) for warning in w)


class TestDynamicLambda:
    """Eq (8): λ(t) = λ₀ (1 + δ · VIX(t) / VIXref)."""

    def test_neutral_vix(self):
        from car_constraint import dynamic_lambda
        from config import DCUDAFAConfig
        cfg = DCUDAFAConfig(lambda0=0.5, delta=0.04, vix_ref=20.0, vix_current=20.0)
        lam = dynamic_lambda(cfg)
        # λ(t) = 0.5 * (1 + 0.04 * 20/20) = 0.5 * 1.04 = 0.52
        assert abs(lam - 0.52) < 1e-9

    def test_stressed_vix(self):
        from car_constraint import dynamic_lambda
        from config import DCUDAFAConfig
        cfg = DCUDAFAConfig(lambda0=0.5, delta=0.04, vix_ref=20.0, vix_current=40.0)
        lam = dynamic_lambda(cfg)
        # λ(t) = 0.5 * (1 + 0.04 * 40/20) = 0.5 * 1.08 = 0.54
        assert abs(lam - 0.54) < 1e-9

    def test_lambda_increases_with_vix(self):
        from car_constraint import dynamic_lambda
        from config import DCUDAFAConfig
        lam_low  = dynamic_lambda(DCUDAFAConfig(vix_current=10.0))
        lam_high = dynamic_lambda(DCUDAFAConfig(vix_current=50.0))
        assert lam_high > lam_low


class TestPsiBasel:
    """Eq (7): ΨBasel = λ · max(0, CARmin − CARport)²."""

    def test_compliant_car_gives_zero(self):
        from car_constraint import psi_basel
        # CAR = 12% ≥ 8% → ΨBasel = 0
        assert psi_basel(car=0.12, lam=1.0, car_min=0.08) == 0.0

    def test_exactly_at_minimum_gives_zero(self):
        from car_constraint import psi_basel
        assert psi_basel(car=0.08, lam=1.0, car_min=0.08) == 0.0

    def test_below_minimum_positive(self):
        from car_constraint import psi_basel
        # CAR = 5% < 8% → shortfall = 0.03 → ΨBasel = 1.0 * 0.03² = 0.0009
        psi = psi_basel(car=0.05, lam=1.0, car_min=0.08)
        assert abs(psi - 0.0009) < 1e-10

    def test_higher_lambda_scales_penalty(self):
        from car_constraint import psi_basel
        psi1 = psi_basel(car=0.05, lam=1.0, car_min=0.08)
        psi2 = psi_basel(car=0.05, lam=2.0, car_min=0.08)
        assert abs(psi2 - 2 * psi1) < 1e-10


class TestPreprocessing:
    """Z-score standardisation and covariance estimation — CPU path."""

    def test_z_score_mean_zero(self):
        """After z-scoring, each feature should have mean ≈ 0."""
        from preprocessing import z_score_standardise
        if not GPU_AVAILABLE:
            pytest.skip("CuPy not available")
        import cupy as cp
        X   = np.random.default_rng(0).normal(5, 3, (500, 10))
        Xp, _ = z_score_standardise(X)
        means = cp.asnumpy(Xp.mean(axis=0))
        np.testing.assert_allclose(means, np.zeros(10), atol=1e-9)

    def test_z_score_std_one(self):
        """After z-scoring, each feature should have std ≈ 1."""
        from preprocessing import z_score_standardise
        if not GPU_AVAILABLE:
            pytest.skip("CuPy not available")
        import cupy as cp
        X   = np.random.default_rng(1).normal(5, 3, (500, 10))
        Xp, _ = z_score_standardise(X)
        stds = cp.asnumpy(Xp.std(axis=0))
        np.testing.assert_allclose(stds, np.ones(10), atol=1e-9)

    def test_precision_matrix_shape(self):
        from preprocessing import compute_precision_gpu
        if not GPU_AVAILABLE:
            pytest.skip("CuPy not available")
        import cupy as cp
        rng = np.random.default_rng(2)
        X   = cp.asarray(rng.normal(0, 1, (200, 15)))
        Si  = compute_precision_gpu(X)
        assert Si.shape == (15, 15)

    def test_precision_matrix_invertibility(self):
        """Σ⁻¹ · Σ should be close to identity."""
        from preprocessing import compute_precision_gpu, compute_covariance_gpu
        if not GPU_AVAILABLE:
            pytest.skip("CuPy not available")
        import cupy as cp
        rng = np.random.default_rng(3)
        X   = cp.asarray(rng.normal(0, 1, (300, 10)))
        Sigma    = compute_covariance_gpu(X)
        Sigma_inv = compute_precision_gpu(X)
        product  = cp.asnumpy(Sigma_inv @ Sigma)
        np.testing.assert_allclose(product, np.eye(10), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# ── GPU Tests (require CUDA) ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.gpu
@skip_no_gpu
class TestKernelCompile:
    """CUDA kernel compiles on sm_86 without error."""

    def test_compile(self):
        from kernel import get_kernels
        mahal_fn, assign_fn = get_kernels()
        assert mahal_fn  is not None
        assert assign_fn is not None


@pytest.mark.gpu
@skip_no_gpu
class TestEWMGPU:
    """compute_ewm_weights GPU path."""

    def test_shape(self):
        import cupy as cp
        from car_constraint import compute_ewm_weights
        X = cp.asarray(np.random.default_rng(0).normal(0, 1, (500, 30)))
        w = compute_ewm_weights(X)
        assert w.shape[0] == 30

    def test_sums_to_one(self):
        import cupy as cp
        from car_constraint import compute_ewm_weights
        X = cp.asarray(np.random.default_rng(1).normal(0, 1, (500, 30)))
        w = compute_ewm_weights(X)
        assert abs(float(w.sum()) - 1.0) < 1e-9

    def test_non_negative(self):
        import cupy as cp
        from car_constraint import compute_ewm_weights
        X = cp.asarray(np.random.default_rng(2).normal(0, 1, (500, 30)))
        w = compute_ewm_weights(X)
        assert float(w.min()) >= 0.0

    def test_pad_to_4(self):
        """pad_ewm_weights should extend short weight vectors to length ≥ 4."""
        import cupy as cp
        from car_constraint import pad_ewm_weights
        w_short = cp.array([0.6, 0.4])
        w_padded = pad_ewm_weights(w_short, min_len=4)
        assert w_padded.shape[0] >= 4


@pytest.mark.gpu
@skip_no_gpu
class TestAssignClusters:
    """cluster_assign_kernel produces valid integer labels."""

    def test_shape(self):
        import cupy as cp
        from solver import assign_clusters, compute_precision_matrix
        rng = np.random.default_rng(10)
        X   = cp.asarray(rng.normal(0, 1, (300, 10)))
        Si  = compute_precision_matrix(X)
        # 3 random centroids from data
        cents = X[:3].copy()
        labels = assign_clusters(X, cents, Si)
        assert labels.shape == (300,)

    def test_dtype(self):
        import cupy as cp
        from solver import assign_clusters, compute_precision_matrix
        rng = np.random.default_rng(11)
        X   = cp.asarray(rng.normal(0, 1, (300, 10)))
        Si  = compute_precision_matrix(X)
        cents = X[:5].copy()
        labels = assign_clusters(X, cents, Si)
        assert labels.dtype == cp.int32

    def test_labels_in_range(self):
        import cupy as cp
        from solver import assign_clusters, compute_precision_matrix
        rng = np.random.default_rng(12)
        X   = cp.asarray(rng.normal(0, 1, (300, 10)))
        Si  = compute_precision_matrix(X)
        k   = 4
        cents = X[:k].copy()
        labels = assign_clusters(X, cents, Si)
        assert int(labels.min()) >= 0
        assert int(labels.max()) <= k - 1


@pytest.mark.gpu
@skip_no_gpu
class TestFitnessKernelShape:
    """mahal_fitness_kernel output shapes are (N,)."""

    def test_output_shapes(self):
        import cupy as cp
        from solver import (
            z_score_standardise, compute_precision_matrix,
            _dispatch_fitness_kernel,
        )
        from car_constraint import compute_ewm_weights, pad_ewm_weights
        from config import DCUDAFAConfig

        rng = np.random.default_rng(20)
        X   = rng.normal(0, 1, (500, 10))
        Xp, _  = z_score_standardise(X)
        Si     = compute_precision_matrix(Xp)

        N, k, d = 8, 3, Xp.shape[1]
        pop = cp.asarray(rng.normal(0, 1, (N, k, d)))
        pd  = cp.zeros(500, dtype=cp.float64)
        cfg = DCUDAFAConfig(n_fireflies=N, k=k)
        stream = cp.cuda.Stream()

        jm, phi_pd, phi_dti = _dispatch_fitness_kernel(Xp, pop, Si, pd, cfg, stream)
        stream.synchronize()
        assert jm.shape      == (N,)
        assert phi_pd.shape  == (N,)
        assert phi_dti.shape == (N,)

    def test_output_finite(self):
        import cupy as cp
        from solver import (
            z_score_standardise, compute_precision_matrix,
            _dispatch_fitness_kernel,
        )
        from config import DCUDAFAConfig

        rng = np.random.default_rng(21)
        X   = rng.normal(0, 1, (200, 10))
        Xp, _ = z_score_standardise(X)
        Si    = compute_precision_matrix(Xp)

        N, k, d = 4, 3, Xp.shape[1]
        pop    = cp.asarray(rng.normal(0, 1, (N, k, d)))
        pd     = cp.zeros(200, dtype=cp.float64)
        cfg    = DCUDAFAConfig(n_fireflies=N, k=k)
        stream = cp.cuda.Stream()

        jm, _, _ = _dispatch_fitness_kernel(Xp, pop, Si, pd, cfg, stream)
        stream.synchronize()
        jm_cpu = cp.asnumpy(jm)
        assert np.all(np.isfinite(jm_cpu)), "J_M kernel returned non-finite values"


@pytest.mark.gpu
@skip_no_gpu
class TestFireflyUpdate:
    """firefly_update returns correct shape and finite values."""

    def test_shape_preserved(self):
        import cupy as cp
        from solver import firefly_update
        from config import DCUDAFAConfig

        rng    = cp.random.default_rng(30)
        N, k, d = 16, 5, 30
        pop     = rng.standard_normal((N, k, d), dtype=cp.float64)
        fitness = rng.standard_normal((N,), dtype=cp.float64)
        cfg     = DCUDAFAConfig(n_fireflies=N, k=k)

        new_pop = firefly_update(pop, fitness, cfg, rng)
        assert new_pop.shape == (N, k, d)

    def test_finite_values(self):
        import cupy as cp
        from solver import firefly_update
        from config import DCUDAFAConfig

        rng     = cp.random.default_rng(31)
        N, k, d = 16, 5, 30
        pop     = rng.standard_normal((N, k, d), dtype=cp.float64)
        fitness = rng.standard_normal((N,), dtype=cp.float64)
        cfg     = DCUDAFAConfig(n_fireflies=N, k=k)

        new_pop = firefly_update(pop, fitness, cfg, rng)
        assert cp.all(cp.isfinite(new_pop)), "Firefly update produced non-finite positions"

    def test_positions_changed(self):
        """Population should change after update (not trivially equal)."""
        import cupy as cp
        from solver import firefly_update
        from config import DCUDAFAConfig

        rng     = cp.random.default_rng(32)
        N, k, d = 16, 5, 10
        pop     = rng.standard_normal((N, k, d), dtype=cp.float64)
        # Assign varied fitness so some attractors exist
        fitness = cp.arange(N, dtype=cp.float64)
        cfg     = DCUDAFAConfig(n_fireflies=N, k=k)

        new_pop = firefly_update(pop, fitness, cfg, rng)
        assert not cp.allclose(pop, new_pop), "Population unchanged after update"


@pytest.mark.gpu
@skip_no_gpu
class TestSolverFitSmall:
    """DCUDAFA.fit() end-to-end on a tiny synthetic dataset."""

    @pytest.fixture(scope="class")
    def fitted_model(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X, _, pd_col = make_synthetic_credit(n=1000, d=30, k=5, seed=0)
        cfg   = DCUDAFAConfig(n_fireflies=8, n_iterations=10, k=5, seed=0)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd_col, verbose=False)
        return model

    def test_fit_completes(self, fitted_model):
        assert fitted_model.best_centroids_ is not None

    def test_centroids_shape(self, fitted_model):
        import cupy as cp
        c = fitted_model.best_centroids_
        assert c.shape == (5, 30)

    def test_labels_shape(self, fitted_model):
        assert fitted_model.best_labels_.shape[0] == 1000

    def test_labels_range(self, fitted_model):
        import cupy as cp
        lb = fitted_model.best_labels_
        assert int(lb.min()) >= 0
        assert int(lb.max()) <= 4

    def test_fit_time_positive(self, fitted_model):
        assert fitted_model.fit_time_s_ > 0.0

    def test_history_length(self, fitted_model):
        assert len(fitted_model.history_["fitness"])  == 10
        assert len(fitted_model.history_["car"])      == 10
        assert len(fitted_model.history_["jm"])       == 10


@pytest.mark.gpu
@skip_no_gpu
class TestSolverPredict:
    """predict() returns correct shape and valid labels."""

    def test_predict_shape(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X_train, _, pd = make_synthetic_credit(n=500, d=30, k=5, seed=10)
        X_test,  _, _  = make_synthetic_credit(n=200, d=30, k=5, seed=11)

        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=5, seed=10)
        model = DCUDAFA(cfg)
        model.fit(X_train, pd_col=pd, verbose=False)

        preds = model.predict(X_test)
        assert preds.shape == (200,)
        assert preds.dtype == np.int32

    def test_predict_valid_labels(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X_train, _, pd = make_synthetic_credit(n=500, d=30, k=3, seed=20)
        X_test,  _, _  = make_synthetic_credit(n=100, d=30, k=3, seed=21)

        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=3, seed=20)
        model = DCUDAFA(cfg)
        model.fit(X_train, pd_col=pd, verbose=False)

        preds = model.predict(X_test)
        assert int(preds.min()) >= 0
        assert int(preds.max()) <= 2


@pytest.mark.gpu
@skip_no_gpu
class TestSolverClusterQuality:
    """cluster_quality() returns a dict with all required Table IV keys."""

    _REQUIRED_KEYS = {
        "JM_mahalanobis",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_score",
        "risk_concentration_ratio",
        "portfolio_car",
        "basel_iii_compliant",
        "fit_time_s",
    }

    def test_all_keys_present(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X, _, pd = make_synthetic_credit(n=800, d=30, k=5, seed=30)
        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=5, seed=30)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        q = model.cluster_quality()
        for key in self._REQUIRED_KEYS:
            assert key in q, f"Missing key: {key}"

    def test_silhouette_in_range(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X, _, pd = make_synthetic_credit(n=800, d=30, k=5, seed=31)
        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=5, seed=31)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        q = model.cluster_quality()
        sil = q["silhouette_score"]
        assert -1.0 <= sil <= 1.0, f"Silhouette {sil} out of [-1, 1]"

    def test_rcr_positive(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X, _, pd = make_synthetic_credit(n=800, d=30, k=5, seed=32)
        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=5, seed=32)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        q = model.cluster_quality()
        assert q["risk_concentration_ratio"] > 0.0


@pytest.mark.gpu
@skip_no_gpu
class TestTailRiskMetrics:
    """tail_risk_metrics / tail_risk_precision_recall returns valid P/R/F1."""

    def test_metrics_range(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X, _, pd = make_synthetic_credit(n=1000, d=30, k=5, seed=40)
        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=5, seed=40)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        true_target = (pd > 0.5).astype(np.int32)
        result = model.tail_risk_metrics(true_target, verbose=False)

        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"]    <= 1.0
        assert 0.0 <= result["f1_score"]  <= 1.0

    def test_validation_module_matches_solver(self):
        """validation.tail_risk_precision_recall and solver.tail_risk_metrics
        should agree on cluster assignment for the same model."""
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA
        from validation import tail_risk_precision_recall

        X, _, pd = make_synthetic_credit(n=1000, d=30, k=5, seed=41)
        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=5, k=5, seed=41)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        true_target = (pd > 0.5).astype(np.int32)

        # via solver method
        r1 = model.tail_risk_metrics(true_target, verbose=False)
        # via standalone validation function
        r2 = tail_risk_precision_recall(
            model.best_labels_, model._pd_col, true_target,
            cfg.k, cfg.tier1, cfg.lgd, verbose=False)

        assert r1["tail_cluster_id"] == r2["tail_cluster_id"]
        assert abs(r1["precision"] - r2["precision"]) < 1e-6


@pytest.mark.gpu
@skip_no_gpu
class TestPortfolioCAR:
    """portfolio_car formula on simple controlled inputs."""

    def test_two_cluster_manual(self):
        """Manually compute CAR for a 2-cluster, equal-size portfolio."""
        import cupy as cp
        from car_constraint import portfolio_car

        n     = 1000
        # Cluster 0: 500 borrowers, PD=0.1
        # Cluster 1: 500 borrowers, PD=0.3
        labels = cp.concatenate([cp.zeros(500, dtype=cp.int32),
                                  cp.ones( 500, dtype=cp.int32)])
        pd_col = cp.concatenate([cp.full(500, 0.1), cp.full(500, 0.3)])

        tier1 = 1_000_000.0
        lgd   = 0.45
        k     = 2

        car, rw = portfolio_car(labels, pd_col, k, tier1, lgd)

        # Manual calculation:
        # RW_0 = 12.5 * 0.1 * 0.45 = 0.5625
        # RW_1 = 12.5 * 0.3 * 0.45 = 1.6875
        # denom = 500 * 0.5625 * 500 + 500 * 1.6875 * 500
        #       = 140625 + 421875 = 562500
        # CAR  = 1_000_000 / 562_500 ≈ 1.7778
        expected_car = 1_000_000.0 / 562_500.0
        assert abs(car - expected_car) < 1e-4

    def test_car_positive(self):
        import cupy as cp
        from car_constraint import portfolio_car

        rng    = np.random.default_rng(50)
        n      = 500
        labels = cp.asarray(rng.integers(0, 5, n).astype(np.int32))
        pd_col = cp.asarray(rng.uniform(0.01, 0.5, n))

        car, _ = portfolio_car(labels, pd_col, k=5, tier1=1e6, lgd=0.45)
        assert car > 0.0


@pytest.mark.gpu
@skip_no_gpu
class TestConvergenceHistory:
    """fit() should populate history with exactly T entries."""

    def test_history_length_matches_iterations(self):
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        T   = 7
        X, _, pd = make_synthetic_credit(n=500, d=30, k=5, seed=60)
        cfg   = DCUDAFAConfig(n_fireflies=4, n_iterations=T, k=5, seed=60)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        assert len(model.history_["fitness"]) == T
        assert len(model.history_["car"])     == T
        assert len(model.history_["jm"])      == T

    def test_best_fitness_non_decreasing(self):
        """Best fitness should be monotonically non-decreasing over iterations."""
        from datasets import make_synthetic_credit
        from config import DCUDAFAConfig
        from solver import DCUDAFA

        X, _, pd = make_synthetic_credit(n=500, d=30, k=5, seed=61)
        cfg   = DCUDAFAConfig(n_fireflies=8, n_iterations=20, k=5, seed=61)
        model = DCUDAFA(cfg)
        model.fit(X, pd_col=pd, verbose=False)

        hist = model.history_["fitness"]
        for i in range(1, len(hist)):
            assert hist[i] >= hist[i - 1] - 1e-9, \
                f"Fitness decreased at iter {i}: {hist[i-1]:.6f} → {hist[i]:.6f}"
