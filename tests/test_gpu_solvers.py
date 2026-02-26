"""Tests for blackhole.gpu.solvers — validates against CPU solvers.

The batched vectorized secant solver may converge to a slightly different
root than the scalar CPU solver (different iteration order, clamping
behaviour), so we use rtol=1e-2 for the solver outputs.
"""

import numpy as np

from blackhole import solvers as cpu_solvers
from blackhole.constants import M_sun
from blackhole.gpu import solvers as gpu_solvers

M_STAR = 10 * M_sun


class TestSolveTemperature:
    def test_returns_positive_T(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        result = gpu_solvers.solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        assert np.all(result > 0)

    def test_output_shape(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        result = gpu_solvers.solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        assert result.shape == (N,)

    def test_does_not_mutate_input(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        original = T_c.copy()
        gpu_solvers.solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        np.testing.assert_array_equal(T_c, original)

    def test_skips_low_sigma(self):
        N = 3
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.array([1e2, 1e-200, 1e2])
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        result = gpu_solvers.solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        assert result[1] == T_c[1]

    def test_matches_cpu_solver(self):
        """GPU solver should agree with CPU solver to within 1%."""
        N = 20
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)

        cpu_result = cpu_solvers.solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        gpu_result = gpu_solvers.solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)

        # Both should find valid temperatures
        assert np.all(cpu_result > 0)
        assert np.all(gpu_result > 0)

        # Compare where both solvers found non-trivial results
        changed = (cpu_result != T_c) & (gpu_result != T_c)
        if np.any(changed):
            np.testing.assert_allclose(
                gpu_result[changed], cpu_result[changed], rtol=1e-2,
            )

    def test_with_irradiation(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        F_irr = np.full(N, 1e10)

        cpu_result = cpu_solvers.solve_temperature(
            H, Sigma, r, T_c, alpha, M_STAR, F_irr=F_irr,
        )
        gpu_result = gpu_solvers.solve_temperature(
            H, Sigma, r, T_c, alpha, M_STAR, F_irr=F_irr,
        )

        assert np.all(gpu_result > 0)
        changed = (cpu_result != T_c) & (gpu_result != T_c)
        if np.any(changed):
            np.testing.assert_allclose(
                gpu_result[changed], cpu_result[changed], rtol=1e-2,
            )


class TestSolveScaleHeight:
    def test_returns_positive_H(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)
        result = gpu_solvers.solve_scale_height(H, Sigma, r, T_c, M_STAR)
        assert np.all(result > 0)

    def test_output_shape(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)
        result = gpu_solvers.solve_scale_height(H, Sigma, r, T_c, M_STAR)
        assert result.shape == (N,)

    def test_does_not_mutate_input(self):
        N = 10
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)
        original = H.copy()
        gpu_solvers.solve_scale_height(H, Sigma, r, T_c, M_STAR)
        np.testing.assert_array_equal(H, original)

    def test_matches_cpu_solver(self):
        """GPU solver should agree with CPU solver to within 1%."""
        N = 20
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)

        cpu_result = cpu_solvers.solve_scale_height(H, Sigma, r, T_c, M_STAR)
        gpu_result = gpu_solvers.solve_scale_height(H, Sigma, r, T_c, M_STAR)

        assert np.all(cpu_result > 0)
        assert np.all(gpu_result > 0)

        changed = (cpu_result != H) & (gpu_result != H)
        if np.any(changed):
            np.testing.assert_allclose(
                gpu_result[changed], cpu_result[changed], rtol=1e-2,
            )


class TestBatchedSecant:
    def test_quadratic_roots(self):
        """Batched secant should find roots of x^2 - a for several a values."""
        a = np.array([4.0, 9.0, 16.0, 25.0])

        def f(x, a_vals):
            return x ** 2 - a_vals

        x0 = np.array([1.5, 2.5, 3.5, 4.5])
        result, converged = gpu_solvers._batched_secant(f, x0, (a,))
        np.testing.assert_allclose(result, np.sqrt(a), atol=1e-6)
        assert np.all(converged)

    def test_convergence_masking(self):
        """Points that converge early should not be disturbed by later iterations."""
        # Mix of easy and harder roots
        targets = np.array([1.0, 100.0, 10000.0])

        def f(x, t):
            return x ** 2 - t

        x0 = np.array([0.9, 9.0, 90.0])
        result, converged = gpu_solvers._batched_secant(f, x0, (targets,))
        np.testing.assert_allclose(result, np.sqrt(targets), atol=1e-6)
