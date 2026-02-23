"""Tests for blackhole.solvers."""

import numpy as np

from blackhole.constants import M_sun
from blackhole.solvers import (
    F_rad,
    F_visc,
    Y_energy,
    energy_balance,
    pressure_balance,
    solve_scale_height,
    solve_temperature,
)

M_STAR = 10 * M_sun


class TestFVisc:
    def test_positive(self):
        val = F_visc(1e8, 1e2, 1e10, 0.1, M_STAR)
        assert val > 0

    def test_proportional_to_alpha(self):
        f1 = F_visc(1e8, 1e2, 1e10, 0.1, M_STAR)
        f2 = F_visc(1e8, 1e2, 1e10, 0.2, M_STAR)
        assert abs(f2 / f1 - 2.0) < 1e-10

    def test_proportional_to_sigma(self):
        f1 = F_visc(1e8, 1e2, 1e10, 0.1, M_STAR)
        f2 = F_visc(1e8, 2e2, 1e10, 0.1, M_STAR)
        assert abs(f2 / f1 - 2.0) < 1e-10


class TestFRad:
    def test_positive(self):
        val = F_rad(1e8, 1e2, 1e5)
        assert val > 0

    def test_increases_with_temperature(self):
        f1 = F_rad(1e8, 1e2, 1e4)
        f2 = F_rad(1e8, 1e2, 1e5)
        assert f2 > f1

    def test_custom_opacity(self):
        def constant_opacity(H, Sigma, T):
            return 0.4
        val = F_rad(1e8, 1e2, 1e5, opacity_func=constant_opacity)
        assert val > 0


class TestEnergyBalance:
    def test_changes_sign_across_T_range(self):
        """Energy balance should cross zero for physical parameters."""
        H = 1e8
        Sigma = 1e2
        R = 1e10
        alpha = 0.1
        vals = [energy_balance(T, H, Sigma, R, alpha, M_STAR)
                for T in [1e2, 1e4, 1e6, 1e8]]
        # At low T, F_rad is tiny → positive residual
        # At high T, F_rad dominates → negative residual
        assert vals[0] > 0
        assert vals[-1] < 0

    def test_irradiation_increases_residual(self):
        H = 1e8
        Sigma = 1e2
        R = 1e10
        alpha = 0.1
        T = 1e5
        without = energy_balance(T, H, Sigma, R, alpha, M_STAR)
        with_irr = energy_balance(T, H, Sigma, R, alpha, M_STAR, F_irr=1e10)
        assert with_irr > without


class TestPressureBalance:
    def test_changes_sign_across_H_range(self):
        """Pressure balance should cross zero for physical parameters."""
        Sigma = 1e2
        R = 1e10
        T = 1e5
        vals = [pressure_balance(H, Sigma, R, T, M_STAR)
                for H in [1e5, 1e7, 1e9, 1e11]]
        # At small H, gas pressure dominates → positive
        # At large H, hydrostatic term dominates → negative
        assert vals[0] > 0
        assert vals[-1] < 0


class TestSolveTemperature:
    def test_returns_positive_T(self):
        N = 5
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        result = solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        assert np.all(result > 0)

    def test_does_not_mutate_input(self):
        N = 5
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        original = T_c.copy()
        solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        np.testing.assert_array_equal(T_c, original)

    def test_output_shape(self):
        N = 5
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        result = solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        assert result.shape == (N,)

    def test_skips_low_sigma(self):
        """Cells with Sigma <= 1e-100 should keep original T."""
        N = 3
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.array([1e2, 1e-200, 1e2])
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e4)
        alpha = np.full(N, 0.1)
        result = solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        assert result[1] == T_c[1]


class TestSolveScaleHeight:
    def test_returns_positive_H(self):
        N = 5
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)
        result = solve_scale_height(H, Sigma, r, T_c, M_STAR)
        assert np.all(result > 0)

    def test_does_not_mutate_input(self):
        N = 5
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)
        original = H.copy()
        solve_scale_height(H, Sigma, r, T_c, M_STAR)
        np.testing.assert_array_equal(H, original)

    def test_output_shape(self):
        N = 5
        r = np.linspace(1e10, 1e11, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        T_c = np.full(N, 1e5)
        result = solve_scale_height(H, Sigma, r, T_c, M_STAR)
        assert result.shape == (N,)


class TestYEnergy:
    def test_positive_values(self):
        R = np.linspace(1e9, 1e12, 100)
        j_val = 50
        dMj = 1e20
        dMj1 = 1e20
        M_dot = 1e16
        y1, y2 = Y_energy(R, j_val, dMj, dMj1, M_STAR, M_dot)
        assert y1 > 0
        assert y2 > 0

    def test_finite(self):
        R = np.linspace(1e9, 1e12, 100)
        j_val = 50
        y1, y2 = Y_energy(R, j_val, 1e20, 1e20, M_STAR, 1e16)
        assert np.isfinite(y1)
        assert np.isfinite(y2)
