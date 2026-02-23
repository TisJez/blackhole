"""Tests for blackhole.cr_solvers."""

import numpy as np

from blackhole.constants import M_sun
from blackhole.cr_solvers import (
    combined_function_CR,
    combined_function_old,
    f_1,
    f_1_old,
    f_2,
    h_2,
    jacobian_CR,
    jacobian_old,
    omega_kepler,
    p_gas,
    p_rad,
    p_tot,
    solve_cr_structure,
)

# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

class TestPGas:
    def test_positive(self):
        assert p_gas(1e-8, 1e5) > 0

    def test_proportional_to_rho(self):
        assert abs(p_gas(2e-8, 1e5) / p_gas(1e-8, 1e5) - 2.0) < 1e-10

    def test_proportional_to_T(self):
        assert abs(p_gas(1e-8, 2e5) / p_gas(1e-8, 1e5) - 2.0) < 1e-10


class TestPRad:
    def test_positive(self):
        assert p_rad(1e5) > 0

    def test_scales_as_T4(self):
        ratio = p_rad(2e5) / p_rad(1e5)
        np.testing.assert_allclose(ratio, 16.0, rtol=1e-10)

    def test_array_input(self):
        T = np.array([1e4, 1e6, 1e8])
        result = p_rad(T)
        assert result.shape == (3,)
        assert np.all(result > 0)


class TestPTot:
    def test_sum_of_parts(self):
        rho, T = 1e-8, 1e6
        np.testing.assert_allclose(p_tot(rho, T), p_gas(rho, T) + p_rad(T), rtol=1e-12)


class TestOmegaKepler:
    def test_positive(self):
        assert omega_kepler(1e10, M_sun) > 0

    def test_decreases_with_radius(self):
        assert omega_kepler(1e10, M_sun) > omega_kepler(1e11, M_sun)

    def test_array_input(self):
        r = np.array([1e9, 1e10, 1e11])
        result = omega_kepler(r, M_sun)
        assert result.shape == (3,)
        assert np.all(np.diff(result) < 0)


# ---------------------------------------------------------------------------
# Scale height
# ---------------------------------------------------------------------------

class TestH2:
    def test_positive(self):
        val = h_2(1e10, 1e-7, M_sun, 5e16, 1e5)
        assert val > 0

    def test_array_input(self):
        r = np.array([1e9, 1e10, 1e11])
        rho = np.array([1e-7, 1e-8, 1e-9])
        T = np.array([1e6, 1e5, 1e4])
        result = h_2(r, rho, M_sun, 5e16, T)
        assert result.shape == (3,)
        assert np.all(result > 0)


# ---------------------------------------------------------------------------
# Energy / stress balance
# ---------------------------------------------------------------------------

class TestF1:
    def test_returns_scalar(self):
        val = f_1(M_sun, 5e16, 1e10, 1e-7, 1e5)
        assert np.isfinite(val)

    def test_old_returns_scalar(self):
        val = f_1_old(M_sun, 5e16, 1e10, 1e-7, 1e5)
        assert np.isfinite(val)


class TestF2:
    def test_returns_scalar(self):
        val = f_2(M_sun, 5e16, 1e10, 1e-7, 1e5, 0.1)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Combined functions and Jacobians
# ---------------------------------------------------------------------------

class TestCombinedFunctions:
    def test_CR_returns_list_of_two(self):
        result = combined_function_CR([1e10, 1e-7], 1e5, 0.1)
        assert len(result) == 2
        assert all(np.isfinite(v) for v in result)

    def test_old_returns_list_of_two(self):
        result = combined_function_old([1e10, 1e-7], 1e5, 0.1)
        assert len(result) == 2
        assert all(np.isfinite(v) for v in result)


class TestJacobians:
    def test_CR_shape(self):
        J = jacobian_CR([1e10, 1e-7], 1e5, 0.1)
        assert J.shape == (2, 2)
        assert np.all(np.isfinite(J))

    def test_old_shape(self):
        J = jacobian_old([1e10, 1e-7], 1e5, 0.1)
        assert J.shape == (2, 2)
        assert np.all(np.isfinite(J))


# ---------------------------------------------------------------------------
# Convenience solver
# ---------------------------------------------------------------------------

class TestSolveCRStructure:
    def test_basic_solve(self):
        T_array = np.logspace(3, 6, 20)
        r_arr, rho_arr = solve_cr_structure(T_array, 1e10, 1e-7, 0.1)
        assert r_arr.shape == (20,)
        assert rho_arr.shape == (20,)
        assert np.all(np.isfinite(r_arr))
        assert np.all(np.isfinite(rho_arr))
        assert np.all(r_arr > 0)
        assert np.all(rho_arr > 0)

    def test_old_opacity(self):
        T_array = np.logspace(3, 6, 10)
        r_arr, rho_arr = solve_cr_structure(T_array, 1e10, 1e-7, 0.1,
                                            use_old_opacity=True)
        assert np.all(np.isfinite(r_arr))
        assert np.all(np.isfinite(rho_arr))

    def test_variable_alpha(self):
        T_array = np.logspace(3, 5, 10)
        alpha_arr = np.linspace(0.04, 0.2, 10)
        r_arr, rho_arr = solve_cr_structure(T_array, 1e10, 1e-7, alpha_arr)
        assert r_arr.shape == (10,)
        assert np.all(np.isfinite(r_arr))
