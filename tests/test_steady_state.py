"""Tests for blackhole.steady_state."""

import numpy as np

from blackhole.constants import G, M_sun, c
from blackhole.steady_state import (
    H_inner,
    H_middle,
    H_outer,
    Sigma_inner,
    Sigma_middle,
    Sigma_outer,
    T_c_inner,
    T_c_middle,
    T_c_outer,
    border_inner_middle,
    border_middle_outer,
    f_boundary,
    m_dot_rel,
    m_rel,
    r_g,
    r_hat,
    rho_inner,
    rho_middle,
    rho_outer,
    tau_inner,
    tau_middle,
    tau_outer,
    u_r_inner,
    u_r_middle,
    u_r_outer,
)


class TestDimensionlessHelpers:
    def test_r_g_10_solar(self, ten_solar_mass):
        """Schwarzschild radius for 10 M_sun ~ 2.95e6 cm."""
        val = r_g(ten_solar_mass)
        expected = 2.0 * G * ten_solar_mass / c**2
        assert abs(val - expected) / expected < 1e-10
        assert abs(val - 2.95e6) / 2.95e6 < 0.02

    def test_r_hat_identity(self, ten_solar_mass):
        r = 1e10
        assert abs(r_hat(r, ten_solar_mass) - r / r_g(ten_solar_mass)) < 1e-5

    def test_m_rel(self):
        assert abs(m_rel(M_sun) - 1.0) < 1e-10

    def test_m_dot_rel(self):
        assert abs(m_dot_rel(1.5e17) - 1.0) < 1e-10

    def test_f_boundary_at_large_r(self, ten_solar_mass):
        """At large r, f -> 1."""
        val = f_boundary(1e15, ten_solar_mass)
        assert abs(val - 1.0) < 0.01


class TestRegionBoundaries:
    def test_borders_positive(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        assert border_inner_middle(alpha, M, M_dot) > 0
        assert border_middle_outer(M_dot) > 0

    def test_inner_before_middle(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        assert border_inner_middle(alpha, M, M_dot) < border_middle_outer(M_dot)


class TestInnerRegion:
    def test_all_positive(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        r = 1e7
        assert H_inner(r, M, M_dot, alpha) > 0
        assert Sigma_inner(r, M, M_dot, alpha) > 0
        assert rho_inner(r, M, M_dot, alpha) > 0
        assert T_c_inner(r, M, M_dot, alpha) > 0
        assert tau_inner(r, M, M_dot, alpha) > 0
        assert u_r_inner(r, M, M_dot, alpha) > 0


class TestMiddleRegion:
    def test_all_positive(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        r = 1e9
        f = f_boundary(r, M)
        if f > 0:
            assert H_middle(r, M, M_dot, alpha) > 0
            assert Sigma_middle(r, M, M_dot, alpha) > 0
            assert rho_middle(r, M, M_dot, alpha) > 0
            assert T_c_middle(r, M, M_dot, alpha) > 0
            assert tau_middle(r, M, M_dot, alpha) > 0
            assert u_r_middle(r, M, M_dot, alpha) > 0


class TestOuterRegion:
    def test_all_positive(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        r = 1e11
        assert H_outer(r, M, M_dot, alpha) > 0
        assert Sigma_outer(r, M, M_dot, alpha) > 0
        assert rho_outer(r, M, M_dot, alpha) > 0
        assert T_c_outer(r, M, M_dot, alpha) > 0
        assert tau_outer(r, M, M_dot, alpha) > 0
        assert u_r_outer(r, M, M_dot, alpha) > 0


class TestArrayInput:
    def test_inner_accepts_array(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        r = np.array([1e7, 2e7, 3e7])
        result = H_inner(r, M, M_dot, alpha)
        assert result.shape == (3,)

    def test_outer_accepts_array(self):
        M = M_sun
        M_dot = 5e16
        alpha = 0.4
        r = np.array([1e11, 2e11, 3e11])
        result = Sigma_outer(r, M, M_dot, alpha)
        assert result.shape == (3,)
