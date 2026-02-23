"""Tests for blackhole.irradiation."""

import numpy as np

from blackhole.constants import ALPHA_COLD, ALPHA_HOT, M_sun
from blackhole.irradiation import (
    Epsilon_irr,
    Flux_irr,
    Sigma_max,
    Sigma_min,
    T_c_max,
    T_c_min,
    alpha_visc_irr,
    r10,
)

M_STAR = 9 * M_sun
M_DOT = 1e17


class TestR10:
    def test_value(self):
        assert r10(1e10) == 1.0

    def test_array(self):
        r = np.array([1e10, 2e10])
        np.testing.assert_array_equal(r10(r), np.array([1.0, 2.0]))


class TestFluxIrr:
    def test_positivity(self):
        val = Flux_irr(
            Sigma_inner=1e2, nu_inner=1e14, r_inner=1e10,
            r=1e12, M_star=M_STAR, M_dot=M_DOT,
        )
        assert val > 0

    def test_decreases_with_r(self):
        kwargs = dict(Sigma_inner=1e2, nu_inner=1e14, r_inner=1e10, M_star=M_STAR, M_dot=M_DOT)
        f1 = Flux_irr(r=1e11, **kwargs)
        f2 = Flux_irr(r=1e12, **kwargs)
        assert f1 > f2

    def test_array_r(self):
        r = np.array([1e11, 1e12, 1e13])
        result = Flux_irr(
            Sigma_inner=1e2, nu_inner=1e14, r_inner=1e10,
            r=r, M_star=M_STAR, M_dot=M_DOT,
        )
        assert result.shape == (3,)


class TestEpsilonIrr:
    def test_non_negative(self):
        val = Epsilon_irr(
            Sigma_inner=1e2, nu_inner=1e14, r_inner=1e10,
            r=1e12, M_star=M_STAR, M_dot=M_DOT,
        )
        assert val >= 0


class TestCriticalValues:
    def test_Sigma_max_positive(self):
        assert Sigma_max(eps_irr=0.0, r=1e11, M_star=M_STAR) > 0

    def test_Sigma_min_positive(self):
        assert Sigma_min(eps_irr=0.0, r=1e11, M_star=M_STAR) > 0

    def test_Sigma_max_gt_Sigma_min(self):
        r = 1e11
        s_max = Sigma_max(eps_irr=0.0, r=r, M_star=M_STAR)
        s_min = Sigma_min(eps_irr=0.0, r=r, M_star=M_STAR)
        assert s_max > s_min

    def test_T_c_max_positive(self):
        assert T_c_max(eps_irr=0.0, r=1e11) > 0

    def test_T_c_min_positive(self):
        val = T_c_min(eps_irr=0.0, r=1e11, M_star=M_STAR)
        assert val > 0

    def test_array_input(self):
        r = np.array([1e10, 1e11, 1e12])
        eps = np.zeros(3)
        result = Sigma_max(eps_irr=eps, r=r, M_star=M_STAR)
        assert result.shape == (3,)


class TestAlphaViscIrr:
    def test_cold_limit(self):
        """At low T, should approach alpha_cold."""
        val = alpha_visc_irr(T_c=1e3, eps_irr=0.0, r=1e11, M_star=M_STAR)
        assert abs(val - ALPHA_COLD) / ALPHA_COLD < 0.05

    def test_hot_limit(self):
        """At high T, should approach alpha_hot."""
        val = alpha_visc_irr(T_c=1e7, eps_irr=0.0, r=1e11, M_star=M_STAR)
        assert abs(val - ALPHA_HOT) / ALPHA_HOT < 0.05

    def test_bounded(self):
        T = np.logspace(2, 8, 50)
        vals = alpha_visc_irr(T_c=T, eps_irr=0.0, r=1e11, M_star=M_STAR)
        assert np.all(vals >= ALPHA_COLD * 0.95)
        assert np.all(vals <= ALPHA_HOT * 1.05)
