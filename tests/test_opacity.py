"""Tests for blackhole.opacity."""

import numpy as np

from blackhole.opacity import (
    kappa_bath,
    kappa_bf,
    kappa_cond,
    kappa_cond_drho,
    kappa_e,
    kappa_ff,
    kappa_Hminus,
    kappa_K,
    kappa_mol,
    kappa_rad,
    kappa_simple,
    kappa_tot,
    kappa_tot_drho,
    kappa_tot_dT,
)


class TestKappaE:
    def test_high_T_low_rho_limit(self):
        """At high T, low rho: kappa_e -> 0.2*(1+X) ~ 0.392."""
        val = kappa_e(1e-12, 1e4)
        assert val > 0
        assert val < 0.2 * (1 + 0.96)  # must be below Thomson limit

    def test_positivity(self, rho_array, T_array):
        rho, T = np.meshgrid(rho_array[:5], T_array[:5])
        vals = kappa_e(rho, T)
        assert np.all(vals > 0)

    def test_array_input(self):
        rho = np.array([1e-10, 1e-8, 1e-6])
        T = np.array([1e4, 1e5, 1e6])
        result = kappa_e(rho, T)
        assert result.shape == (3,)


class TestKappaK:
    def test_positivity(self):
        assert kappa_K(1e-8, 1e4) > 0

    def test_increases_with_density(self):
        assert kappa_K(1e-6, 1e4) > kappa_K(1e-10, 1e4)


class TestKappaHminus:
    def test_positivity(self):
        assert kappa_Hminus(1e-8, 5e3) > 0

    def test_increases_with_T(self):
        assert kappa_Hminus(1e-8, 8e3) > kappa_Hminus(1e-8, 3e3)


class TestKappaMol:
    def test_positivity(self):
        assert kappa_mol() > 0


class TestKappaRad:
    def test_positivity(self):
        assert kappa_rad(1e-8, 1e4) > 0

    def test_array_input(self):
        rho = np.array([1e-10, 1e-8])
        T = np.array([1e4, 1e5])
        result = kappa_rad(rho, T)
        assert result.shape == (2,)


class TestKappaCond:
    def test_positivity(self):
        assert kappa_cond(1e-8, 1e6) > 0


class TestKappaTot:
    def test_positivity(self):
        val = kappa_tot(1e-8, 1e4)
        assert val > 0

    def test_less_than_radiative(self):
        """Parallel combination must be <= radiative alone."""
        rho, T = 1e-8, 1e4
        assert kappa_tot(rho, T) <= kappa_rad(rho, T) * 1.001  # small tolerance

    def test_array_vectorization(self):
        rho = np.logspace(-12, -6, 20)
        T = np.full(20, 1e4)
        result = kappa_tot(rho, T)
        assert result.shape == (20,)
        assert np.all(result > 0)


class TestSimplifiedOpacities:
    def test_kappa_ff_positivity(self):
        assert kappa_ff(1e-8, 1e4) > 0

    def test_kappa_simple_positivity(self):
        assert kappa_simple(1e-8, 1e4) > 0

    def test_kappa_simple_floor(self):
        """At high T, kappa_simple -> kappa_es = 0.4."""
        val = kappa_simple(1e-10, 1e8)
        assert abs(val - 0.4) / 0.4 < 0.01

    def test_kappa_bf_positivity(self):
        assert kappa_bf(1e-8, 1e4) > 0


class TestKappaBath:
    def test_positivity(self):
        assert kappa_bath(1e8, 1e2, 1e4) > 0

    def test_consistent_with_kappa_tot(self):
        H, Sigma, T = 1e8, 1e2, 1e4
        rho = Sigma / (2.0 * H)
        assert abs(kappa_bath(H, Sigma, T) - kappa_tot(rho, T)) < 1e-30


class TestDerivatives:
    def test_kappa_tot_drho_finite(self):
        val = kappa_tot_drho(1e-8, 1e4)
        assert np.isfinite(val)

    def test_kappa_tot_dT_finite(self):
        val = kappa_tot_dT(1e-8, 1e4)
        assert np.isfinite(val)

    def test_kappa_cond_drho_finite(self):
        val = kappa_cond_drho(1e-8, 1e6)
        assert np.isfinite(val)
