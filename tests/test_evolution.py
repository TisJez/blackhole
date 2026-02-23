"""Tests for blackhole.evolution."""

import numpy as np

from blackhole.constants import M_sun
from blackhole.disk_physics import X_func
from blackhole.evolution import (
    MassTransferResult,
    add_mass,
    calculate_timestep,
    disk_evap,
    evolve_surface_density,
)

M_STAR = 10 * M_sun


class TestCalculateTimestep:
    def test_positive(self):
        X = np.linspace(1e3, 1e5, 100)
        nu = np.full(100, 1e14)
        dX = X[1] - X[0]
        dt = calculate_timestep(X, nu, dX)
        assert dt > 0

    def test_scales_inversely_with_max_nu(self):
        X = np.linspace(1e3, 1e5, 100)
        dX = X[1] - X[0]
        nu1 = np.full(100, 1e14)
        nu2 = np.full(100, 2e14)
        dt1 = calculate_timestep(X, nu1, dX)
        dt2 = calculate_timestep(X, nu2, dX)
        assert abs(dt1 / dt2 - 2.0) < 1e-10

    def test_scales_with_dX_squared(self):
        X = np.linspace(1e3, 1e5, 100)
        nu = np.full(100, 1e14)
        dX1 = 1.0
        dX2 = 2.0
        dt1 = calculate_timestep(X, nu, dX1)
        dt2 = calculate_timestep(X, nu, dX2)
        assert abs(dt2 / dt1 - 4.0) < 1e-10


class TestDiskEvap:
    def test_positive(self):
        r = np.array([1e10, 1e11, 1e12])
        result = disk_evap(r, M_STAR)
        assert np.all(result > 0)

    def test_scalar_input(self):
        result = disk_evap(1e10, M_STAR)
        assert result > 0

    def test_array_shape(self):
        r = np.logspace(9, 12, 20)
        result = disk_evap(r, M_STAR)
        assert result.shape == (20,)


class TestAddMass:
    def _make_grid(self, N=100):
        r_in = 1e9
        r_out = 1e12
        r = np.linspace(r_in, r_out, N)
        X = X_func(r)
        dX = X[1] - X[0]
        X_K = X[int(N * 0.7)]
        X_N = X[-1]
        return X, dX, X_K, X_N

    def test_returns_mass_transfer_result(self):
        N = 100
        X, dX, X_K, X_N = self._make_grid(N)
        Sigma = np.full(N, 10.0)
        result = add_mass(Sigma, 1e16, 1.0, X, N, X_K, X_N, dX, 1e-10)
        assert isinstance(result, MassTransferResult)

    def test_result_has_valid_fields(self):
        N = 100
        X, dX, X_K, X_N = self._make_grid(N)
        Sigma = np.full(N, 10.0)
        result = add_mass(Sigma, 1e16, 1.0, X, N, X_K, X_N, dX, 1e-10)
        assert result.Sigma.shape == (N,)
        assert 0 <= result.j_val < N
        assert np.isfinite(result.dMj)
        assert np.isfinite(result.dMj1)

    def test_sigma_above_floor(self):
        N = 100
        X, dX, X_K, X_N = self._make_grid(N)
        min_Sigma = 1e-10
        Sigma = np.full(N, 10.0)
        result = add_mass(Sigma, 1e16, 1.0, X, N, X_K, X_N, dX, min_Sigma)
        assert np.all(result.Sigma >= min_Sigma)

    def test_does_not_mutate_input(self):
        N = 100
        X, dX, X_K, X_N = self._make_grid(N)
        Sigma = np.full(N, 10.0)
        original = Sigma.copy()
        add_mass(Sigma, 1e16, 1.0, X, N, X_K, X_N, dX, 1e-10)
        np.testing.assert_array_equal(Sigma, original)


class TestEvolveSurfaceDensity:
    def _make_state(self, N=100):
        r_in = 1e9
        r_out = 1e12
        r = np.linspace(r_in, r_out, N)
        X = X_func(r)
        dX = X[1] - X[0]
        Sigma = np.full(N, 10.0)
        nu = np.full(N, 1e14)
        return Sigma, nu, X, dX

    def test_preserves_positivity(self):
        N = 100
        min_Sigma = 1e-10
        Sigma, nu, X, dX = self._make_state(N)
        result = evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma,
        )
        assert np.all(result >= min_Sigma)

    def test_output_shape(self):
        N = 100
        Sigma, nu, X, dX = self._make_state(N)
        result = evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, 1e-10,
        )
        assert result.shape == (N,)

    def test_does_not_mutate_input(self):
        N = 100
        Sigma, nu, X, dX = self._make_state(N)
        original = Sigma.copy()
        evolve_surface_density(Sigma, 1.0, nu, X, dX, N, 1e-10)
        np.testing.assert_array_equal(Sigma, original)

    def test_with_tidal_params(self):
        N = 100
        min_Sigma = 1e-10
        Sigma, nu, X, dX = self._make_state(N)
        tidal = {"cw": 0.2, "a_1": 1.5e15, "n_1": 5, "trunc_frac": 0.93}
        result = evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma,
            tidal_params=tidal,
        )
        assert np.all(result >= min_Sigma)

    def test_evaporation_capped_at_available(self):
        """Evaporation cannot remove more mass than available above min_Sigma."""
        N = 100
        min_Sigma = 1e-5
        Sigma, nu, X, dX = self._make_state(N)
        # Set Sigma very low so evaporation would overshoot
        Sigma[:] = min_Sigma * 2.0

        def heavy_evap(r_arr):
            # Return a very large evaporation rate
            return np.full_like(r_arr, 1e30)

        result = evolve_surface_density(
            Sigma, 1e10, nu, X, dX, N, min_Sigma,
            evap_func=heavy_evap,
        )
        assert np.all(result >= min_Sigma)


class TestDiskEvapLRatio:
    def test_l_ratio_scales_linearly(self):
        """disk_evap with L_ratio=0.5 returns half the default rate."""
        r = np.array([1e10, 1e11, 1e12])
        full = disk_evap(r, M_STAR)
        half = disk_evap(r, M_STAR, L_ratio=0.5)
        np.testing.assert_allclose(half, 0.5 * full)

    def test_l_ratio_zero_gives_zero(self):
        """disk_evap with L_ratio=0 returns zero evaporation."""
        r = np.array([1e10, 1e11])
        result = disk_evap(r, M_STAR, L_ratio=0.0)
        np.testing.assert_array_equal(result, 0.0)

    def test_l_ratio_default_unchanged(self):
        """Default L_ratio=1.0 gives the same result as the old signature."""
        r = np.logspace(9, 12, 20)
        default = disk_evap(r, M_STAR)
        explicit = disk_evap(r, M_STAR, L_ratio=1.0)
        np.testing.assert_array_equal(default, explicit)


class TestAddMassSigmaCap:
    def _make_grid(self, N=100):
        r_in = 1e9
        r_out = 1e12
        r = np.linspace(r_in, r_out, N)
        X = X_func(r)
        dX = X[1] - X[0]
        X_K = X[int(N * 0.7)]
        X_N = X[-1]
        return X, dX, X_K, X_N

    def test_default_sigma_cap_backward_compatible(self):
        """Default sigma_cap=200 gives the same result as before."""
        N = 100
        X, dX, X_K, X_N = self._make_grid(N)
        Sigma = np.full(N, 10.0)
        r1 = add_mass(Sigma, 1e16, 1.0, X, N, X_K, X_N, dX, 1e-10)
        r2 = add_mass(Sigma, 1e16, 1.0, X, N, X_K, X_N, dX, 1e-10, sigma_cap=200.0)
        np.testing.assert_array_equal(r1.Sigma, r2.Sigma)

    def test_large_sigma_cap_allows_higher_deposition(self):
        """A larger sigma_cap allows mass to be deposited where default would block."""
        N = 100
        X, dX, X_K, X_N = self._make_grid(N)
        # Start with high Sigma that might be near the cap
        Sigma = np.full(N, 150.0)
        # With default cap=200, deposition near 150+deposit may be blocked
        r_default = add_mass(Sigma, 1e18, 100.0, X, N, X_K, X_N, dX, 1e-10,
                             sigma_cap=200.0)
        r_large = add_mass(Sigma, 1e18, 100.0, X, N, X_K, X_N, dX, 1e-10,
                           sigma_cap=1e6)
        # With a much larger cap, more mass should be deposited
        # (or at least the same amount)
        assert np.sum(r_large.Sigma) >= np.sum(r_default.Sigma)
