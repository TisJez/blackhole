"""Tests for blackhole.gpu.evolution — validates against CPU evolution."""

import numpy as np

from blackhole import evolution as cpu_evol
from blackhole.constants import M_sun
from blackhole.gpu import evolution as gpu_evol
from blackhole.gpu.disk_physics import X_func

M_STAR = 10 * M_sun


class TestCalculateTimestep:
    def test_positive(self):
        X = np.linspace(1e3, 1e5, 100)
        nu = np.full(100, 1e14)
        dX = X[1] - X[0]
        dt = gpu_evol.calculate_timestep(X, nu, dX)
        assert dt > 0

    def test_matches_cpu(self):
        X = np.linspace(1e3, 1e5, 100)
        nu = np.full(100, 1e14)
        dX = X[1] - X[0]
        cpu_dt = cpu_evol.calculate_timestep(X, nu, dX)
        gpu_dt = gpu_evol.calculate_timestep(X, nu, dX)
        np.testing.assert_allclose(gpu_dt, cpu_dt, rtol=1e-10)

    def test_scales_inversely_with_max_nu(self):
        X = np.linspace(1e3, 1e5, 100)
        dX = X[1] - X[0]
        nu1 = np.full(100, 1e14)
        nu2 = np.full(100, 2e14)
        dt1 = gpu_evol.calculate_timestep(X, nu1, dX)
        dt2 = gpu_evol.calculate_timestep(X, nu2, dX)
        np.testing.assert_allclose(dt1 / dt2, 2.0, rtol=1e-10)


class TestDiskEvap:
    def test_positive(self):
        r = np.array([1e10, 1e11, 1e12])
        result = gpu_evol.disk_evap(r, M_STAR)
        assert np.all(result > 0)

    def test_matches_cpu(self):
        r = np.logspace(9, 12, 50)
        cpu = cpu_evol.disk_evap(r, M_STAR)
        gpu = gpu_evol.disk_evap(r, M_STAR)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-10)

    def test_l_ratio_scales(self):
        r = np.array([1e10, 1e11, 1e12])
        full = gpu_evol.disk_evap(r, M_STAR)
        half = gpu_evol.disk_evap(r, M_STAR, L_ratio=0.5)
        np.testing.assert_allclose(half, 0.5 * full)


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
        result = gpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma,
        )
        assert np.all(result >= min_Sigma)

    def test_output_shape(self):
        N = 100
        Sigma, nu, X, dX = self._make_state(N)
        result = gpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, 1e-10,
        )
        assert result.shape == (N,)

    def test_does_not_mutate_input(self):
        N = 100
        Sigma, nu, X, dX = self._make_state(N)
        original = Sigma.copy()
        gpu_evol.evolve_surface_density(Sigma, 1.0, nu, X, dX, N, 1e-10)
        np.testing.assert_array_equal(Sigma, original)

    def test_explicit_matches_cpu(self):
        """Explicit Euler should match CPU exactly."""
        N = 100
        min_Sigma = 1e-10
        Sigma, nu, X, dX = self._make_state(N)
        cpu = cpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma, theta=0.0,
        )
        gpu = gpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma, theta=0.0,
        )
        np.testing.assert_allclose(gpu, cpu, rtol=1e-10)

    def test_crank_nicolson_matches_cpu(self):
        """Crank-Nicolson should match CPU to high precision."""
        N = 100
        min_Sigma = 1e-10
        Sigma, nu, X, dX = self._make_state(N)
        cpu = cpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma, theta=0.5,
        )
        gpu = gpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma, theta=0.5,
        )
        np.testing.assert_allclose(gpu, cpu, rtol=1e-10)

    def test_with_tidal_params(self):
        N = 100
        min_Sigma = 1e-10
        Sigma, nu, X, dX = self._make_state(N)
        tidal = {"cw": 0.2, "a_1": 1.5e15, "n_1": 5, "trunc_frac": 0.93}
        cpu = cpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma, tidal_params=tidal,
        )
        gpu = gpu_evol.evolve_surface_density(
            Sigma, 1.0, nu, X, dX, N, min_Sigma, tidal_params=tidal,
        )
        np.testing.assert_allclose(gpu, cpu, rtol=1e-10)

    def test_adaptive_theta_prevents_negative_sigma(self):
        """When Courant >> 1, adaptive theta falls back to backward Euler,
        keeping Sigma positive and physically bounded."""
        N = 1000
        r_in = 1e9
        r_out = 1e12
        r = np.linspace(r_in, r_out, N)
        X = X_func(r)
        dX = X[1] - X[0]
        min_Sigma = 1e-10

        # Create a sharp Sigma peak (simulating a deposition front)
        Sigma = np.full(N, 1.0)
        peak = N // 2
        Sigma[peak - 5:peak + 5] = 1000.0

        # High viscosity to push Courant number well above 1
        nu = np.full(N, 1e15)

        # Large dt that would make Courant >> 1 with CN
        dt_cfl = gpu_evol.calculate_timestep(X, nu, dX)
        dt = dt_cfl * 1000  # 1000x CFL → Courant ~ 1000

        result = gpu_evol.evolve_surface_density(
            Sigma, dt, nu, X, dX, N, min_Sigma, theta=0.5,
        )

        # Sigma must remain positive everywhere (no negative → clamped blowup)
        assert np.all(result >= min_Sigma), (
            f"Sigma went negative: min={result.min():.2e}"
        )
        # Sigma must not grow beyond the initial peak (diffusion only spreads)
        assert result.max() <= Sigma.max() * 1.1, (
            f"Sigma grew: max={result.max():.2e} vs initial {Sigma.max():.2e}"
        )

    def test_evaporation_capped(self):
        N = 100
        min_Sigma = 1e-5
        Sigma, nu, X, dX = self._make_state(N)
        Sigma[:] = min_Sigma * 2.0

        def heavy_evap(r_arr):
            return np.full_like(r_arr, 1e30)

        result = gpu_evol.evolve_surface_density(
            Sigma, 1e10, nu, X, dX, N, min_Sigma, evap_func=heavy_evap,
        )
        assert np.all(result >= min_Sigma)
