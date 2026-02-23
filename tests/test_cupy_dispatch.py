"""Tests verifying CuPy array dispatch for all physics modules.

Each test feeds CuPy arrays into module functions and checks:
1. The result is a CuPy array (dispatch worked).
2. The numerical result matches the NumPy path within tolerance.

All tests auto-skip on CPU-only machines (no CuPy installed).
"""

import numpy as np

from conftest import requires_cupy

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_cupy(*arrays):
    """Convert numpy arrays to cupy arrays."""
    import cupy
    return tuple(cupy.asarray(a) for a in arrays)


def _assert_cupy_close(cp_result, np_result, rtol=1e-10):
    """Assert CuPy result matches NumPy result and is a CuPy array."""
    import cupy
    assert isinstance(cp_result, cupy.ndarray), "Result should be a CuPy array"
    np.testing.assert_allclose(cupy.asnumpy(cp_result), np_result, rtol=rtol)


# ---------------------------------------------------------------------------
# opacity.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestOpacityDispatch:
    def test_kappa_e(self):
        from blackhole.opacity import kappa_e
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e5, 1e7])
        np_result = kappa_e(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_e(cp_rho, cp_T), np_result)

    def test_kappa_K(self):
        from blackhole.opacity import kappa_K
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e5, 1e7])
        np_result = kappa_K(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_K(cp_rho, cp_T), np_result)

    def test_kappa_Hminus(self):
        from blackhole.opacity import kappa_Hminus
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e3, 1e4])
        np_result = kappa_Hminus(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_Hminus(cp_rho, cp_T), np_result)

    def test_kappa_cond(self):
        from blackhole.opacity import kappa_cond
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e6, 1e8])
        np_result = kappa_cond(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_cond(cp_rho, cp_T), np_result)

    def test_kappa_tot(self):
        from blackhole.opacity import kappa_tot
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e5, 1e7])
        np_result = kappa_tot(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_tot(cp_rho, cp_T), np_result)

    def test_kappa_ff(self):
        from blackhole.opacity import kappa_ff
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e5, 1e7])
        np_result = kappa_ff(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_ff(cp_rho, cp_T), np_result)

    def test_kappa_simple(self):
        from blackhole.opacity import kappa_simple
        rho = np.array([1e-8, 1e-6])
        T = np.array([1e5, 1e7])
        np_result = kappa_simple(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(kappa_simple(cp_rho, cp_T), np_result)


# ---------------------------------------------------------------------------
# viscosity.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestViscosityDispatch:
    def test_alpha_visc(self):
        from blackhole.viscosity import alpha_visc
        T_c = np.array([1e3, 2.5e4, 1e6])
        np_result = alpha_visc(T_c)
        (cp_T_c,) = _to_cupy(T_c)
        _assert_cupy_close(alpha_visc(cp_T_c), np_result)


# ---------------------------------------------------------------------------
# steady_state.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestSteadyStateDispatch:
    def test_f_boundary(self):
        from blackhole.constants import M_sun
        from blackhole.steady_state import f_boundary
        r = np.array([1e9, 1e10, 1e11])
        M = 10.0 * M_sun
        np_result = f_boundary(r, M)
        (cp_r,) = _to_cupy(r)
        _assert_cupy_close(f_boundary(cp_r, M), np_result)

    def test_border_inner_middle(self):
        from blackhole.constants import M_sun
        from blackhole.steady_state import border_inner_middle
        alpha = np.array([0.04, 0.2])
        M = 10.0 * M_sun
        M_dot = np.array([1e16, 1e17])
        np_result = border_inner_middle(alpha, M, M_dot)
        cp_alpha, cp_M_dot = _to_cupy(alpha, M_dot)
        _assert_cupy_close(border_inner_middle(cp_alpha, M, cp_M_dot), np_result)

    def test_border_middle_outer(self):
        from blackhole.steady_state import border_middle_outer
        M_dot = np.array([1e16, 1e17])
        np_result = border_middle_outer(M_dot)
        (cp_M_dot,) = _to_cupy(M_dot)
        _assert_cupy_close(border_middle_outer(cp_M_dot), np_result)


# ---------------------------------------------------------------------------
# disk_physics.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestDiskPhysicsDispatch:
    def test_X_func(self):
        from blackhole.disk_physics import X_func
        r = np.array([1e8, 1e10, 1e12])
        np_result = X_func(r)
        (cp_r,) = _to_cupy(r)
        _assert_cupy_close(X_func(cp_r), np_result)

    def test_R_func(self):
        from blackhole.disk_physics import R_func
        x = np.array([1e4, 1e5, 1e6])
        np_result = R_func(x)
        (cp_x,) = _to_cupy(x)
        _assert_cupy_close(R_func(cp_x), np_result)

    def test_omega(self):
        from blackhole.constants import M_sun
        from blackhole.disk_physics import omega
        R = np.array([1e9, 1e10, 1e11])
        M = 10.0 * M_sun
        np_result = omega(R, M)
        (cp_R,) = _to_cupy(R)
        _assert_cupy_close(omega(cp_R, M), np_result)

    def test_pressure(self):
        from blackhole.disk_physics import pressure
        H = np.array([1e8, 1e9])
        Sigma = np.array([1e2, 1e3])
        T = np.array([1e5, 1e6])
        np_result = pressure(H, Sigma, T)
        cp_H, cp_Sigma, cp_T = _to_cupy(H, Sigma, T)
        _assert_cupy_close(pressure(cp_H, cp_Sigma, cp_T), np_result)

    def test_Marr(self):
        from blackhole.disk_physics import Marr
        X = np.array([1e4, 2e4, 3e4])
        Sigma = np.array([1e2, 2e2, 3e2])
        dX = 1e4
        np_result = Marr(X, Sigma, dX)
        cp_X, cp_Sigma = _to_cupy(X, Sigma)
        _assert_cupy_close(Marr(cp_X, cp_Sigma, dX), np_result)


# ---------------------------------------------------------------------------
# irradiation.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestIrradiationDispatch:
    def test_Sigma_max(self):
        from blackhole.constants import M_sun
        from blackhole.irradiation import Sigma_max
        eps = np.array([0.0, 0.01, 0.1])
        r = np.array([1e10, 2e10, 3e10])
        M = 10.0 * M_sun
        np_result = Sigma_max(eps, r, M)
        cp_eps, cp_r = _to_cupy(eps, r)
        _assert_cupy_close(Sigma_max(cp_eps, cp_r, M), np_result)

    def test_Sigma_min(self):
        from blackhole.constants import M_sun
        from blackhole.irradiation import Sigma_min
        eps = np.array([0.0, 0.01, 0.1])
        r = np.array([1e10, 2e10, 3e10])
        M = 10.0 * M_sun
        np_result = Sigma_min(eps, r, M)
        cp_eps, cp_r = _to_cupy(eps, r)
        _assert_cupy_close(Sigma_min(cp_eps, cp_r, M), np_result)

    def test_T_c_max(self):
        from blackhole.irradiation import T_c_max
        eps = np.array([0.0, 0.01, 0.1])
        r = np.array([1e10, 2e10, 3e10])
        np_result = T_c_max(eps, r)
        cp_eps, cp_r = _to_cupy(eps, r)
        _assert_cupy_close(T_c_max(cp_eps, cp_r), np_result)

    def test_T_c_min(self):
        from blackhole.constants import M_sun
        from blackhole.irradiation import T_c_min
        eps = np.array([0.0, 0.01, 0.1])
        r = np.array([1e10, 2e10, 3e10])
        M = 10.0 * M_sun
        np_result = T_c_min(eps, r, M)
        cp_eps, cp_r = _to_cupy(eps, r)
        _assert_cupy_close(T_c_min(cp_eps, cp_r, M), np_result)

    def test_alpha_visc_irr(self):
        from blackhole.constants import M_sun
        from blackhole.irradiation import alpha_visc_irr
        T_c = np.array([1e3, 2.5e4, 1e6])
        eps = np.array([0.0, 0.01, 0.1])
        r = np.array([1e10, 2e10, 3e10])
        M = 10.0 * M_sun
        np_result = alpha_visc_irr(T_c, eps, r, M)
        cp_T_c, cp_eps, cp_r = _to_cupy(T_c, eps, r)
        _assert_cupy_close(alpha_visc_irr(cp_T_c, cp_eps, cp_r, M), np_result)


# ---------------------------------------------------------------------------
# evolution.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestEvolutionDispatch:
    def test_calculate_timestep(self):
        from blackhole.evolution import calculate_timestep
        X = np.linspace(1e4, 1e6, 100)
        nu = np.full(100, 1e14)
        dX = X[1] - X[0]
        np_result = calculate_timestep(X, nu, dX)
        cp_X, cp_nu = _to_cupy(X, nu)
        cp_result = calculate_timestep(cp_X, cp_nu, dX)
        assert isinstance(cp_result, float)
        np.testing.assert_allclose(cp_result, np_result, rtol=1e-10)

    def test_evolve_surface_density(self):
        from blackhole.evolution import evolve_surface_density
        N = 50
        X = np.linspace(1e4, 1e6, N)
        dX = X[1] - X[0]
        Sigma = np.full(N, 100.0)
        nu = np.full(N, 1e14)
        dt = 1.0
        np_result = evolve_surface_density(Sigma, dt, nu, X, dX, N, 1e-10)
        cp_Sigma, cp_nu, cp_X = _to_cupy(Sigma, nu, X)
        cp_result = evolve_surface_density(cp_Sigma, dt, cp_nu, cp_X, dX, N, 1e-10)
        _assert_cupy_close(cp_result, np_result)


# ---------------------------------------------------------------------------
# solvers.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestSolversDispatch:
    def test_Y_energy(self):
        from blackhole.constants import M_sun
        from blackhole.solvers import Y_energy
        R = np.linspace(1e9, 1e11, 20)
        M = 10.0 * M_sun
        j_val = 15
        dMj = 1e20
        dMj1 = 2e20
        M_dot = 1e17
        np_y1, np_y2 = Y_energy(R, j_val, dMj, dMj1, M, M_dot)
        (cp_R,) = _to_cupy(R)
        cp_y1, cp_y2 = Y_energy(cp_R, j_val, dMj, dMj1, M, M_dot)
        # Results are scalar-like (single element from array indexing)
        np.testing.assert_allclose(float(cp_y1), float(np_y1), rtol=1e-10)
        np.testing.assert_allclose(float(cp_y2), float(np_y2), rtol=1e-10)


# ---------------------------------------------------------------------------
# luminosity.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestLuminosityDispatch:
    def _make_arrays(self, N=20):
        from blackhole.constants import M_sun
        r = np.linspace(1e9, 1e11, N)
        dr = r[1] - r[0]
        Sigma = np.full(N, 100.0)
        H = np.full(N, 1e8)
        alpha = 0.1
        M_star = 10.0 * M_sun
        return Sigma, H, alpha, r, dr, M_star

    def test_L_rad(self):
        from blackhole.luminosity import L_rad
        Sigma, H, alpha, r, dr, M_star = self._make_arrays()
        np_result = L_rad(Sigma, H, alpha, r, dr, M_star)
        cp_Sigma, cp_H, cp_r = _to_cupy(Sigma, H, r)
        cp_result = L_rad(cp_Sigma, cp_H, alpha, cp_r, dr, M_star)
        assert isinstance(cp_result, float)
        np.testing.assert_allclose(cp_result, np_result, rtol=1e-10)

    def test_L_rad_array(self):
        from blackhole.luminosity import L_rad_array
        Sigma, H, alpha, r, dr, M_star = self._make_arrays()
        np_result = L_rad_array(Sigma, H, alpha, r, dr, M_star)
        cp_Sigma, cp_H, cp_r = _to_cupy(Sigma, H, r)
        _assert_cupy_close(
            L_rad_array(cp_Sigma, cp_H, alpha, cp_r, dr, M_star),
            np_result,
        )

    def test_T_eff(self):
        from blackhole.luminosity import T_eff
        Sigma, H, alpha, r, _, M_star = self._make_arrays()
        np_result = T_eff(Sigma, H, alpha, r, M_star)
        cp_Sigma, cp_H, cp_r = _to_cupy(Sigma, H, r)
        _assert_cupy_close(
            T_eff(cp_Sigma, cp_H, alpha, cp_r, M_star),
            np_result,
        )


# ---------------------------------------------------------------------------
# cr_solvers.py
# ---------------------------------------------------------------------------

@requires_cupy
class TestCRSolversDispatch:
    def test_p_rad(self):
        from blackhole.cr_solvers import p_rad
        T = np.array([1e4, 1e6, 1e8])
        np_result = p_rad(T)
        (cp_T,) = _to_cupy(T)
        _assert_cupy_close(p_rad(cp_T), np_result)

    def test_p_tot(self):
        from blackhole.cr_solvers import p_tot
        rho = np.array([1e-8, 1e-6, 1e-4])
        T = np.array([1e4, 1e6, 1e8])
        np_result = p_tot(rho, T)
        cp_rho, cp_T = _to_cupy(rho, T)
        _assert_cupy_close(p_tot(cp_rho, cp_T), np_result)

    def test_omega_kepler(self):
        from blackhole.constants import M_sun
        from blackhole.cr_solvers import omega_kepler
        r = np.array([1e9, 1e10, 1e11])
        np_result = omega_kepler(r, M_sun)
        (cp_r,) = _to_cupy(r)
        _assert_cupy_close(omega_kepler(cp_r, M_sun), np_result)

    def test_h_2(self):
        from blackhole.constants import M_sun
        from blackhole.cr_solvers import h_2
        r = np.array([1e9, 1e10, 1e11])
        rho = np.array([1e-8, 1e-7, 1e-6])
        T = np.array([1e5, 1e6, 1e7])
        np_result = h_2(r, rho, M_sun, 5e16, T)
        cp_r, cp_rho, cp_T = _to_cupy(r, rho, T)
        _assert_cupy_close(h_2(cp_r, cp_rho, M_sun, 5e16, cp_T), np_result)
