"""Vectorized surface-density evolution — pure array ops, no JIT.

Drop-in replacements for :func:`blackhole.evolution.calculate_timestep`,
:func:`blackhole.evolution.disk_evap`, and
:func:`blackhole.evolution.evolve_surface_density`.

``add_mass`` is NOT reimplemented here because it contains an inherently
sequential backward loop.  Use :func:`blackhole.evolution.add_mass` on
the CPU with a small host transfer (~824 bytes at N=103).
"""

import numpy as np

from blackhole.constants import G, M_sun, c
from blackhole.gpu import get_xp
from blackhole.gpu.disk_physics import R_func, S_factor, Sigma_from_S

# Cache for the tridiagonal CSR sparsity pattern (GPU fallback path).
_tridiag_cache: dict = {}

# Thomas algorithm CUDA kernel for GPU tridiagonal solve.
# Launched as a single-thread kernel; the Thomas algorithm is O(N) serial work
# but eliminates all cuSPARSE overhead (CSR construction, analysis phase, etc.).
_thomas_kernel = None


def _get_thomas_kernel():
    """Lazily compile the Thomas algorithm RawKernel."""
    global _thomas_kernel
    if _thomas_kernel is not None:
        return _thomas_kernel
    try:
        import cupy as cp

        _thomas_kernel = cp.RawKernel(
            r"""
extern "C" __global__
void thomas_solve(double* __restrict__ diag,
                  const double* __restrict__ lower,
                  const double* __restrict__ upper,
                  double* __restrict__ rhs,
                  double* __restrict__ x,
                  const int n) {
    // Forward elimination (modifies diag and rhs in-place)
    for (int i = 1; i < n; i++) {
        double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    // Backward substitution
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
    }
}
""",
            "thomas_solve",
        )
        return _thomas_kernel
    except Exception:
        return None


def calculate_timestep(X, nu, dX):
    """CFL-like stability condition for the viscous diffusion equation.

    Parameters
    ----------
    X : array
        X-coordinate grid.
    nu : array
        Kinematic viscosity array.
    dX : float
        Grid spacing in X.

    Returns
    -------
    float
        Stable timestep (s).
    """
    xp = get_xp(X, nu)
    finite_nu = nu[xp.isfinite(nu)]
    if finite_nu.size == 0:
        max_nu = 1.0
    else:
        max_nu = float(xp.max(finite_nu))
    if max_nu <= 0.0:
        max_nu = 1.0
    return 0.5 * float(X[0]) ** 2 * float(dX) ** 2 / (12.0 * max_nu)


def disk_evap(r, M_star, L_ratio=1.0):
    """X-ray evaporation rate at radius *r*.

    Parameters
    ----------
    r : array
        Radius (cm).
    M_star : float
        Central object mass (g).
    L_ratio : float, optional
        Ratio of actual luminosity to Eddington luminosity.

    Returns
    -------
    array
        Evaporation rate (g/s).
    """
    R_min = 5e8
    epsilon = 0.1 * (R_min / r) ** (-2)
    M_edd = 1.4e18 * (M_star / M_sun)
    R_s = 2.0 * G * M_star / c ** 2
    M_ev = 0.08 * M_edd * (
        (r / R_s) ** (1.0 / 4.0) + epsilon * (r / (800.0 * R_s)) ** 2
    ) ** (-1)
    return M_ev * L_ratio


def evolve_surface_density(Sigma, dt, nu, X, dX, N, min_Sigma,
                           tidal_params=None, evap_func=None, theta=0.0):
    """Evolve surface density by one timestep via viscous diffusion.

    Parameters
    ----------
    Sigma : array
        Surface density array.
    dt : float
        Timestep (s).
    nu : array
        Kinematic viscosity array.
    X : array
        X-coordinate grid.
    dX : float
        Grid spacing in X.
    N : int
        Number of grid cells.
    min_Sigma : float
        Minimum surface density floor.
    tidal_params : dict, optional
        Tidal torque parameters with keys ``cw``, ``a_1``, ``n_1``,
        ``trunc_frac``.
    evap_func : callable, optional
        Evaporation function ``evap_func(r)`` returning mass-loss rate.
    theta : float, optional
        Implicitness parameter for diffusion (0=explicit, 0.5=Crank-Nicolson,
        1=fully implicit).

    Returns
    -------
    array
        Updated surface density array.
    """
    xp = get_xp(Sigma, nu, X)

    S_arr = S_factor(X, Sigma)
    r = R_func(X)

    if theta <= 0.0:
        # Explicit forward Euler
        new_S = S_arr.copy()
        new_S[1:-1] += (12.0 * dt / (X[1:-1] ** 2 * dX ** 2)) * (
            S_arr[:-2] * nu[:-2]
            + S_arr[2:] * nu[2:]
            - 2.0 * S_arr[1:-1] * nu[1:-1]
        )
        new_S = xp.nan_to_num(new_S, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Implicit / Crank-Nicolson scheme
        coeff = 12.0 * dt / (X[1:-1] ** 2 * dX ** 2)

        # Adaptive theta: if the local Courant number c_i * nu_i exceeds 1,
        # the CN explicit part would produce oscillatory (negative) S values.
        # Fall back to backward Euler (theta=1.0) which guarantees positivity.
        max_courant = float(xp.max(coeff * nu[1:-1]))
        if max_courant > 1.0 and theta < 1.0:
            theta = 1.0

        # Explicit part
        rhs = S_arr[1:-1].copy()
        expl_weight = 1.0 - theta
        rhs += expl_weight * coeff * (
            nu[:-2] * S_arr[:-2]
            + nu[2:] * S_arr[2:]
            - 2.0 * nu[1:-1] * S_arr[1:-1]
        )

        # Tridiagonal system
        n_int = len(rhs)
        lower = -theta * coeff * nu[:-2]
        diag = 1.0 + 2.0 * theta * coeff * nu[1:-1]
        upper = -theta * coeff * nu[2:]

        # Boundary contributions
        rhs_arr = rhs.copy()
        rhs_arr[0] += theta * coeff[0] * nu[0] * S_arr[0]
        rhs_arr[-1] += theta * coeff[-1] * nu[-1] * S_arr[-1]

        if xp.__name__ == "cupy":
            _kern = _get_thomas_kernel()
            if _kern is not None:
                # Thomas algorithm via single CUDA kernel launch — O(N),
                # avoids CSR construction + cuSPARSE overhead entirely.
                # The kernel modifies diag/rhs in-place during forward
                # elimination, so we pass copies.
                diag_buf = diag.copy()
                rhs_buf = rhs_arr.copy()
                new_S_interior = xp.empty(n_int, dtype=xp.float64)
                _kern(
                    (1,), (1,),
                    (diag_buf, lower, upper, rhs_buf,
                     new_S_interior, np.int32(n_int)),
                )
            else:
                # Fallback: cuSPARSE sparse tridiagonal solve.
                import cupyx.scipy.sparse as sp_sparse
                import cupyx.scipy.sparse.linalg as sp_linalg

                if n_int not in _tridiag_cache:
                    nnz = 3 * n_int - 2
                    indptr = np.empty(n_int + 1, dtype=np.int32)
                    indices = np.empty(nnz, dtype=np.int32)
                    indptr[0] = 0
                    indptr[1] = 2
                    indices[0], indices[1] = 0, 1
                    for i in range(1, n_int - 1):
                        indptr[i + 1] = indptr[i] + 3
                        base = 2 + (i - 1) * 3
                        indices[base] = i - 1
                        indices[base + 1] = i
                        indices[base + 2] = i + 1
                    indptr[n_int] = indptr[n_int - 1] + 2
                    indices[nnz - 2] = n_int - 2
                    indices[nnz - 1] = n_int - 1
                    _tridiag_cache[n_int] = (
                        xp.asarray(indptr),
                        xp.asarray(indices),
                    )
                cached_indptr, cached_indices = _tridiag_cache[n_int]
                nnz = 3 * n_int - 2
                data = xp.empty(nnz, dtype=xp.float64)
                data[0::3][:n_int] = diag
                data[1::3][:n_int - 1] = upper[:-1]
                data[2::3][:n_int - 1] = lower[1:]
                A = sp_sparse.csr_matrix(
                    (data, cached_indices, cached_indptr),
                    shape=(n_int, n_int),
                )
                new_S_interior = sp_linalg.spsolve(A, rhs_arr)
        else:
            # CPU path: optimal LAPACK Thomas algorithm via banded solver.
            from scipy.linalg import solve_banded as sp_solve_banded

            ab = xp.empty((3, n_int))
            ab[0, 0] = 0.0
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]
            ab[2, -1] = 0.0
            new_S_interior = sp_solve_banded(
                (1, 1), ab, rhs_arr, check_finite=False,
            )

        new_S = S_arr.copy()
        new_S[1:-1] = new_S_interior
        new_S = xp.nan_to_num(new_S, nan=0.0, posinf=0.0, neginf=0.0)

    new_Sigma = Sigma_from_S(new_S, X)
    new_Sigma[-1] = new_Sigma[-2]

    # Tidal torques
    if tidal_params is not None:
        cw = tidal_params["cw"]
        a_1 = tidal_params["a_1"]
        n_1 = tidal_params["n_1"]
        trunc_frac = tidal_params["trunc_frac"]

        trunc_rad = int(N * trunc_frac)

        T_tid = (
            cw * r[1:-1] * nu[1:-1] * new_Sigma[1:-1]
            * (r[1:-1] / a_1) ** n_1
        )
        tidal_torque = dt / (2.0 * xp.pi * r[1:-1]) * T_tid
        tidal_torque = xp.nan_to_num(tidal_torque, nan=0.0, posinf=0.0, neginf=0.0)

        if trunc_rad < len(tidal_torque):
            new_Sigma[1 + trunc_rad:-1] -= tidal_torque[trunc_rad:]

    # Evaporation
    if evap_func is not None:
        dr = xp.diff(r)
        dr = xp.concatenate([dr, dr[-1:]])
        M_ev = evap_func(r[1:-1])
        dSigma_evap = M_ev * dt / (2.0 * xp.pi * r[1:-1] * dr[1:-1])
        available = xp.maximum(new_Sigma[1:-1] - min_Sigma, 0.0)
        dSigma_evap = xp.minimum(dSigma_evap, available)
        new_Sigma[1:-1] -= dSigma_evap

    # Final cleanup
    new_Sigma = xp.nan_to_num(new_Sigma, nan=min_Sigma, posinf=min_Sigma, neginf=min_Sigma)
    new_Sigma = xp.maximum(new_Sigma, min_Sigma)
    return new_Sigma
