"""Vectorized surface-density evolution — pure array ops, no JIT.

Drop-in replacements for :func:`blackhole.evolution.calculate_timestep`,
:func:`blackhole.evolution.disk_evap`, and
:func:`blackhole.evolution.evolve_surface_density`.

``add_mass`` is NOT reimplemented here because it contains an inherently
sequential backward loop.  Use :func:`blackhole.evolution.add_mass` on
the CPU with a small host transfer (~824 bytes at N=103).
"""

from blackhole.constants import G, M_sun, c
from blackhole.gpu import get_xp
from blackhole.gpu.disk_physics import R_func, S_factor, Sigma_from_S


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
            # GPU-native sparse tridiagonal solve via cuSPARSE.
            # For tridiagonal matrices LU has zero fill-in → O(N).
            import cupyx.scipy.sparse as sp_sparse
            import cupyx.scipy.sparse.linalg as sp_linalg

            A = sp_sparse.diags(
                [lower[1:], diag, upper[:-1]], [-1, 0, 1],
                shape=(n_int, n_int), format="csr",
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
