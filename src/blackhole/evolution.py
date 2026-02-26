"""Surface density time-evolution, mass addition, tidal torques, evaporation.

Extracted from the GPU_timedep notebook. Global state (j_val, dMj, dMj1)
has been eliminated; add_mass returns a MassTransferResult dataclass instead.
"""

import dataclasses

import numpy as np
from scipy.linalg import solve_banded

from blackhole.constants import G, M_sun, c
from blackhole.disk_physics import Marr, R_func, S_factor, Sigma_from_S


@dataclasses.dataclass
class MassTransferResult:
    """Result of mass addition with angular momentum conservation.

    Attributes
    ----------
    Sigma : np.ndarray
        Updated surface density array.
    j_val : int
        Grid index where mass is deposited.
    dMj : float
        Mass deposited at cell j.
    dMj1 : float
        Mass deposited at cell j-1.
    """

    Sigma: np.ndarray
    j_val: int
    dMj: float
    dMj1: float


def calculate_timestep(X, nu, dX):
    """CFL-like stability condition for the viscous diffusion equation.

    Parameters
    ----------
    X : np.ndarray
        X-coordinate grid.
    nu : np.ndarray
        Kinematic viscosity array.
    dX : float
        Grid spacing in X.

    Returns
    -------
    float
        Stable timestep (s).
    """
    finite_nu = nu[np.isfinite(nu)]
    max_nu = float(np.max(finite_nu)) if finite_nu.size > 0 else 1.0
    if max_nu <= 0.0:
        max_nu = 1.0
    return 0.5 * float(X[0]) ** 2 * dX ** 2 / (12.0 * max_nu)


def disk_evap(r, M_star, L_ratio=1.0):
    """X-ray evaporation rate at radius *r*.

    The evaporation rate is scaled by *L_ratio* = L_actual / L_edd so that
    evaporation is reduced when the accretion luminosity is sub-Eddington
    (Liu et al. 2002).

    Parameters
    ----------
    r : float or np.ndarray
        Radius (cm).
    M_star : float
        Central object mass (g).
    L_ratio : float, optional
        Ratio of actual accretion luminosity to Eddington luminosity.
        Default 1.0 (full Eddington-rate evaporation, backward compatible).

    Returns
    -------
    float or np.ndarray
        Evaporation rate (g/s).
    """
    R_min = 5e8
    epsilon = 0.1 * (R_min / r) ** (-2)
    M_edd = 1.4e18 * (M_star / M_sun)  # g/s
    R_s = 2.0 * G * M_star / c ** 2
    M_ev = 0.08 * M_edd * (
        (r / R_s) ** (1.0 / 4.0) + epsilon * (r / (800.0 * R_s)) ** 2
    ) ** (-1)
    return M_ev * L_ratio


def add_mass(Sigma, M_dot, dt, X, N, X_K, X_N, dX, min_Sigma, sigma_cap=200.0):
    """Add mass at the outer disk edge with angular momentum conservation.

    The notebook version mutated globals ``j_val``, ``dMj``, ``dMj1``.
    This version returns them in a :class:`MassTransferResult`.

    Parameters
    ----------
    Sigma : np.ndarray
        Surface density array (length N).
    M_dot : float
        Mass transfer rate (g/s).
    dt : float
        Timestep (s).
    X : np.ndarray
        X-coordinate grid (length N).
    N : int
        Number of grid cells.
    X_K : float
        X-coordinate of the circularisation radius.
    X_N : float
        X-coordinate of the outer disk edge.
    dX : float
        Grid spacing in X.
    min_Sigma : float
        Minimum surface density floor.
    sigma_cap : float, optional
        Safety cap for deposited surface density.  Mass is only deposited
        when the computed Sigma at the deposition cell is below this value.
        Default 200.0 (appropriate for stellar-mass systems).  For SMBHs,
        scale from ``Sigma_max`` to allow higher surface densities.

    Returns
    -------
    MassTransferResult
        Updated Sigma and mass-transfer bookkeeping values.
    """
    dM = M_dot * dt
    X_K_dM = dM * X_K

    Mass = Marr(X, Sigma, dX)

    j_val = N - 1
    dMj = 0.0
    dMj1 = 0.0

    new_Sigma = Sigma.copy()

    for j in range(N - 1, 0, -1):
        massarr = Mass[j:]
        xmassarr = X[j:] * Mass[j:]

        sum_dM_i = float(np.sum(massarr))
        X_sum_dM_i = float(np.sum(xmassarr))

        D_J = (X_K_dM + X_sum_dM_i) / (dM + sum_dM_i)

        if D_J > X[j - 1]:
            dM_J = ((X[j] - D_J) / dX) * (dM + sum_dM_i)
            dM_J_minus_1 = ((D_J - X[j - 1]) / dX) * (dM + sum_dM_i) + Mass[j - 1]

            Sj = 4.0 * dM_J / (np.pi * X[j] ** 2 * dX)
            Sj1 = 4.0 * dM_J_minus_1 / (np.pi * X[j - 1] ** 2 * dX)

            sjx = Sj / X[j + 1] if j + 1 < N else Sj / X[j]
            sj1x = Sj1 / X[j]

            if sjx < sigma_cap:
                new_Sigma[j] = max(sjx, min_Sigma)
            if sj1x < sigma_cap:
                new_Sigma[j - 1] = max(sj1x, min_Sigma)

            j_val = j
            dMj = dM_J
            dMj1 = dM_J_minus_1
            break

    return MassTransferResult(
        Sigma=new_Sigma,
        j_val=j_val,
        dMj=dMj,
        dMj1=dMj1,
    )


def evolve_surface_density(Sigma, dt, nu, X, dX, N, min_Sigma,
                           tidal_params=None, evap_func=None, theta=0.0):
    """Evolve surface density by one timestep via viscous diffusion.

    Parameters
    ----------
    Sigma : np.ndarray
        Surface density array.
    dt : float
        Timestep (s).
    nu : np.ndarray
        Kinematic viscosity array.
    X : np.ndarray
        X-coordinate grid.
    dX : float
        Grid spacing in X.
    N : int
        Number of grid cells.
    min_Sigma : float
        Minimum surface density floor.
    tidal_params : dict, optional
        Tidal torque parameters with keys ``cw``, ``a_1``, ``n_1``,
        ``trunc_frac``. If None, tidal torques are not applied.
    evap_func : callable, optional
        Evaporation function ``evap_func(r)`` returning mass-loss rate.
        If None, no evaporation is applied.
    theta : float, optional
        Implicitness parameter for diffusion: 0.0 = explicit forward Euler
        (default), 0.5 = Crank-Nicolson, 1.0 = fully implicit backward Euler.
        Values > 0 are unconditionally stable, allowing larger timesteps.

    Returns
    -------
    np.ndarray
        Updated surface density array.
    """
    S_arr = S_factor(X, Sigma)

    r = R_func(X)

    if theta <= 0.0:
        # Explicit forward Euler (original scheme)
        new_S = S_arr.copy()
        new_S[1:-1] += (12.0 * dt / (X[1:-1] ** 2 * dX ** 2)) * (
            S_arr[:-2] * nu[:-2]
            + S_arr[2:] * nu[2:]
            - 2.0 * S_arr[1:-1] * nu[1:-1]
        )
        np.nan_to_num(new_S, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Implicit/Crank-Nicolson scheme: solve tridiagonal system
        # c_i = 12 * dt / (X_i^2 * dX^2)
        c = 12.0 * dt / (X[1:-1] ** 2 * dX ** 2)

        # Adaptive theta: if the local Courant number c_i * nu_i exceeds 1,
        # the CN explicit part would produce oscillatory (negative) S values.
        # Fall back to backward Euler (theta=1.0) which guarantees positivity.
        max_courant = float(np.max(c * nu[1:-1]))
        if max_courant > 1.0 and theta < 1.0:
            theta = 1.0

        # Explicit part: rhs = S^n + (1-theta) * explicit_diffusion
        rhs = S_arr[1:-1].copy()
        expl_weight = 1.0 - theta
        rhs += expl_weight * c * (
            nu[:-2] * S_arr[:-2]
            + nu[2:] * S_arr[2:]
            - 2.0 * nu[1:-1] * S_arr[1:-1]
        )

        # Implicit part: tridiagonal matrix (1 + 2*theta*c*nu_i) on diagonal
        n_int = len(rhs)  # N - 2 interior points
        # sub-diagonal (lower): -theta * c_i * nu_{i-1}
        lower = -theta * c * nu[:-2]
        # diagonal: 1 + 2*theta * c_i * nu_i
        diag = 1.0 + 2.0 * theta * c * nu[1:-1]
        # super-diagonal (upper): -theta * c_i * nu_{i+1}
        upper = -theta * c * nu[2:]

        # Boundary contributions: S[0] and S[N-1] are known (unchanged)
        # For i=1 (first interior): the lower term references S[0]
        rhs[0] += theta * c[0] * nu[0] * S_arr[0]
        # For i=N-2 (last interior): the upper term references S[N-1]
        rhs[-1] += theta * c[-1] * nu[-1] * S_arr[-1]

        # Pack into banded form for solve_banded: shape (3, n_int)
        # Row 0 = super-diagonal (first element unused)
        # Row 1 = diagonal
        # Row 2 = sub-diagonal (last element unused)
        ab = np.empty((3, n_int))
        ab[0, 0] = 0.0  # unused
        ab[0, 1:] = upper[:-1]   # A[k, k+1] for k=0..n_int-2
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]   # A[k, k-1] for k=1..n_int-1
        ab[2, -1] = 0.0  # unused

        new_S_interior = solve_banded((1, 1), ab, rhs)

        new_S = S_arr.copy()
        new_S[1:-1] = new_S_interior
        np.nan_to_num(new_S, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

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
        tidal_torque = dt / (2.0 * np.pi * r[1:-1]) * T_tid

        np.nan_to_num(tidal_torque, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if trunc_rad < len(tidal_torque):
            new_Sigma[1 + trunc_rad:-1] -= tidal_torque[trunc_rad:]

    # Evaporation: disk_evap returns a total mass-flow rate M_ev (g/s) at
    # each radius.  Convert to a surface-density loss rate by dividing by the
    # ring area 2*pi*r*dr, where dr is the radial cell width.
    if evap_func is not None:
        dr = np.diff(r)
        dr = np.append(dr, dr[-1])  # pad last cell
        M_ev = evap_func(r[1:-1])
        dSigma_evap = M_ev * dt / (2.0 * np.pi * r[1:-1] * dr[1:-1])
        # Cap: cannot remove more mass than available above the floor
        available = np.maximum(new_Sigma[1:-1] - min_Sigma, 0.0)
        dSigma_evap = np.minimum(dSigma_evap, available)
        new_Sigma[1:-1] -= dSigma_evap

    # Final cleanup: replace any remaining NaN/inf and enforce floor
    np.nan_to_num(new_Sigma, copy=False, nan=min_Sigma, posinf=min_Sigma, neginf=min_Sigma)
    new_Sigma = np.maximum(new_Sigma, min_Sigma)
    return new_Sigma
