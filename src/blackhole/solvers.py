"""Energy balance and hydrostatic equilibrium Newton solvers.

Extracted from the GPU_timedep notebook. Functions that referenced globals
(M_star, opacity, etc.) now take them as explicit parameters.
"""

from scipy.optimize import newton

from blackhole import get_xp
from blackhole.constants import G, a_rad, c
from blackhole.disk_physics import (
    kinematic_viscosity,
    omega,
    pressure,
    pressure_2,
)
from blackhole.opacity import kappa_bath

# ---------------------------------------------------------------------------
# Flux functions
# ---------------------------------------------------------------------------

def F_visc(H, Sigma, R, alpha, M_star):
    """Viscous heating rate per unit area.

    Parameters
    ----------
    H : float or np.ndarray
        Scale height (cm).
    Sigma : float or np.ndarray
        Surface density (g/cm^2).
    R : float or np.ndarray
        Radius (cm).
    alpha : float or np.ndarray
        Viscosity parameter.
    M_star : float
        Central object mass (g).

    Returns
    -------
    float or np.ndarray
        Viscous dissipation flux (erg/cm^2/s).
    """
    nu = kinematic_viscosity(H, R, alpha, M_star)
    return (9.0 / 8.0) * nu * Sigma * omega(R, M_star) ** 2


def F_rad(H, Sigma, T, opacity_func=None):
    """Radiative cooling flux.

    Parameters
    ----------
    H : float or np.ndarray
        Scale height (cm).
    Sigma : float or np.ndarray
        Surface density (g/cm^2).
    T : float or np.ndarray
        Midplane temperature (K).
    opacity_func : callable, optional
        Opacity function ``f(H, Sigma, T)``. Defaults to :func:`kappa_bath`.

    Returns
    -------
    float or np.ndarray
        Radiative flux (erg/cm^2/s).
    """
    if opacity_func is None:
        opacity_func = kappa_bath
    kappa = opacity_func(H, Sigma, T)
    return (2.0 * a_rad * c * T ** 4) / (3.0 * Sigma * kappa)


# ---------------------------------------------------------------------------
# Root functions for solvers
# ---------------------------------------------------------------------------

def energy_balance(T, H, Sigma, R, alpha, M_star, F_irr=0.0, opacity_func=None):
    """Root function for the temperature solver.

    Returns zero when viscous heating + irradiation = radiative cooling.

    Parameters
    ----------
    T : float
        Trial midplane temperature (K).
    H : float
        Scale height (cm).
    Sigma : float
        Surface density (g/cm^2).
    R : float
        Radius (cm).
    alpha : float
        Viscosity parameter.
    M_star : float
        Central object mass (g).
    F_irr : float, optional
        Irradiation flux (erg/cm^2/s).
    opacity_func : callable, optional
        Opacity function ``f(H, Sigma, T)``.

    Returns
    -------
    float
        Residual: ``2*F_visc + F_irr - F_rad``.
    """
    return (
        2.0 * F_visc(H, Sigma, R, alpha, M_star)
        + F_irr
        - F_rad(H, Sigma, T, opacity_func)
    )


def pressure_balance(H, Sigma, R, T, M_star):
    """Root function for the scale-height solver.

    Returns zero when gas+radiation pressure = hydrostatic equilibrium.

    Parameters
    ----------
    H : float
        Trial scale height (cm).
    Sigma : float
        Surface density (g/cm^2).
    R : float
        Radius (cm).
    T : float
        Midplane temperature (K).
    M_star : float
        Central object mass (g).

    Returns
    -------
    float
        Residual: ``pressure - pressure_2``.
    """
    return pressure(H, Sigma, T) - pressure_2(H, Sigma, R, M_star)


# ---------------------------------------------------------------------------
# Newton solvers
# ---------------------------------------------------------------------------

def solve_temperature(H, Sigma, r, T_c, alpha, M_star,
                      F_irr=None, opacity_func=None):
    """Solve for midplane temperature at each grid point.

    Uses Newton's method with fallback initial guesses.

    Parameters
    ----------
    H : np.ndarray
        Scale height array (cm).
    Sigma : np.ndarray
        Surface density array (g/cm^2).
    r : np.ndarray
        Radius array (cm).
    T_c : np.ndarray
        Current midplane temperature array (K).
    alpha : np.ndarray or float
        Viscosity parameter (scalar or per-cell).
    M_star : float
        Central object mass (g).
    F_irr : np.ndarray, optional
        Irradiation flux at each point. If None, zero everywhere.
    opacity_func : callable, optional
        Opacity function ``f(H, Sigma, T)``.

    Returns
    -------
    np.ndarray
        Updated temperature array (does not mutate input).
    """
    xp = get_xp(H, Sigma, r, T_c)
    T_new = T_c.copy()
    alpha_arr = xp.broadcast_to(alpha, Sigma.shape)
    T_min_limit = 1e0
    T_max_limit = 2e10
    initial_guesses = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

    for i in range(len(Sigma)):
        if float(Sigma[i]) <= 1e-100:
            continue

        f_irr_i = 0.0 if F_irr is None else float(F_irr[i])
        args = (
            float(H[i]), float(Sigma[i]), float(r[i]),
            float(alpha_arr[i]), M_star, f_irr_i, opacity_func,
        )

        solved = False
        # Primary guess: 1.5x current value
        try:
            new_T = newton(energy_balance, float(T_c[i]) * 1.5, args=args)
            if T_min_limit < new_T < T_max_limit:
                T_new[i] = new_T
                solved = True
        except RuntimeError:
            pass

        # Fallback guesses
        if not solved:
            for guess in initial_guesses:
                try:
                    new_T = newton(energy_balance, guess, args=args)
                    if T_min_limit < new_T < T_max_limit:
                        T_new[i] = new_T
                        break
                except RuntimeError:
                    continue

    return T_new


def solve_scale_height(H, Sigma, r, T_c, M_star):
    """Solve for scale height at each grid point.

    Uses Newton's method with fallback initial guesses.

    Parameters
    ----------
    H : np.ndarray
        Current scale height array (cm).
    Sigma : np.ndarray
        Surface density array (g/cm^2).
    r : np.ndarray
        Radius array (cm).
    T_c : np.ndarray
        Midplane temperature array (K).
    M_star : float
        Central object mass (g).

    Returns
    -------
    np.ndarray
        Updated scale height array (does not mutate input).
    """
    H_new = H.copy()
    H_min_limit = 1e7
    H_max_limit = 2e10
    initial_guesses = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]

    for i in range(len(Sigma)):
        if float(Sigma[i]) <= 1e-100:
            continue

        args = (float(Sigma[i]), float(r[i]), float(T_c[i]), M_star)

        solved = False
        try:
            H_val = newton(pressure_balance, float(H[i]) * 1.5, args=args)
            if H_min_limit < H_val < H_max_limit:
                H_new[i] = H_val
                solved = True
        except RuntimeError:
            pass

        if not solved:
            for guess in initial_guesses:
                try:
                    H_val = newton(pressure_balance, guess, args=args)
                    if H_min_limit < H_val < H_max_limit:
                        H_new[i] = H_val
                        break
                except RuntimeError:
                    continue

    return H_new


# ---------------------------------------------------------------------------
# Energy generation at mass-transfer zones
# ---------------------------------------------------------------------------

def Y_energy(R, j_val, dMj, dMj1, M_star, M_dot):
    """Energy generation rate at mass-transfer transition zones.

    Parameters
    ----------
    R : np.ndarray
        Radius array (cm).
    j_val : int
        Grid index of the mass deposition cell.
    dMj : float
        Mass deposited at cell j.
    dMj1 : float
        Mass deposited at cell j-1.
    M_star : float
        Central object mass (g).
    M_dot : float
        Mass transfer rate (g/s).

    Returns
    -------
    tuple of float
        ``(Y_jminus1, Y_j)`` — energy generation rates at cells j-1 and j.
    """
    xp = get_xp(R)
    fact1 = (
        G * M_star * M_dot
        / (2.0 * xp.pi * R[j_val - 1] ** 2
           * (R[j_val - 1] - R[j_val - 2]))
    )
    fact2 = dMj1 / (dMj1 + dMj)
    Y_jminus1 = fact1 * fact2

    fact3 = (
        G * M_star * M_dot
        / (2.0 * xp.pi * R[j_val] ** 2
           * (R[j_val] - R[j_val - 1]))
    )
    fact4 = dMj / (dMj1 + dMj)
    Y_j = fact3 * fact4

    return Y_jminus1, Y_j
