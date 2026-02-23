"""Energy balance and hydrostatic equilibrium Newton solvers.

Extracted from the GPU_timedep notebook. Functions that referenced globals
(M_star, opacity, etc.) now take them as explicit parameters.

Performance: The solver loops and root functions are fully JIT-compiled via
numba, using an inline secant method matching scipy.optimize.newton's algorithm.
"""

import numpy as np
from scipy.optimize import newton

from blackhole import cpu_jit
from blackhole.constants import G, a_rad, c, mu, r_gas
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
# JIT'd root functions (inlined, no optional arguments)
# ---------------------------------------------------------------------------

@cpu_jit
def _energy_balance_jit(T, H, Sigma, R, alpha, M_star, F_irr):
    """JIT'd energy balance with kappa_bath opacity (default case)."""
    # Clamp to safe range: prevents underflow in T**7.7 (opacity) and
    # division by zero in T**3.5 denominators. H,Sigma must be positive.
    if H <= 0.0 or Sigma <= 0.0 or T < 1.0:
        return 1e30
    nu = kinematic_viscosity(H, R, alpha, M_star)
    om = omega(R, M_star)
    f_visc = (9.0 / 8.0) * nu * Sigma * om ** 2
    kappa = kappa_bath(H, Sigma, T)
    denom = 3.0 * Sigma * kappa
    if denom <= 0.0:
        return 1e30
    f_rad = (2.0 * a_rad * c * T ** 4) / denom
    return 2.0 * f_visc + F_irr - f_rad


@cpu_jit
def _pressure_balance_jit(H, Sigma, R, T, M_star):
    """JIT'd pressure balance."""
    if H < 1.0 or Sigma <= 0.0 or T <= 0.0:
        return 1e30
    p_gas = (r_gas * Sigma * T) / (mu * 2.0 * H)
    p_rad = (1.0 / 3.0) * a_rad * T ** 4
    p1 = p_gas + p_rad
    p2 = 0.5 * Sigma * H * (G * M_star / R ** 3)
    return p1 - p2


# ---------------------------------------------------------------------------
# JIT'd secant method solver loops
#
# These implement the secant method identically to scipy.optimize.newton
# (when fprime=None): second point at x0*(1+1e-4)+1e-4, convergence check
# abs(p - p1) <= tol, maxiter=50, tol=1.48e-8.
# ---------------------------------------------------------------------------

@cpu_jit
def _secant_temperature(x0, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i):
    """Secant method for temperature root with NaN/residual guards."""
    tol = 1.48e-8
    maxiter = 50

    p0 = x0
    p1 = x0 * (1.0 + 1e-4) + 1e-4
    q0 = _energy_balance_jit(p0, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i)
    q1 = _energy_balance_jit(p1, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i)
    q_init = abs(q0)  # save initial residual for verification

    if q0 != q0 or q1 != q1:  # NaN check
        return p1, False

    for _ in range(maxiter):
        if q1 == q0:
            break
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if p != p or p < 1.0 or p > 1e20:  # NaN, below floor, or overflow
            return p1, False
        if abs(p - p1) <= tol:
            p1 = p
            break
        p0 = p1
        q0 = q1
        p1 = p
        q1 = _energy_balance_jit(p1, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i)
        if q1 != q1:  # NaN check
            return p1, False

    # Verify the residual is actually small (catch false convergence in flat regions)
    q_final = _energy_balance_jit(p1, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i)
    if q_final != q_final:  # NaN
        return p1, False
    if abs(q_final) > 1e-4 * max(q_init, 1.0):
        return p1, False

    return p1, True


@cpu_jit
def _secant_scale_height(x0, sig_i, r_i, t_i, M_star):
    """Secant method for scale height root with NaN/residual guards."""
    tol = 1.48e-8
    maxiter = 50

    p0 = x0
    p1 = x0 * (1.0 + 1e-4) + 1e-4
    q0 = _pressure_balance_jit(p0, sig_i, r_i, t_i, M_star)
    q1 = _pressure_balance_jit(p1, sig_i, r_i, t_i, M_star)
    q_init = abs(q0)  # save initial residual for verification

    if q0 != q0 or q1 != q1:  # NaN check
        return p1, False

    for _ in range(maxiter):
        if q1 == q0:
            break
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if p != p or p < 1.0 or p > 1e20:  # NaN, below floor, or overflow
            return p1, False
        if abs(p - p1) <= tol:
            p1 = p
            break
        p0 = p1
        q0 = q1
        p1 = p
        q1 = _pressure_balance_jit(p1, sig_i, r_i, t_i, M_star)
        if q1 != q1:  # NaN check
            return p1, False

    # Verify the residual is actually small (catch false convergence in flat regions)
    q_final = _pressure_balance_jit(p1, sig_i, r_i, t_i, M_star)
    if q_final != q_final:  # NaN
        return p1, False
    if abs(q_final) > 1e-4 * max(q_init, 1.0):
        return p1, False

    return p1, True


@cpu_jit
def _solve_temperature_jit(H, Sigma, r, T_c, alpha_arr, M_star, F_irr, T_new):
    """JIT'd temperature solve loop with inline secant method."""
    T_min = 1e0
    T_max = 2e10

    for i in range(len(Sigma)):
        if Sigma[i] <= 1e-100:
            continue

        h_i = H[i]
        sig_i = Sigma[i]
        r_i = r[i]
        alpha_i = alpha_arr[i]
        f_irr_i = F_irr[i]

        # Primary guess: 1.5x current value
        val, ok = _secant_temperature(T_c[i] * 1.5, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i)
        if ok and T_min < val < T_max:
            T_new[i] = val
            continue

        # Fallback guesses
        solved = False
        for g in range(7):
            if g == 0:
                x0 = 1e0
            elif g == 1:
                x0 = 1e1
            elif g == 2:
                x0 = 1e2
            elif g == 3:
                x0 = 1e3
            elif g == 4:
                x0 = 1e4
            elif g == 5:
                x0 = 1e5
            else:
                x0 = 1e6

            val, ok = _secant_temperature(x0, h_i, sig_i, r_i, alpha_i, M_star, f_irr_i)
            if ok and T_min < val < T_max:
                T_new[i] = val
                solved = True
                break

            if solved:
                break

    return T_new


@cpu_jit
def _solve_scale_height_jit(H, Sigma, r, T_c, M_star, H_new):
    """JIT'd scale-height solve loop with inline secant method."""
    H_min = 1e7
    H_max = 2e10

    for i in range(len(Sigma)):
        if Sigma[i] <= 1e-100:
            continue

        sig_i = Sigma[i]
        r_i = r[i]
        t_i = T_c[i]

        # Primary guess: 1.5x current value
        val, ok = _secant_scale_height(H[i] * 1.5, sig_i, r_i, t_i, M_star)
        if ok and H_min < val < H_max:
            H_new[i] = val
            continue

        # Fallback guesses
        solved = False
        for g in range(8):
            if g == 0:
                x0 = 1e4
            elif g == 1:
                x0 = 1e5
            elif g == 2:
                x0 = 1e6
            elif g == 3:
                x0 = 1e7
            elif g == 4:
                x0 = 1e8
            elif g == 5:
                x0 = 1e9
            elif g == 6:
                x0 = 1e10
            else:
                x0 = 1e11

            val, ok = _secant_scale_height(x0, sig_i, r_i, t_i, M_star)
            if ok and H_min < val < H_max:
                H_new[i] = val
                solved = True
                break

            if solved:
                break

    return H_new


# ---------------------------------------------------------------------------
# Newton solvers (public API)
# ---------------------------------------------------------------------------

def solve_temperature(H, Sigma, r, T_c, alpha, M_star,
                      F_irr=None, opacity_func=None):
    """Solve for midplane temperature at each grid point.

    Uses the secant method with fallback initial guesses. When using the
    default opacity (kappa_bath), the entire solve loop is JIT-compiled.

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
        Opacity function ``f(H, Sigma, T)``. Defaults to :func:`kappa_bath`.

    Returns
    -------
    np.ndarray
        Updated temperature array (does not mutate input).
    """
    T_new = T_c.copy()
    alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=np.float64), Sigma.shape).copy()
    F_irr_arr = np.zeros_like(Sigma) if F_irr is None else np.asarray(F_irr, dtype=np.float64)

    # Fast path: JIT'd solver for default opacity
    if opacity_func is None or opacity_func is kappa_bath:
        return _solve_temperature_jit(H, Sigma, r, T_c, alpha_arr, M_star, F_irr_arr, T_new)

    # Slow path: scipy.optimize.newton for custom opacity
    return _solve_temperature_scipy(H, Sigma, r, T_c, alpha_arr, M_star, F_irr_arr, T_new, opacity_func)


def _solve_temperature_scipy(H, Sigma, r, T_c, alpha_arr, M_star, F_irr_arr, T_new, opacity_func):
    """Fallback scipy-based temperature solver for custom opacity functions."""
    T_min_limit = 1e0
    T_max_limit = 2e10
    initial_guesses = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

    for i in range(len(Sigma)):
        if float(Sigma[i]) <= 1e-100:
            continue

        args = (
            float(H[i]), float(Sigma[i]), float(r[i]),
            float(alpha_arr[i]), M_star, float(F_irr_arr[i]), opacity_func,
        )

        solved = False
        try:
            new_T = newton(energy_balance, float(T_c[i]) * 1.5, args=args)
            if T_min_limit < new_T < T_max_limit:
                T_new[i] = new_T
                solved = True
        except (RuntimeError, ZeroDivisionError, FloatingPointError, ValueError):
            pass

        if not solved:
            for guess in initial_guesses:
                try:
                    new_T = newton(energy_balance, guess, args=args)
                    if T_min_limit < new_T < T_max_limit:
                        T_new[i] = new_T
                        break
                except (RuntimeError, ZeroDivisionError, FloatingPointError, ValueError):
                    continue

    return T_new


def solve_scale_height(H, Sigma, r, T_c, M_star):
    """Solve for scale height at each grid point.

    Uses the secant method with fallback initial guesses. The entire
    solve loop is JIT-compiled.

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
    return _solve_scale_height_jit(H, Sigma, r, T_c, M_star, H_new)


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
    fact1 = (
        G * M_star * M_dot
        / (2.0 * np.pi * R[j_val - 1] ** 2
           * (R[j_val - 1] - R[j_val - 2]))
    )
    fact2 = dMj1 / (dMj1 + dMj)
    Y_jminus1 = fact1 * fact2

    fact3 = (
        G * M_star * M_dot
        / (2.0 * np.pi * R[j_val] ** 2
           * (R[j_val] - R[j_val - 1]))
    )
    fact4 = dMj / (dMj1 + dMj)
    Y_j = fact3 * fact4

    return Y_jminus1, Y_j
