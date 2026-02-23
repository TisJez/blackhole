"""Critical-regime (CR) steady-state disk structure solvers.

Solves the coupled energy balance and viscous stress equations to find
(r, rho) as a function of temperature T for a given alpha viscosity.
Supports both fitted (Bath/total) opacity and Kramers-only opacity.

Extracted from GPU_cre_equations_bath_params.ipynb and GPU_cr_graph_normal_bh.ipynb.
"""

import numpy as np
from scipy import optimize

from blackhole.constants import G, M_sun, a_rad, c, k_B, m_p, mu
from blackhole.opacity import kappa_simple, kappa_tot, kappa_tot_drho

# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

def p_gas(rho, T):
    """Gas pressure."""
    return (k_B / (mu * m_p)) * rho * T


def p_rad(T):
    """Radiation pressure."""
    return (a_rad / 3.0) * T ** 4


def p_tot(rho, T):
    """Total pressure (gas + radiation)."""
    return p_gas(rho, T) + p_rad(T)


def omega_kepler(r, M_star):
    """Keplerian angular velocity."""
    return np.sqrt(G * M_star / r ** 3)


# ---------------------------------------------------------------------------
# Scale height
# ---------------------------------------------------------------------------

def h_2(r, rho, M_star, M_dot, T):
    """Scale height from vertical hydrostatic equilibrium (self-gravity included).

    Parameters
    ----------
    r : float
        Radius (cm).
    rho : float
        Midplane density (g/cm^3).
    M_star : float
        Central object mass (g).
    M_dot : float
        Mass accretion rate (g/s). Not used directly but kept for API compat.
    T : float
        Midplane temperature (K).
    """
    a1 = p_tot(rho, T)
    a2 = rho * G * M_star / r ** 3
    a3 = 4.0 * np.pi * G * rho ** 2
    return np.sqrt(a1 / (a2 + a3))


# ---------------------------------------------------------------------------
# f_1: Energy balance equation
# ---------------------------------------------------------------------------

def _f_1_1(T):
    """Radiative term: (a_rad * T^4)^2."""
    return a_rad ** 2 * T ** 8


def _f_1_2(M_star, M_dot, r, rho, T):
    """Viscous dissipation term with fitted (total) opacity."""
    n1 = 9.0 * G * M_star ** 2 * M_dot ** 2
    n2 = rho * p_tot(rho, T) * kappa_tot(rho, T) ** 2
    d1 = 16.0 * np.pi ** 2 * c ** 2 * r ** 3
    d2 = M_star + 4.0 * np.pi * rho * r ** 3
    return (n1 / d1) * (n2 / d2)


def _f_1_2_old(M_star, M_dot, r, rho, T):
    """Viscous dissipation term with Kramers (simple) opacity."""
    n1 = 9.0 * G * M_star ** 2 * M_dot ** 2
    n2 = rho * p_tot(rho, T) * kappa_simple(rho, T) ** 2
    d1 = 16.0 * np.pi ** 2 * c ** 2 * r ** 3
    d2 = M_star + 4.0 * np.pi * rho * r ** 3
    return (n1 / d1) * (n2 / d2)


def f_1(M_star, M_dot, r, rho, T):
    """Energy balance root function (fitted opacity)."""
    return _f_1_1(T) - _f_1_2(M_star, M_dot, r, rho, T)


def f_1_old(M_star, M_dot, r, rho, T):
    """Energy balance root function (Kramers opacity)."""
    return _f_1_1(T) - _f_1_2_old(M_star, M_dot, r, rho, T)


# ---------------------------------------------------------------------------
# f_2: Viscous stress balance equation
# ---------------------------------------------------------------------------

def _f_2_1(M_star, M_dot, r, rho, alpha_var):
    """Viscous stress term."""
    n = G ** 2 * M_dot ** 2 * M_star * rho
    d = 16.0 * np.pi ** 2 * alpha_var ** 2 * r ** 6
    return n / d


def _f_2_2(M_star, r, rho, T):
    """Pressure term."""
    n = p_tot(rho, T) ** 3
    d = M_star + 4.0 * np.pi * rho * r ** 3
    return n / d


def f_2(M_star, M_dot, r, rho, T, alpha_var):
    """Viscous stress balance root function."""
    return _f_2_1(M_star, M_dot, r, rho, alpha_var) - _f_2_2(M_star, r, rho, T)


# ---------------------------------------------------------------------------
# Jacobian components
# ---------------------------------------------------------------------------

def _kappa_simple_drho(rho, T, drho=1e-5):
    """Numerical derivative of kappa_simple w.r.t. rho."""
    return (kappa_simple(rho + drho, T) - kappa_simple(rho, T)) / drho


def _df_1_dr(M_star, M_dot, r, rho, T):
    """d(f_1)/dr with fitted opacity."""
    n1 = 9.0 * G * M_star ** 2 * M_dot ** 2
    n2 = rho * p_tot(rho, T) * kappa_tot(rho, T) ** 2
    d1 = 16.0 * np.pi ** 2 * c ** 2 * r ** 3
    d2 = M_star + 4.0 * np.pi * rho * r ** 3
    f3 = 3.0 / r
    f4 = 12.0 * np.pi * rho * r ** 2 / d2
    return (n1 / d1) * (n2 / d2) * (f3 + f4)


def _df_1_dr_old(M_star, M_dot, r, rho, T):
    """d(f_1)/dr with Kramers opacity."""
    n1 = 9.0 * G * M_star ** 2 * M_dot ** 2
    n2 = rho * p_tot(rho, T) * kappa_simple(rho, T) ** 2
    d1 = 16.0 * np.pi ** 2 * c ** 2 * r ** 3
    d2 = M_star + 4.0 * np.pi * rho * r ** 3
    f3 = 3.0 / r
    f4 = 12.0 * np.pi * rho * r ** 2 / d2
    return (n1 / d1) * (n2 / d2) * (f3 + f4)


def _df_1_drho(M_star, M_dot, r, rho, T):
    """d(f_1)/drho with fitted opacity."""
    n1 = 9.0 * G * M_star ** 2 * M_dot ** 2
    n2 = 4.0 * np.pi * r ** 3 * rho * kappa_tot(rho, T) ** 2 * p_tot(rho, T)
    r3 = r ** 3
    d1 = 16.0 * np.pi ** 2 * c ** 2 * r3 * (M_star + 4.0 * np.pi * rho * r3)
    d2 = M_star + 4.0 * np.pi * rho * r3
    f2 = (p_gas(rho, T) + p_tot(rho, T)) * kappa_tot(rho, T) ** 2
    f3 = 2.0 * kappa_tot(rho, T) * kappa_tot_drho(rho, T) * rho * p_tot(rho, T)
    f4 = n2 / d2
    return -(n1 / d1) * (f2 + f3 - f4)


def _df_1_drho_old(M_star, M_dot, r, rho, T):
    """d(f_1)/drho with Kramers opacity."""
    n1 = 9.0 * G * M_star ** 2 * M_dot ** 2
    n2 = 4.0 * np.pi * r ** 3 * rho * kappa_simple(rho, T) ** 2 * p_tot(rho, T)
    r3 = r ** 3
    d1 = 16.0 * np.pi ** 2 * c ** 2 * r3 * (M_star + 4.0 * np.pi * rho * r3)
    d2 = M_star + 4.0 * np.pi * rho * r3
    f2 = (p_gas(rho, T) + p_tot(rho, T)) * kappa_simple(rho, T) ** 2
    f3 = 2.0 * kappa_simple(rho, T) * _kappa_simple_drho(rho, T) * rho * p_tot(rho, T)
    f4 = n2 / d2
    return -(n1 / d1) * (f2 + f3 - f4)


def _df_2_dr(M_star, M_dot, r, rho, T, alpha_var):
    """d(f_2)/dr."""
    n1 = 6.0 * G ** 2 * M_dot ** 2 * M_star * rho
    n2 = 12.0 * np.pi * rho * p_tot(rho, T) ** 3 * r ** 2
    d1 = 16.0 * np.pi ** 2 * alpha_var ** 2 * r ** 7
    d2 = M_star + 4.0 * np.pi * rho * r ** 3
    return -(n1 / d1) + n2 / d2 ** 2


def _df_2_drho(M_star, M_dot, r, rho, T, alpha_var):
    """d(f_2)/drho."""
    R_cgs = k_B / m_p
    n1 = G ** 2 * M_dot ** 2 * M_star
    n2 = 3.0 * p_tot(rho, T) ** 2 * (R_cgs / mu) * T
    n3 = 4.0 * np.pi * r ** 3 * p_tot(rho, T) ** 3
    d1 = 16.0 * np.pi ** 2 * alpha_var ** 2 * r ** 6
    d2 = M_star + 4.0 * np.pi * rho * r ** 3
    return (n1 / d1) - (n2 / d2) + n3 / d2 ** 2


# ---------------------------------------------------------------------------
# Combined root functions and Jacobians
# ---------------------------------------------------------------------------

def combined_function_CR(rr_vals, T, alpha_var, M_star=M_sun, M_dot=5e16):
    """Root function for fitted opacity solver: [f_1, f_2]."""
    return [
        f_1(M_star, M_dot, rr_vals[0], rr_vals[1], T),
        f_2(M_star, M_dot, rr_vals[0], rr_vals[1], T, alpha_var),
    ]


def combined_function_old(rr_vals, T, alpha_var, M_star=M_sun, M_dot=5e16):
    """Root function for Kramers opacity solver: [f_1_old, f_2]."""
    return [
        f_1_old(M_star, M_dot, rr_vals[0], rr_vals[1], T),
        f_2(M_star, M_dot, rr_vals[0], rr_vals[1], T, alpha_var),
    ]


def jacobian_CR(rr_vals, T, alpha_var, M_star=M_sun, M_dot=5e16):
    """Jacobian for fitted opacity solver."""
    r, rho = rr_vals
    return np.array([
        [_df_1_dr(M_star, M_dot, r, rho, T), _df_1_drho(M_star, M_dot, r, rho, T)],
        [_df_2_dr(M_star, M_dot, r, rho, T, alpha_var), _df_2_drho(M_star, M_dot, r, rho, T, alpha_var)],
    ])


def jacobian_old(rr_vals, T, alpha_var, M_star=M_sun, M_dot=5e16):
    """Jacobian for Kramers opacity solver."""
    r, rho = rr_vals
    return np.array([
        [_df_1_dr_old(M_star, M_dot, r, rho, T), _df_1_drho_old(M_star, M_dot, r, rho, T)],
        [_df_2_dr(M_star, M_dot, r, rho, T, alpha_var), _df_2_drho(M_star, M_dot, r, rho, T, alpha_var)],
    ])


# ---------------------------------------------------------------------------
# Convenience solvers
# ---------------------------------------------------------------------------

def solve_cr_structure(T_array, r_guess, rho_guess, alpha_var,
                       M_star=M_sun, M_dot=5e16, use_old_opacity=False):
    """Solve for (r, rho) at each temperature in T_array.

    Parameters
    ----------
    T_array : array-like
        Temperature grid (K).
    r_guess, rho_guess : float
        Initial guesses for radius and density.
    alpha_var : float or array-like
        Viscosity parameter (scalar or per-temperature).
    M_star : float
        Central mass (g).
    M_dot : float
        Mass accretion rate (g/s).
    use_old_opacity : bool
        If True, use Kramers opacity; otherwise use fitted (total) opacity.

    Returns
    -------
    r_array, rho_array : np.ndarray
        Solved radius and density arrays.
    """
    if use_old_opacity:
        func = combined_function_old
        jac = jacobian_old
    else:
        func = combined_function_CR
        jac = jacobian_CR

    alpha_arr = np.broadcast_to(alpha_var, np.shape(T_array))

    r_array = np.empty(len(T_array))
    rho_array = np.empty(len(T_array))

    sol = optimize.root(func, [r_guess, rho_guess],
                        args=(T_array[0], alpha_arr[0], M_star, M_dot),
                        jac=jac, method='lm')
    r_array[0] = sol.x[0]
    rho_array[0] = sol.x[1]

    for i in range(len(T_array) - 1):
        sol = optimize.root(func, [r_array[i], rho_array[i]],
                            args=(T_array[i + 1], alpha_arr[i + 1], M_star, M_dot),
                            jac=jac, method='lm')
        r_array[i + 1] = sol.x[0]
        rho_array[i + 1] = sol.x[1]

    return r_array, rho_array
