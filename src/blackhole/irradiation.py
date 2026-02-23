"""Irradiation feedback functions.

Extracted from alpha-t dependence.ipynb and GPU_timedep notebooks.
Functions that referenced globals now take explicit parameters.
"""

import numpy as np

from blackhole.constants import ALPHA_COLD, ALPHA_HOT, M_sun, c, sigma_SB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r10(r):
    """Radius in units of 10^10 cm."""
    return r / 1e10


# ---------------------------------------------------------------------------
# Irradiation flux and epsilon
# ---------------------------------------------------------------------------

def Flux_irr(Sigma_inner, nu_inner, r_inner, r, M_star, M_dot, C=5e-3):
    """Irradiation flux at radius *r*.

    Parameters
    ----------
    Sigma_inner, nu_inner, r_inner : float
        Surface density, viscosity, and radius at the inner disk edge.
    r : float or array
        Radius (radii) at which to evaluate the flux.
    M_star : float
        Central object mass (g).
    M_dot : float
        Mass transfer rate (g/s), used as boundary for efficiency.
    C : float
        Irradiation constant (default 5e-3).
    """
    L_edd = 1.4e38 * (M_star / M_sun)
    M_dot_inner = 2.0 * np.pi * r_inner * Sigma_inner * 3.0 * nu_inner / r_inner

    epsilon = np.where(M_dot_inner >= M_dot, 0.1, 0.1 * (M_dot_inner / M_dot))

    L_inner = M_dot_inner * c**2
    L_x = epsilon * np.minimum(L_edd, L_inner)
    return C * L_x / (4.0 * np.pi * r**2)


def Epsilon_irr(Sigma_inner, nu_inner, r_inner, r, M_star, M_dot, C=5e-3):
    """Irradiation parameter epsilon_irr = (T_irr / 10^4)^2.

    Parameters have the same meaning as in :func:`Flux_irr`.
    """
    L_edd = 1.4e38 * (M_star / M_sun)
    M_dot_inner = 2.0 * np.pi * r_inner * Sigma_inner * 3.0 * nu_inner / r_inner

    epsilon = np.where(M_dot_inner >= M_dot, 0.1, 0.1 * (M_dot_inner / M_dot))

    L_inner = M_dot_inner * c**2
    L_x = epsilon * np.minimum(L_edd, L_inner)
    T_irr = (C * L_x / (4.0 * np.pi * sigma_SB * r**2)) ** 0.25
    return (T_irr / 1e4) ** 2


# ---------------------------------------------------------------------------
# Critical surface densities and temperatures (DIM S-curve)
# ---------------------------------------------------------------------------

def Sigma_max(eps_irr, r, M_star, alpha_cold=ALPHA_COLD):
    """Maximum (upper) critical surface density (g/cm^2)."""
    a1 = 10.8 - 10.3 * eps_irr
    a2 = alpha_cold**(-0.84)
    a3 = (M_star / M_sun) ** (-0.37 + 0.1 * eps_irr)
    a4 = r10(r) ** (1.11 - 0.27 * eps_irr)
    return a1 * a2 * a3 * a4


def Sigma_min(eps_irr, r, M_star, alpha_hot=ALPHA_HOT):
    """Minimum (lower) critical surface density (g/cm^2)."""
    a1 = 8.3 - 7.1 * eps_irr
    a2 = alpha_hot**(-0.77)
    a3 = (M_star / M_sun) ** (-0.37)
    a4 = r10(r) ** (1.12 - 0.23 * eps_irr)
    return a1 * a2 * a3 * a4


def T_c_max(eps_irr, r, alpha_cold=ALPHA_COLD):
    """Maximum critical central temperature (K)."""
    a1 = 10700.0 * alpha_cold**(-0.1)
    a2 = r10(r) ** (-0.05 * eps_irr)
    return a1 * a2


def T_c_min(eps_irr, r, M_star, alpha_hot=ALPHA_HOT):
    """Minimum critical central temperature (K)."""
    a1 = 20900.0 - 11300.0 * eps_irr
    a2 = alpha_hot**(-0.22)
    a3 = (M_star / M_sun) ** (-0.01)
    a4 = r10(r) ** (0.05 - 0.12 * eps_irr)
    result = a1 * a2 * a3 * a4
    return np.maximum(result, 1e-3)


# ---------------------------------------------------------------------------
# Irradiation-modified alpha viscosity
# ---------------------------------------------------------------------------

def alpha_visc_irr(T_c, eps_irr, r, M_star, alpha_cold=ALPHA_COLD, alpha_hot=ALPHA_HOT):
    """Alpha viscosity with irradiation-shifted critical temperature.

    Parameters
    ----------
    T_c : float or array
        Central temperature (K).
    eps_irr : float or array
        Irradiation parameter (from :func:`Epsilon_irr`).
    r : float or array
        Radius (cm).
    M_star : float
        Central object mass (g).
    alpha_cold, alpha_hot : float
        Cold/hot viscosity parameters.
    """
    T_crit = 0.5 * (T_c_max(eps_irr, r, alpha_cold) + T_c_min(eps_irr, r, M_star, alpha_hot))
    log_alpha_0 = np.log(alpha_cold)
    log_alpha_1 = np.log(alpha_hot) - np.log(alpha_cold)
    log_alpha_2 = 1.0 + (T_crit / T_c) ** 8
    log_alpha = log_alpha_0 + log_alpha_1 / log_alpha_2
    return np.exp(log_alpha)
