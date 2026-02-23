"""Radiative luminosity and effective temperature diagnostics.

Extracted from Outburst_graphs.ipynb. Functions now take single-timestep
arrays rather than history arrays with a time index.
"""

import numpy as np

from blackhole.constants import G, sigma_SB
from blackhole.disk_physics import kinematic_viscosity


def L_rad(Sigma, H, alpha, r, dr, M_star):
    """Total radiative luminosity of the disk (scalar).

    Parameters
    ----------
    Sigma : np.ndarray
        Surface density array (g/cm^2) at a single timestep.
    H : np.ndarray
        Scale height array (cm).
    alpha : np.ndarray or float
        Viscosity parameter (scalar or per-cell).
    r : np.ndarray
        Radius array (cm).
    dr : float or np.ndarray
        Radial bin width (cm).
    M_star : float
        Central object mass (g).

    Returns
    -------
    float
        Total disk luminosity (erg/s).
    """
    nu = kinematic_viscosity(H, r, alpha, M_star)
    M_dot_array = 2.0 * np.pi * r * Sigma * 3.0 * nu / r
    L_per_annulus = (3.0 / 2.0) * G * M_star * M_dot_array * dr / r ** 2
    return float(np.sum(L_per_annulus))


def L_rad_array(Sigma, H, alpha, r, dr, M_star):
    """Radiative luminosity as a function of radius.

    Parameters
    ----------
    Sigma : np.ndarray
        Surface density array (g/cm^2) at a single timestep.
    H : np.ndarray
        Scale height array (cm).
    alpha : np.ndarray or float
        Viscosity parameter (scalar or per-cell).
    r : np.ndarray
        Radius array (cm).
    dr : float or np.ndarray
        Radial bin width (cm).
    M_star : float
        Central object mass (g).

    Returns
    -------
    np.ndarray
        Luminosity per annulus (erg/s).
    """
    nu = kinematic_viscosity(H, r, alpha, M_star)
    M_dot_array = 2.0 * np.pi * r * Sigma * 3.0 * nu / r
    return (3.0 / 2.0) * G * M_star * M_dot_array * dr / r ** 2


def T_eff(Sigma, H, alpha, r, M_star):
    """Effective temperature profile of the disk.

    Parameters
    ----------
    Sigma : np.ndarray
        Surface density array (g/cm^2).
    H : np.ndarray
        Scale height array (cm).
    alpha : np.ndarray or float
        Viscosity parameter (scalar or per-cell).
    r : np.ndarray
        Radius array (cm).
    M_star : float
        Central object mass (g).

    Returns
    -------
    np.ndarray
        Effective temperature at each radius (K).
    """
    nu = kinematic_viscosity(H, r, alpha, M_star)
    M_dot_local = 2.0 * np.pi * r * Sigma * 3.0 * nu / r
    r_in = r[0]
    T4 = (
        (3.0 * G * M_star * M_dot_local)
        / (8.0 * np.pi * sigma_SB * r ** 3)
        * (1.0 - np.sqrt(r_in / r))
    )
    # Clamp negatives (can arise at r = r_in) before taking 4th root
    T4 = np.maximum(T4, 0.0)
    return np.float_power(T4, 0.25)
