"""Vectorized luminosity and effective temperature — pure array ops, no JIT.

Drop-in replacements for :mod:`blackhole.luminosity` that work with both
NumPy and CuPy arrays via :func:`~blackhole.gpu.get_xp`.
"""

from blackhole.constants import G, sigma_SB
from blackhole.gpu import get_xp
from blackhole.gpu.disk_physics import kinematic_viscosity


def L_rad(Sigma, H, alpha, r, dr, M_star):
    """Total radiative luminosity of the disk (scalar).

    Parameters
    ----------
    Sigma : array
        Surface density array (g/cm^2).
    H : array
        Scale height array (cm).
    alpha : array or float
        Viscosity parameter.
    r : array
        Radius array (cm).
    dr : float or array
        Radial bin width (cm).
    M_star : float
        Central object mass (g).

    Returns
    -------
    float
        Total disk luminosity (erg/s).
    """
    xp = get_xp(Sigma, H, r)
    nu = kinematic_viscosity(H, r, alpha, M_star)
    M_dot_array = 2.0 * xp.pi * r * Sigma * 3.0 * nu / r
    L_per_annulus = (3.0 / 2.0) * G * M_star * M_dot_array * dr / r ** 2
    # Guard against overflow (e.g. numerically divergent Sigma at fine grids)
    L_per_annulus = xp.where(xp.isfinite(L_per_annulus), L_per_annulus, 0.0)
    return float(xp.sum(L_per_annulus))


def L_rad_array(Sigma, H, alpha, r, dr, M_star):
    """Radiative luminosity as a function of radius.

    Parameters
    ----------
    Sigma : array
        Surface density array (g/cm^2).
    H : array
        Scale height array (cm).
    alpha : array or float
        Viscosity parameter.
    r : array
        Radius array (cm).
    dr : float or array
        Radial bin width (cm).
    M_star : float
        Central object mass (g).

    Returns
    -------
    array
        Luminosity per annulus (erg/s).
    """
    xp = get_xp(Sigma, H, r)
    nu = kinematic_viscosity(H, r, alpha, M_star)
    M_dot_array = 2.0 * xp.pi * r * Sigma * 3.0 * nu / r
    L_per_annulus = (3.0 / 2.0) * G * M_star * M_dot_array * dr / r ** 2
    # Guard against overflow (e.g. numerically divergent Sigma at fine grids)
    return xp.where(xp.isfinite(L_per_annulus), L_per_annulus, 0.0)


def T_eff(Sigma, H, alpha, r, M_star):
    """Effective temperature profile of the disk.

    Parameters
    ----------
    Sigma : array
        Surface density array (g/cm^2).
    H : array
        Scale height array (cm).
    alpha : array or float
        Viscosity parameter.
    r : array
        Radius array (cm).
    M_star : float
        Central object mass (g).

    Returns
    -------
    array
        Effective temperature at each radius (K).
    """
    xp = get_xp(Sigma, H, r)
    nu = kinematic_viscosity(H, r, alpha, M_star)
    M_dot_local = 2.0 * xp.pi * r * Sigma * 3.0 * nu / r
    r_in = r[0]
    T4 = (
        (3.0 * G * M_star * M_dot_local)
        / (8.0 * xp.pi * sigma_SB * r ** 3)
        * (1.0 - xp.sqrt(r_in / r))
    )
    # Clamp overflow and negatives before taking 4th root
    T4 = xp.where(xp.isfinite(T4), T4, 0.0)
    T4 = xp.maximum(T4, 0.0)
    return T4 ** 0.25
