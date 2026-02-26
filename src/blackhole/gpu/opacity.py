"""Vectorized opacity functions — pure array ops, no JIT.

Drop-in replacements for :mod:`blackhole.opacity` that work with both
NumPy and CuPy arrays via :func:`~blackhole.gpu.get_xp`.
"""

from blackhole.constants import X_HYDROGEN, Z_METALS, Z_STAR
from blackhole.gpu import get_xp

# ---------------------------------------------------------------------------
# Individual opacity components
# ---------------------------------------------------------------------------

def kappa_e(rho, T, X=X_HYDROGEN):
    """Electron scattering opacity with relativistic + degeneracy corrections."""
    a = 0.2 * (1 + X) * (1.0 / (1.0 + 2.7e11 * (rho / T ** 2)))
    b = 1.0 / (1.0 + (T / 4.5e8) ** 0.86)
    return a * b


def kappa_K(rho, T, X=X_HYDROGEN, Z=Z_METALS):
    """Kramers (bound-free + free-free) opacity."""
    a = 4e25 * (1 + X) * (Z + 0.001)
    b = rho / T ** 3.5
    return a * b


def kappa_Hminus(rho, T, Z=Z_METALS):
    """H-minus opacity."""
    xp = get_xp(rho, T)
    a = 1.1e-25 * xp.sqrt(Z * rho)
    b = T ** 7.7
    return a * b


def kappa_mol(Z=Z_METALS):
    """Molecular opacity (constant floor)."""
    return 0.1 * Z


def kappa_rad(rho, T, X=X_HYDROGEN, Z=Z_METALS):
    """Combined radiative opacity (electron + Kramers + H- + molecular)."""
    xp = get_xp(rho, T)
    ke = kappa_e(rho, T, X)
    kk = kappa_K(rho, T, X, Z)
    khm = kappa_Hminus(rho, T, Z)
    km = kappa_mol(Z)
    high = ke + kk
    khm = xp.maximum(khm, 1e-300)
    high = xp.maximum(high, 1e-300)
    inv_sum = 1.0 / khm + 1.0 / high
    inv_sum = xp.maximum(inv_sum, 1e-300)
    blend = 1.0 / inv_sum
    return km + blend


def kappa_cond(rho, T, Z_s=Z_STAR):
    """Conductive opacity."""
    xp = get_xp(rho, T)
    a = 2.6e-7 * Z_s
    rho_sq = xp.maximum(rho ** 2, 1e-300)
    b = T ** 2 / rho_sq
    c = 1.0 + (rho / 2e6) ** (2.0 / 3.0)
    return a * b * c


def kappa_tot(rho, T, X=X_HYDROGEN, Z=Z_METALS, Z_s=Z_STAR):
    """Total opacity (radiative || conductive)."""
    xp = get_xp(rho, T)
    kr = kappa_rad(rho, T, X, Z)
    kc = kappa_cond(rho, T, Z_s)
    kr = xp.maximum(kr, 1e-300)
    kc = xp.maximum(kc, 1e-300)
    inv_sum = 1.0 / kr + 1.0 / kc
    inv_sum = xp.maximum(inv_sum, 1e-300)
    return 1.0 / inv_sum


# ---------------------------------------------------------------------------
# Simplified / legacy opacities
# ---------------------------------------------------------------------------

def kappa_ff(rho, T, kappa_0=6.4e22):
    """Free-free (Kramers) opacity only."""
    return kappa_0 * rho * T ** (-3.5)


def kappa_simple(rho, T, kappa_es=0.4, kappa_0=6.4e22):
    """Simple electron-scattering + Kramers opacity."""
    return kappa_es + kappa_0 * rho * T ** (-3.5)


def kappa_bf(rho, T, kappa_es=0.4, kappa_bf0=5e24):
    """Bound-free + electron-scattering opacity."""
    return kappa_es + kappa_bf0 * rho * T ** (-3.5)


# ---------------------------------------------------------------------------
# Full Bath opacity (takes H, Sigma)
# ---------------------------------------------------------------------------

def kappa_bath(H, Sigma, T, X=X_HYDROGEN, Z=Z_METALS, Z_s=Z_STAR):
    """Total opacity from H and Sigma (density derived internally)."""
    rho = Sigma / (2.0 * H)
    return kappa_tot(rho, T, X, Z, Z_s)
