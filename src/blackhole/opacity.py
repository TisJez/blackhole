"""Opacity regimes for accretion disk modelling.

Extracted from opacity_formulae.ipynb and GPU_timedep notebooks.
All functions accept scalars or numpy arrays.
"""

import numpy as np

from blackhole import cpu_jit
from blackhole.constants import X_HYDROGEN, Z_METALS, Z_STAR

# ---------------------------------------------------------------------------
# Individual opacity components
# ---------------------------------------------------------------------------

@cpu_jit
def kappa_e(rho, T, X=X_HYDROGEN):
    """Electron scattering opacity with relativistic + degeneracy corrections."""
    a = 0.2 * (1 + X) * (1.0 / (1.0 + 2.7e11 * (rho / T ** 2)))
    b = 1.0 / (1.0 + (T / 4.5e8) ** 0.86)
    return a * b


@cpu_jit
def kappa_K(rho, T, X=X_HYDROGEN, Z=Z_METALS):
    """Kramers (bound-free + free-free) opacity."""
    a = 4e25 * (1 + X) * (Z + 0.001)
    b = rho / T ** 3.5
    return a * b


@cpu_jit
def kappa_Hminus(rho, T, Z=Z_METALS):
    """H-minus opacity."""
    a = 1.1e-25 * np.sqrt(Z * rho)
    b = T ** 7.7
    return a * b


@cpu_jit
def kappa_mol(Z=Z_METALS):
    """Molecular opacity (constant floor)."""
    return 0.1 * Z


@cpu_jit
def kappa_rad(rho, T, X=X_HYDROGEN, Z=Z_METALS):
    """Combined radiative opacity (electron + Kramers + H- + molecular)."""
    ke = kappa_e(rho, T, X)
    kk = kappa_K(rho, T, X, Z)
    khm = kappa_Hminus(rho, T, Z)
    km = kappa_mol(Z)
    high = ke + kk
    # Harmonic mean of khm and high. Numba raises ZeroDivisionError for
    # 1.0/0.0 (unlike CPython which returns inf), so clamp both inputs
    # AND the reciprocal sum to prevent crashes from zero or overflow→inf.
    khm = np.maximum(khm, 1e-300)
    high = np.maximum(high, 1e-300)
    inv_sum = 1.0 / khm + 1.0 / high
    inv_sum = np.maximum(inv_sum, 1e-300)
    blend = 1.0 / inv_sum
    return km + blend


@cpu_jit
def kappa_cond(rho, T, Z_s=Z_STAR):
    """Conductive opacity."""
    a = 2.6e-7 * Z_s
    rho_sq = np.maximum(rho ** 2, 1e-300)
    b = T ** 2 / rho_sq
    c = 1.0 + (rho / 2e6) ** (2.0 / 3.0)
    return a * b * c


@cpu_jit
def kappa_tot(rho, T, X=X_HYDROGEN, Z=Z_METALS, Z_s=Z_STAR):
    """Total opacity (radiative || conductive)."""
    kr = kappa_rad(rho, T, X, Z)
    kc = kappa_cond(rho, T, Z_s)
    kr = np.maximum(kr, 1e-300)
    kc = np.maximum(kc, 1e-300)
    inv_sum = 1.0 / kr + 1.0 / kc
    inv_sum = np.maximum(inv_sum, 1e-300)
    return 1.0 / inv_sum


# ---------------------------------------------------------------------------
# Simplified / legacy opacities (from diskequations_SS_bath_params.ipynb)
# ---------------------------------------------------------------------------

@cpu_jit
def kappa_ff(rho, T, kappa_0=6.4e22):
    """Free-free (Kramers) opacity only."""
    return kappa_0 * rho * T ** (-3.5)


@cpu_jit
def kappa_simple(rho, T, kappa_es=0.4, kappa_0=6.4e22):
    """Simple electron-scattering + Kramers opacity."""
    return kappa_es + kappa_0 * rho * T ** (-3.5)


@cpu_jit
def kappa_bf(rho, T, kappa_es=0.4, kappa_bf0=5e24):
    """Bound-free + electron-scattering opacity."""
    return kappa_es + kappa_bf0 * rho * T ** (-3.5)


# ---------------------------------------------------------------------------
# Full Bath opacity (GPU_timedep version that takes H, Sigma)
# ---------------------------------------------------------------------------

@cpu_jit
def kappa_bath(H, Sigma, T, X=X_HYDROGEN, Z=Z_METALS, Z_s=Z_STAR):
    """Total opacity from H and Sigma (density derived internally).

    This mirrors the ``kappa_bath`` function in the GPU_timedep notebook.
    """
    rho = Sigma / (2.0 * H)
    return kappa_tot(rho, T, X, Z, Z_s)


# ---------------------------------------------------------------------------
# Numerical derivatives (not JIT'd — call JIT'd functions)
# ---------------------------------------------------------------------------

def kappa_tot_drho(rho, T, drho=1e-5):
    """Numerical derivative of kappa_tot w.r.t. rho."""
    return (kappa_tot(rho + drho, T) - kappa_tot(rho, T)) / drho


def kappa_tot_dT(rho, T, dT=1e-5):
    """Numerical derivative of kappa_tot w.r.t. T."""
    return (kappa_tot(rho, T + dT) - kappa_tot(rho, T)) / dT


def kappa_cond_drho(rho, T, drho=1e-5):
    """Numerical derivative of kappa_cond w.r.t. rho."""
    return (kappa_cond(rho + drho, T) - kappa_cond(rho, T)) / drho
