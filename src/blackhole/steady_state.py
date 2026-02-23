"""Shakura-Sunyaev steady-state disk structure (3 regions).

Extracted from diskequations_SS_bath_params.ipynb.
All functions take explicit physical parameters rather than reading globals.
"""

import numpy as np

from blackhole import cpu_jit
from blackhole.constants import M_DOT_CRIT, G, M_sun, c

# ---------------------------------------------------------------------------
# Dimensionless helpers
# ---------------------------------------------------------------------------

@cpu_jit
def r_g(M):
    """Schwarzschild radius (cm)."""
    return 2.0 * G * M / (c**2)


@cpu_jit
def r_hat(r, M):
    """Radius in units of Schwarzschild radius."""
    return r / r_g(M)


@cpu_jit
def m_rel(M):
    """Mass in solar units."""
    return M / M_sun


@cpu_jit
def m_dot_rel(M_dot):
    """Accretion rate in units of M_dot_crit."""
    return M_dot / M_DOT_CRIT


@cpu_jit
def f_boundary(r, M):
    """Inner boundary correction factor f(r)."""
    return 1.0 - np.sqrt(3.0 * r_g(M) / r)


# ---------------------------------------------------------------------------
# Region boundaries (in r_hat units)
# ---------------------------------------------------------------------------

@cpu_jit
def border_inner_middle(alpha, M, M_dot):
    """r_hat boundary between inner and middle regions."""
    return 150.0 * (alpha * m_rel(M)) ** (2.0 / 21.0) * m_dot_rel(M_dot) ** (16.0 / 21.0)


@cpu_jit
def border_middle_outer(M_dot):
    """r_hat boundary between middle and outer regions."""
    return 6.3e3 * m_dot_rel(M_dot) ** (2.0 / 3.0)


# ---------------------------------------------------------------------------
# INNER REGION (radiation pressure dominated, electron scattering opacity)
# ---------------------------------------------------------------------------

@cpu_jit
def H_inner(r, M, M_dot, alpha):
    """Disk scale height — inner region (cm)."""
    return 5.5e4 * m_rel(M) * m_dot_rel(M_dot) * f_boundary(r, M)


@cpu_jit
def Sigma_inner(r, M, M_dot, alpha):
    """Surface density — inner region (g/cm^2)."""
    return 1e2 * alpha**(-1) * m_dot_rel(M_dot)**(-1) * r_hat(r, M)**(3.0 / 2.0) * f_boundary(r, M)**(-1)


@cpu_jit
def rho_inner(r, M, M_dot, alpha):
    """Midplane density — inner region (g/cm^3)."""
    return (
        9e-4 * alpha**(-1) * m_rel(M)**(-1) * m_dot_rel(M_dot)**(-2) * r_hat(r, M)**(3.0 / 2.0) * f_boundary(r, M)**(-2)
    )


@cpu_jit
def T_c_inner(r, M, M_dot, alpha):
    """Central temperature — inner region (K)."""
    return 4.9e7 * alpha**(-1.0 / 4.0) * m_rel(M)**(-1.0 / 4.0) * r_hat(r, M)**(-3.0 / 8.0)


@cpu_jit
def tau_inner(r, M, M_dot, alpha):
    """Effective optical depth — inner region."""
    return (
        8.4e-3
        * alpha**(-17.0 / 16.0)
        * m_rel(M)**(-1.0 / 16.0)
        * m_dot_rel(M_dot)**(-2)
        * r_hat(r, M)**(93.0 / 32.0)
        * f_boundary(r, M)**(-2)
    )


@cpu_jit
def u_r_inner(r, M, M_dot, alpha):
    """Radial drift velocity — inner region (cm/s)."""
    return 7.6e8 * alpha * m_dot_rel(M_dot)**2 * r_hat(r, M)**(-5.0 / 2.0) * f_boundary(r, M)


# ---------------------------------------------------------------------------
# MIDDLE REGION (gas pressure dominated, electron scattering opacity)
# ---------------------------------------------------------------------------

@cpu_jit
def H_middle(r, M, M_dot, alpha):
    """Disk scale height — middle region (cm)."""
    return (
        2.7e3
        * alpha**(-1.0 / 10.0)
        * m_rel(M)**(9.0 / 10.0)
        * m_dot_rel(M_dot)**(1.0 / 5.0)
        * r_hat(r, M)**(21.0 / 20.0)
        * f_boundary(r, M)**(1.0 / 5.0)
    )


@cpu_jit
def Sigma_middle(r, M, M_dot, alpha):
    """Surface density — middle region (g/cm^2)."""
    return (
        4.3e4
        * alpha**(-4.0 / 5.0)
        * m_rel(M)**(1.0 / 5.0)
        * m_dot_rel(M_dot)**(3.0 / 5.0)
        * r_hat(r, M)**(-3.0 / 5.0)
        * f_boundary(r, M)**(3.0 / 5.0)
    )


@cpu_jit
def rho_middle(r, M, M_dot, alpha):
    """Midplane density — middle region (g/cm^3)."""
    return (
        8.0
        * alpha**(-7.0 / 10.0)
        * m_rel(M)**(-7.0 / 10.0)
        * m_dot_rel(M_dot)**(2.0 / 5.0)
        * r_hat(r, M)**(-33.0 / 20.0)
        * f_boundary(r, M)**(2.0 / 5.0)
    )


@cpu_jit
def T_c_middle(r, M, M_dot, alpha):
    """Central temperature — middle region (K)."""
    return (
        2.2e8
        * alpha**(-1.0 / 5.0)
        * m_rel(M)**(-1.0 / 5.0)
        * m_dot_rel(M_dot)**(2.0 / 5.0)
        * r_hat(r, M)**(-9.0 / 10.0)
        * f_boundary(r, M)**(2.0 / 5.0)
    )


@cpu_jit
def tau_middle(r, M, M_dot, alpha):
    """Effective optical depth — middle region."""
    return (
        2.4e1
        * alpha**(-4.0 / 5.0)
        * m_rel(M)**(1.0 / 5.0)
        * m_dot_rel(M_dot)**(1.0 / 10.0)
        * r_hat(r, M)**(3.0 / 20.0)
        * f_boundary(r, M)**(1.0 / 10.0)
    )


@cpu_jit
def u_r_middle(r, M, M_dot, alpha):
    """Radial drift velocity — middle region (cm/s)."""
    return (
        1.7e6
        * alpha**(4.0 / 5.0)
        * m_rel(M)**(-1.0 / 5.0)
        * m_dot_rel(M_dot)**(2.0 / 5.0)
        * r_hat(r, M)**(-2.0 / 5.0)
        * f_boundary(r, M)**(-3.0 / 5.0)
    )


# ---------------------------------------------------------------------------
# OUTER REGION (gas pressure dominated, Kramers opacity)
# ---------------------------------------------------------------------------

@cpu_jit
def H_outer(r, M, M_dot, alpha):
    """Disk scale height — outer region (cm)."""
    return (
        1.5e3
        * alpha**(-1.0 / 10.0)
        * m_rel(M)**(9.0 / 10.0)
        * m_dot_rel(M_dot)**(3.0 / 20.0)
        * r_hat(r, M)**(9.0 / 8.0)
        * f_boundary(r, M)**(3.0 / 20.0)
    )


@cpu_jit
def Sigma_outer(r, M, M_dot, alpha):
    """Surface density — outer region (g/cm^2)."""
    return (
        1.4e5
        * alpha**(-4.0 / 5.0)
        * m_rel(M)**(1.0 / 5.0)
        * m_dot_rel(M_dot)**(7.0 / 10.0)
        * r_hat(r, M)**(-3.0 / 4.0)
        * f_boundary(r, M)**(7.0 / 10.0)
    )


@cpu_jit
def rho_outer(r, M, M_dot, alpha):
    """Midplane density — outer region (g/cm^3)."""
    return (
        4.7e1
        * alpha**(-7.0 / 10.0)
        * m_rel(M)**(-7.0 / 10.0)
        * m_dot_rel(M_dot)**(11.0 / 20.0)
        * r_hat(r, M)**(-15.0 / 8.0)
        * f_boundary(r, M)**(11.0 / 20.0)
    )


@cpu_jit
def T_c_outer(r, M, M_dot, alpha):
    """Central temperature — outer region (K)."""
    return (
        6.9e7
        * alpha**(-1.0 / 5.0)
        * m_rel(M)**(-1.0 / 5.0)
        * m_dot_rel(M_dot)**(3.0 / 10.0)
        * r_hat(r, M)**(-3.0 / 4.0)
        * f_boundary(r, M)**(3.0 / 10.0)
    )


@cpu_jit
def tau_outer(r, M, M_dot, alpha):
    """Effective optical depth — outer region."""
    return (
        7.9e1
        * alpha**(-4.0 / 5.0)
        * m_rel(M)**(1.0 / 5.0)
        * m_dot_rel(M_dot)**(1.0 / 5.0)
        * f_boundary(r, M)**(1.0 / 5.0)
    )


@cpu_jit
def u_r_outer(r, M, M_dot, alpha):
    """Radial drift velocity — outer region (cm/s)."""
    return (
        5.4e5
        * alpha**(4.0 / 5.0)
        * m_rel(M)**(-1.0 / 5.0)
        * m_dot_rel(M_dot)**(3.0 / 10.0)
        * r_hat(r, M)**(-1.0 / 4.0)
        * f_boundary(r, M)**(-7.0 / 10.0)
    )
