"""Vectorized disk physics functions — pure array ops, no JIT.

Drop-in replacements for :mod:`blackhole.disk_physics` that work with
both NumPy and CuPy arrays via :func:`~blackhole.gpu.get_xp`.
"""

from blackhole.constants import G, a_rad, mu, r_gas
from blackhole.gpu import get_xp

# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def X_func(r):
    """Transform radius to X coordinate: X = 2*sqrt(r)."""
    xp = get_xp(r)
    return 2.0 * xp.sqrt(r)


def R_func(x):
    """Transform X coordinate back to radius: r = x^2 / 4."""
    return x ** 2 / 4.0


# ---------------------------------------------------------------------------
# Keplerian quantities
# ---------------------------------------------------------------------------

def omega(R, M_star):
    """Keplerian angular velocity (rad/s)."""
    xp = get_xp(R)
    return xp.sqrt(G * M_star / R ** 3)


# ---------------------------------------------------------------------------
# Thermodynamic / structural quantities
# ---------------------------------------------------------------------------

def kinematic_viscosity(H, R, alpha, M_star):
    """Kinematic viscosity nu = (2/3) * alpha * omega * H^2."""
    return (2.0 / 3.0) * alpha * omega(R, M_star) * H ** 2


def density(H, Sigma):
    """Midplane density rho = Sigma / (2*H)."""
    return Sigma / (2.0 * H)


def pressure(H, Sigma, T):
    """Total pressure (gas + radiation)."""
    p_gas = (r_gas * Sigma * T) / (mu * 2.0 * H)
    p_rad = (1.0 / 3.0) * a_rad * T ** 4
    return p_gas + p_rad


def pressure_2(H, Sigma, R, M_star):
    """Pressure from vertical hydrostatic equilibrium."""
    return 0.5 * Sigma * H * (G * M_star / R ** 3)


def scale_height(Sigma, p, R, M_star):
    """Scale height from pressure balance: H = sqrt(2*p*R^3 / (Sigma*G*M))."""
    a1 = 2.0 * p / Sigma
    a2 = R ** 3 / (G * M_star)
    return a1 * a2


# ---------------------------------------------------------------------------
# Surface-density helpers
# ---------------------------------------------------------------------------

def S_factor(X, Sigma):
    """S = X * Sigma (conserved variable for diffusion equation)."""
    return X * Sigma


def Sigma_from_S(S, X):
    """Recover Sigma from S."""
    return S / X


def Marr(X, Sigma, dX):
    """Mass array: M_i = pi * S_i * X_i^2 * dX / 4."""
    xp = get_xp(X, Sigma)
    S_f = X * Sigma
    return xp.pi * S_f * X ** 2 * dX / 4.0
