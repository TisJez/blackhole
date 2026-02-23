"""Core disk physics functions.

Extracted from the GPU_timedep notebook. Functions that referenced globals
(M_star, etc.) now take them as explicit parameters.
"""

import numpy as np

from blackhole.constants import G, a_rad, mu, r_gas

# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def X_func(r):
    """Transform radius to X coordinate: X = 2*sqrt(r)."""
    return 2.0 * np.sqrt(r)


def R_func(x):
    """Transform X coordinate back to radius: r = x^2 / 4."""
    return np.float_power(x, 2) / 4.0


# ---------------------------------------------------------------------------
# Keplerian quantities
# ---------------------------------------------------------------------------

def omega(R, M_star):
    """Keplerian angular velocity (rad/s)."""
    return np.sqrt(G * M_star / np.float_power(R, 3))


# ---------------------------------------------------------------------------
# Thermodynamic / structural quantities
# ---------------------------------------------------------------------------

def kinematic_viscosity(H, R, alpha, M_star):
    """Kinematic viscosity nu = (2/3) * alpha * omega * H^2."""
    return (2.0 / 3.0) * alpha * omega(R, M_star) * H**2


def density(H, Sigma):
    """Midplane density rho = Sigma / (2*H)."""
    return Sigma / (2.0 * H)


def pressure(H, Sigma, T):
    """Total pressure (gas + radiation)."""
    p_gas = (r_gas * Sigma * T) / (mu * 2.0 * H)
    p_rad = (1.0 / 3.0) * a_rad * np.float_power(T, 4)
    return p_gas + p_rad


def pressure_2(H, Sigma, R, M_star):
    """Pressure from vertical hydrostatic equilibrium."""
    return 0.5 * Sigma * H * (G * M_star / np.float_power(R, 3))


def scale_height(Sigma, p, R, M_star):
    """Scale height from pressure balance: H = sqrt(2*p*R^3 / (Sigma*G*M))."""
    a1 = 2.0 * p / Sigma
    a2 = np.float_power(R, 3) / (G * M_star)
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
    S_f = X * Sigma
    return np.pi * S_f * X**2 * dX / 4.0
