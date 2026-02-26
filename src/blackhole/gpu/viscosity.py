"""Vectorized alpha viscosity model — pure array ops, no JIT.

Drop-in replacement for :mod:`blackhole.viscosity` that works with both
NumPy and CuPy arrays via :func:`~blackhole.gpu.get_xp`.
"""

from blackhole.constants import ALPHA_COLD, ALPHA_HOT
from blackhole.gpu import get_xp


def alpha_visc(T_c, alpha_cold=ALPHA_COLD, alpha_hot=ALPHA_HOT, T_crit=2.5e4):
    """Alpha viscosity as a smooth function of central temperature.

    Parameters
    ----------
    T_c : array
        Central (midplane) temperature in K.
    alpha_cold : float
        Cold-state viscosity parameter.
    alpha_hot : float
        Hot-state viscosity parameter.
    T_crit : float
        Critical temperature for the transition (K).

    Returns
    -------
    array
        Alpha viscosity value(s).
    """
    xp = get_xp(T_c)
    log_alpha_0 = xp.log(alpha_cold)
    log_alpha_1 = xp.log(alpha_hot) - xp.log(alpha_cold)
    log_alpha_2 = 1.0 + (T_crit / T_c) ** 8
    log_alpha = log_alpha_0 + log_alpha_1 / log_alpha_2
    return xp.exp(log_alpha)
