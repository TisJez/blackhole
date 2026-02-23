"""Temperature-dependent alpha viscosity model.

Extracted from alpha-t dependence.ipynb and GPU_timedep notebooks.
Uses the Hameury prescription for the thermal instability transition.
"""


from blackhole import get_xp
from blackhole.constants import ALPHA_COLD, ALPHA_HOT


def alpha_visc(T_c, alpha_cold=ALPHA_COLD, alpha_hot=ALPHA_HOT, T_crit=2.5e4):
    """Alpha viscosity as a smooth function of central temperature.

    Parameters
    ----------
    T_c : float or array
        Central (midplane) temperature in K.
    alpha_cold : float
        Cold-state viscosity parameter.
    alpha_hot : float
        Hot-state viscosity parameter.
    T_crit : float
        Critical temperature for the transition (K).

    Returns
    -------
    float or array
        Alpha viscosity value(s).
    """
    xp = get_xp(T_c)
    log_alpha_0 = xp.log(alpha_cold)
    log_alpha_1 = xp.log(alpha_hot) - xp.log(alpha_cold)
    log_alpha_2 = 1.0 + xp.float_power(T_crit / T_c, 8)
    log_alpha = log_alpha_0 + log_alpha_1 / log_alpha_2
    return xp.exp(log_alpha)
