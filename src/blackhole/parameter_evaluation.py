"""Parameter evaluation for DIM simulations.

Pre-flight checks for mass deposition and instability constraints.
Lets users verify parameters before running multi-hour simulations.
"""

import dataclasses

import numpy as np

from blackhole.constants import ALPHA_COLD, ALPHA_HOT, G, k_B, m_p, mu
from blackhole.disk_physics import R_func, X_func
from blackhole.evolution import calculate_timestep
from blackhole.irradiation import Sigma_max as dim_Sigma_max


@dataclasses.dataclass
class EvaluationResult:
    """Result of parameter evaluation.

    Attributes
    ----------
    mass_deposition_ok : bool
        True if dt_used < dt_max (mass deposition constraint satisfied).
    instability_ok : bool
        True if Sigma_ss / Sigma_max > instability threshold.
    valid : bool
        True if both constraints are satisfied.
    dt_cfl : float
        Raw CFL timestep (s).
    dt_used : float
        CFL * dt_mult, clamped to [dt_floor, dt_cap] (s).
    dt_max : float
        Mass deposition limit (s).
    deposition_margin : float
        dt_max / dt_used (>1 means safe).
    sigma_ss : float
        Cold-disk steady-state surface density (g/cm^2).
    sigma_max : float
        DIM critical Sigma_max (g/cm^2).
    instability_ratio : float
        Sigma_ss / Sigma_max.
    t_viscous : float
        Viscous timescale at R_K (s).
    t_thermal : float
        Thermal timescale at R_K with alpha_hot (s).
    thermal_resolution_ok : bool
        True if dt_used <= thermal_mult * t_thermal. When False, the
        simulation needs adaptive timestep control during outbursts.
    thermal_margin : float
        thermal_mult * t_thermal / dt_used (>1 means resolved without
        adaptive dt; <1 means adaptive dt is required).
    """

    mass_deposition_ok: bool
    instability_ok: bool
    valid: bool
    dt_cfl: float
    dt_used: float
    dt_max: float
    deposition_margin: float
    sigma_ss: float
    sigma_max: float
    instability_ratio: float
    t_viscous: float
    t_thermal: float
    thermal_resolution_ok: bool
    thermal_margin: float


class ParameterEvaluation:
    """Pre-flight parameter evaluation for DIM simulations.

    Checks two critical constraints:

    1. **Mass deposition** — ``dt_used < dt_max`` prevents ``add_mass`` from
       silently skipping deposition (the Sigma < 200 safety cap).
    2. **DIM instability** — ``Sigma_ss / Sigma_max > threshold`` ensures the
       cold-disk steady-state surface density is high enough for outbursts.

    The constructor sets up the physical system (grid, cold-disk viscosity);
    :meth:`evaluate` tests a specific timestep configuration, allowing
    multiple dt strategies to be tested on the same system.

    Parameters
    ----------
    M_star : float
        Central object mass (g).
    R_1 : float
        Inner disk radius (cm).
    R_K : float
        Circularisation (mass transfer) radius (cm).
    R_N : float
        Outer disk radius (cm).
    M_dot : float
        Mass transfer rate (g/s).
    alpha_cold : float
        Cold-state alpha viscosity.
    alpha_hot : float
        Hot-state alpha viscosity.
    N : int
        Number of base grid cells.
    N_n : int
        Extra grid cells beyond the outer radius (matches notebook convention).
    T_cold : float
        Cold-disk temperature (K) for the analytical viscosity estimate.
    instability_threshold : float
        Minimum Sigma_ss / Sigma_max ratio to consider the disk unstable.
    """

    def __init__(self, M_star, R_1, R_K, R_N, M_dot,
                 alpha_cold=ALPHA_COLD, alpha_hot=ALPHA_HOT,
                 N=100, N_n=3, T_cold=1000.0,
                 instability_threshold=0.3):
        self.M_star = M_star
        self.R_1 = R_1
        self.R_K = R_K
        self.R_N = R_N
        self.M_dot = M_dot
        self.alpha_cold = alpha_cold
        self.alpha_hot = alpha_hot
        self.N_base = N
        self.N_n = N_n
        self.T_cold = T_cold
        self.instability_threshold = instability_threshold

        # Build X grid matching notebook convention:
        # 1. N points from X_1 to X_N → get dX
        # 2. Extend by N_n cells → total N + N_n points
        X_1 = X_func(R_1)
        X_N = X_func(R_N)
        X_base = np.linspace(X_1, X_N, N)
        dX = float(np.diff(X_base)[0])
        self.X = np.linspace(X_1, X_N + N_n * dX, N + N_n)
        self.dX = float(np.diff(self.X)[0])
        self.N = N + N_n
        self.X_K = X_func(R_K)

        # Compute cold-disk viscosity analytically at each grid point
        R = R_func(self.X)
        omega_arr = np.sqrt(G * M_star / R**3)
        cs = np.sqrt(k_B * T_cold / (mu * m_p))
        H = cs / omega_arr
        self.nu_array = (2.0 / 3.0) * alpha_cold * omega_arr * H**2

        # Viscosity at R_K (interpolate to circularisation radius)
        self.nu_K = float(np.interp(self.X_K, self.X, self.nu_array))

    def evaluate(self, dt_mult, dt_floor=0.0, dt_cap=None, thermal_mult=20.0):
        """Evaluate parameter constraints for a given timestep configuration.

        Parameters
        ----------
        dt_mult : float
            Timestep multiplier applied to the CFL timestep.
        dt_floor : float
            Minimum allowed timestep (s).
        dt_cap : float or None
            Maximum allowed timestep (s). None means no cap.
        thermal_mult : float
            Maximum number of thermal timescales per timestep during
            outbursts (default 20). The thermal timescale at R_K is
            ``t_th = 1 / (alpha_hot * Omega_K)``. If ``dt_used`` exceeds
            ``thermal_mult * t_th``, the alpha(T) transition is unresolved
            and the simulation requires adaptive timestep control.

        Returns
        -------
        EvaluationResult
            Evaluation results with constraint checks and diagnostics.
        """
        # 1. CFL timestep
        dt_cfl = calculate_timestep(self.X, self.nu_array, self.dX)

        # 2. Applied timestep with floor and cap
        dt_used = dt_cfl * dt_mult
        dt_used = max(dt_used, dt_floor)
        if dt_cap is not None:
            dt_used = min(dt_used, dt_cap)

        # 3. Mass deposition limit: dt_max = 200 * pi * X_K^3 * dX / (4 * M_dot)
        dt_max = 50.0 * np.pi * self.X_K**3 * self.dX / self.M_dot

        # 4. Cold-disk steady-state surface density
        sigma_ss = self.M_dot / (3.0 * np.pi * self.nu_K)

        # 5. DIM critical surface density (no-irradiation baseline)
        sigma_max = dim_Sigma_max(0.0, self.R_K, self.M_star, self.alpha_cold)

        # 6. Viscous timescale at R_K
        t_viscous = self.R_K**2 / self.nu_K

        # 7. Thermal timescale at R_K with alpha_hot
        #    t_thermal = 1 / (alpha_hot * Omega_K) where Omega_K = sqrt(G*M/R^3)
        omega_K = np.sqrt(G * self.M_star / self.R_K**3)
        t_thermal = 1.0 / (self.alpha_hot * omega_K)
        dt_thermal_limit = thermal_mult * t_thermal
        thermal_resolution_ok = bool(dt_used <= dt_thermal_limit)
        thermal_margin = dt_thermal_limit / dt_used

        # Constraint checks
        mass_deposition_ok = bool(dt_used < dt_max)
        deposition_margin = dt_max / dt_used
        instability_ratio = sigma_ss / sigma_max
        instability_ok = bool(instability_ratio > self.instability_threshold)

        return EvaluationResult(
            mass_deposition_ok=mass_deposition_ok,
            instability_ok=instability_ok,
            valid=mass_deposition_ok and instability_ok,
            dt_cfl=dt_cfl,
            dt_used=dt_used,
            dt_max=dt_max,
            deposition_margin=deposition_margin,
            sigma_ss=sigma_ss,
            sigma_max=sigma_max,
            instability_ratio=instability_ratio,
            t_viscous=t_viscous,
            t_thermal=t_thermal,
            thermal_resolution_ok=thermal_resolution_ok,
            thermal_margin=thermal_margin,
        )
