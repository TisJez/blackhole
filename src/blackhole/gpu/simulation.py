"""GPU simulation orchestrator — keeps arrays GPU-resident.

Provides a :class:`SimulationConfig` dataclass and :func:`run_simulation`
function that mirrors the timestep loop from the CPU notebooks but runs
all physics via :mod:`blackhole.gpu` array operations.

``add_mass`` is called on the CPU (inherently sequential); all other
operations run on whichever device the arrays reside on.
"""

import dataclasses
import warnings

import numpy as np

from blackhole.constants import G, M_sun, c
from blackhole.evolution import add_mass  # CPU-only, sequential
from blackhole.gpu import get_xp, is_gpu_available, to_device, to_host
from blackhole.gpu.disk_physics import (
    R_func,
    X_func,
    kinematic_viscosity,
)
from blackhole.gpu.evolution import (
    disk_evap,
    evolve_surface_density,
)
from blackhole.gpu.solvers import solve_scale_height, solve_temperature
from blackhole.gpu.viscosity import alpha_visc
from blackhole.irradiation import (
    Epsilon_irr,
    Flux_irr,
    alpha_visc_irr,
)
from blackhole.irradiation import Sigma_max as dim_Sigma_max

# GPU kernel launch overhead exceeds computation benefit below this grid size.
GPU_N_THRESHOLD = 500


@dataclasses.dataclass
class SimulationConfig:
    """Configuration for a time-dependent accretion disk simulation.

    Attributes
    ----------
    M_star : float
        Central object mass (g).
    R_1 : float
        Inner disk radius (cm).
    R_K : float
        Circularisation radius (cm).
    R_N : float
        Outer disk radius (cm).
    M_dot : float
        Mass transfer rate (g/s).
    alpha_cold : float
        Cold-state alpha viscosity.
    alpha_hot : float
        Hot-state alpha viscosity.
    N_base : int
        Base number of grid points (before N_n extension).
    N_n : int
        Extra grid points beyond outer radius.
    min_Sigma : float
        Minimum surface density floor.
    timesteps : int
        Total number of timesteps to simulate.
    output_interval : int
        Save a snapshot every *output_interval* timesteps.
    dt_multiplier : float
        Multiply CFL timestep by this factor.
    dt_min : float
        Minimum timestep floor (s).
    tidal_params : dict or None
        Tidal torque parameters.
    enable_irradiation : bool
        If ``True``, compute irradiation feedback.
    enable_evaporation : bool
        If ``True``, apply disk evaporation.
    theta : float
        Implicitness parameter (0=explicit, 0.5=Crank-Nicolson).
    sigma_cap : float or None
        Safety cap for ``add_mass``.  ``None`` = auto-scale from Sigma_max.
    dt_max : float or None
        Maximum timestep cap (for SMBH simulations).
    thermal_dt_factor : float
        Factor for thermal-timescale dt constraint during outbursts.
    use_gpu : bool
        If ``True`` and CuPy is available, run on GPU.
    """

    M_star: float
    R_1: float
    R_K: float
    R_N: float
    M_dot: float
    alpha_cold: float = 0.04
    alpha_hot: float = 0.2
    N_base: int = 100
    N_n: int = 3
    min_Sigma: float = 1e-5
    timesteps: int = 1_500_001
    output_interval: int = 5
    dt_multiplier: float = 30.0
    dt_min: float = 200.0
    tidal_params: dict = None
    enable_irradiation: bool = False
    enable_evaporation: bool = False
    theta: float = 0.5
    sigma_cap: float = None
    dt_max: float = None
    thermal_dt_factor: float = 20.0
    use_gpu: bool = True


@dataclasses.dataclass
class SimulationResult:
    """Result container for simulation output.

    Attributes
    ----------
    Sigma_history : list[np.ndarray]
        Snapshots of surface density (CPU arrays).
    Temp_history : list[np.ndarray]
        Snapshots of midplane temperature.
    H_history : list[np.ndarray]
        Snapshots of scale height.
    alpha_history : list[np.ndarray]
        Snapshots of alpha viscosity.
    t_history : list[float]
        Cumulative time at each snapshot.
    Sigma_transfer_history : list[float]
        Inner-edge surface density at each snapshot.
    """

    Sigma_history: list
    Temp_history: list
    H_history: list
    alpha_history: list
    t_history: list
    Sigma_transfer_history: list


def _find_inner_index(Sigma, min_Sigma):
    """Find index of first cell with significant surface density."""
    xp = get_xp(Sigma)
    candidates = xp.where(Sigma > min_Sigma * 21)[0]
    if len(candidates) == 0:
        candidates = xp.where(Sigma > min_Sigma)[0]
    return int(candidates[0]) if len(candidates) > 0 else 0


def _compute_irr_quantities(Sigma, nu, r, M_star, M_dot, min_Sigma):
    """Compute irradiation epsilon and flux from disk state."""
    xp = get_xp(Sigma)
    candidates = xp.where(Sigma > min_Sigma * 21)[0]
    if len(candidates) == 0:
        candidates = xp.where(Sigma > min_Sigma)[0]
    if len(candidates) == 0:
        zeros = xp.zeros_like(r)
        return zeros, zeros
    inner_idx = int(candidates[0])
    # Irradiation functions are already CuPy-compatible (no JIT)
    Sigma_inner = float(Sigma[inner_idx])
    nu_inner = float(nu[inner_idx])
    r_inner = float(r[inner_idx])
    # These functions use np.where which works on device arrays too
    r_host = to_host(r)
    eps = Epsilon_irr(Sigma_inner, nu_inner, r_inner, r_host, M_star, M_dot)
    flux = Flux_irr(Sigma_inner, nu_inner, r_inner, r_host, M_star, M_dot)
    return xp.asarray(eps, dtype=xp.float64), xp.asarray(flux, dtype=xp.float64)


def run_simulation(config, perf_logger=None):
    """Run a full time-dependent accretion disk simulation.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters.
    perf_logger : PerformanceLogger or None
        Optional performance logger.  When provided, every stage of the
        timestep loop is timed (with GPU sync barriers) and results are
        saved to disk at the end.  ``None`` means no instrumentation.

    Returns
    -------
    SimulationResult
        Collected history arrays (all on CPU).
    """
    cfg = config

    # Determine output timesteps
    n_snapshots = max(cfg.timesteps // cfg.output_interval, 1)
    output_times = set(
        np.linspace(1, cfg.timesteps - 1, n_snapshots, dtype="int64")
    )

    # Build grid (on CPU first)
    X_1 = float(X_func(np.float64(cfg.R_1)))
    X_K = float(X_func(np.float64(cfg.R_K)))
    X_N_init = float(X_func(np.float64(cfg.R_N)))

    X_cpu = np.linspace(X_1, X_N_init, cfg.N_base)
    dX = float(np.diff(X_cpu)[0])
    X_cpu = np.linspace(X_1, X_N_init + cfg.N_n * dX, cfg.N_base + cfg.N_n)
    dX = float(np.diff(X_cpu)[0])
    X_N = X_N_init + cfg.N_n * dX
    N = cfg.N_base + cfg.N_n

    r_cpu = R_func(X_cpu)
    dr_cpu = np.diff(r_cpu)
    dr_cpu = np.insert(dr_cpu, 0, 0)

    # Auto-scale sigma_cap if not set
    sigma_cap = cfg.sigma_cap
    if sigma_cap is None:
        sigma_cap = 200.0
        # For SMBH-scale simulations, auto-scale from Sigma_max
        if cfg.M_star > 100 * M_sun:
            sigma_cap = float(
                2.0 * dim_Sigma_max(0.0, cfg.R_K, cfg.M_star, cfg.alpha_cold)
            )

    # --- Auto-dt: physics-ceiling computation (once, before loop) ---
    dt_max_deposit = sigma_cap * np.pi * X_K ** 3 * dX / (4.0 * cfg.M_dot)
    omega_K_ref = np.sqrt(G * cfg.M_star / cfg.R_K ** 3)
    t_thermal_ref = 1.0 / (cfg.alpha_hot * omega_K_ref)
    dt_max_thermal = cfg.thermal_dt_factor * t_thermal_ref
    auto_dt_max = 0.9 * min(dt_max_deposit, dt_max_thermal)

    # User can only tighten, not loosen
    if cfg.dt_max is not None:
        effective_dt_max = min(cfg.dt_max, auto_dt_max)
    else:
        effective_dt_max = auto_dt_max

    # Initial conditions (CPU)
    Sigma_cpu = np.full(N, cfg.min_Sigma)
    T_c_cpu = np.full(N, 1e3)
    H_cpu = np.full(N, 1e7)
    alpha_cpu = np.full(N, cfg.alpha_cold)

    # Move to device if GPU requested and available
    use_device = cfg.use_gpu and is_gpu_available()

    # Auto-fallback: GPU overhead exceeds benefit at small N
    if cfg.use_gpu and N < GPU_N_THRESHOLD:
        warnings.warn(
            f"N={N} is below GPU threshold ({GPU_N_THRESHOLD}); "
            f"falling back to CPU (NumPy). Set N_base >= {GPU_N_THRESHOLD} "
            f"to use GPU, or set use_gpu=False to silence this warning.",
            stacklevel=2,
        )
        use_device = False

    if use_device:
        X = to_device(X_cpu)
        r = to_device(r_cpu)
        Sigma = to_device(Sigma_cpu)
        T_c = to_device(T_c_cpu)
        H = to_device(H_cpu)
        alpha = to_device(alpha_cpu)
    else:
        X = X_cpu.copy()
        r = r_cpu.copy()
        Sigma = Sigma_cpu.copy()
        T_c = T_c_cpu.copy()
        H = H_cpu.copy()
        alpha = alpha_cpu.copy()

    xp = get_xp(Sigma)

    nu = kinematic_viscosity(H, r, alpha, cfg.M_star)
    dt = max(effective_dt_max, cfg.dt_min)

    # Eddington luminosity for evaporation scaling
    L_edd = 1.4e38 * (cfg.M_star / M_sun) if cfg.enable_evaporation else 0.0

    trunc_rad = int(N * cfg.tidal_params["trunc_frac"]) if cfg.tidal_params else N

    # History (always CPU)
    Sigma_history = [to_host(Sigma).copy()]
    Temp_history = [to_host(T_c).copy()]
    H_history = [to_host(H).copy()]
    alpha_history = [to_host(alpha).copy()]
    t_history = [0.0]
    Sigma_transfer_history = [float(Sigma[1])]

    totalt = 0.0

    _pl = perf_logger  # short alias (None → no-op)

    for timestep in range(cfg.timesteps):
        if _pl is not None:
            _pl.set_timestep(timestep)

        # --- Stage 1: Add mass (CPU) ---
        if _pl is not None:
            _pl.start("add_mass")
        Sigma_host = to_host(Sigma)
        result = add_mass(
            Sigma_host, cfg.M_dot, dt, X_cpu, N, X_K, X_N, dX,
            cfg.min_Sigma, sigma_cap=sigma_cap,
        )
        Sigma = xp.asarray(result.Sigma, dtype=xp.float64) if use_device else result.Sigma
        j_val = result.j_val
        if _pl is not None:
            _pl.stop()

        # --- Evaporation setup ---
        if _pl is not None:
            _pl.start("evap_setup")
        evap_func = None
        if cfg.enable_evaporation:
            M_dot_inner = 6.0 * xp.pi * Sigma[1] * nu[1]
            L_actual = 0.1 * float(M_dot_inner) * c ** 2
            L_ratio = min(L_actual / L_edd, 1.0) if L_edd > 0 else 0.0
            _lr = L_ratio

            def evap_func(r_arr, lr=_lr):
                return disk_evap(r_arr, cfg.M_star, L_ratio=lr)
        if _pl is not None:
            _pl.stop()

        # --- Stage 2: Evolve surface density ---
        if _pl is not None:
            _pl.start("evolve_surface_density")
        tp = cfg.tidal_params if (cfg.tidal_params and j_val >= trunc_rad) else None
        Sigma = evolve_surface_density(
            Sigma, dt, nu, X, dX, N, cfg.min_Sigma,
            tidal_params=tp, evap_func=evap_func,
            theta=cfg.theta,
        )
        if _pl is not None:
            _pl.stop()

        # --- Stage 3: Irradiation (optional) ---
        if _pl is not None:
            _pl.start("irradiation")
        F_irr_arr = None
        eps_irr = None
        if cfg.enable_irradiation:
            eps_irr, F_irr_arr = _compute_irr_quantities(
                Sigma, nu, r, cfg.M_star, cfg.M_dot, cfg.min_Sigma,
            )
        if _pl is not None:
            _pl.stop()

        # --- Stage 4: Temperature ---
        if _pl is not None:
            _pl.start("solve_temperature")
        T_c = solve_temperature(
            H, Sigma, r, T_c, alpha, cfg.M_star, F_irr=F_irr_arr,
        )
        if _pl is not None:
            _pl.stop()

        # --- Stage 5: Scale height ---
        if _pl is not None:
            _pl.start("solve_scale_height")
        H = solve_scale_height(H, Sigma, r, T_c, cfg.M_star)
        if _pl is not None:
            _pl.stop()

        # --- Stage 6: Alpha viscosity ---
        if _pl is not None:
            _pl.start("alpha_viscosity")
        if cfg.enable_irradiation and eps_irr is not None:
            eps_host = to_host(eps_irr)
            r_host = to_host(r)
            T_c_host = to_host(T_c)
            alpha_host = alpha_visc_irr(
                T_c_host, eps_host, r_host, cfg.M_star,
                alpha_cold=cfg.alpha_cold, alpha_hot=cfg.alpha_hot,
            )
            alpha = xp.asarray(alpha_host, dtype=xp.float64) if use_device else alpha_host
        else:
            alpha = alpha_visc(
                T_c, alpha_cold=cfg.alpha_cold, alpha_hot=cfg.alpha_hot,
            )
        if _pl is not None:
            _pl.stop()

        # --- Update viscosity and timestep ---
        if _pl is not None:
            _pl.start("kinematic_viscosity")
        nu = kinematic_viscosity(H, r, alpha, cfg.M_star)
        if _pl is not None:
            _pl.stop()

        if _pl is not None:
            _pl.start("thermal_dt_check")
        dt = effective_dt_max

        # Thermal-timescale constraint: reduce dt during outbursts
        alpha_host_check = to_host(alpha)
        alpha_max_val = float(np.max(alpha_host_check[1:-1]))
        if alpha_max_val > 2.0 * cfg.alpha_cold:
            i_hot = int(np.argmax(alpha_host_check[1:-1])) + 1
            r_host_check = to_host(r)
            Omega_K = np.sqrt(G * cfg.M_star / r_host_check[i_hot] ** 3)
            t_thermal = 1.0 / (alpha_max_val * Omega_K)
            dt = min(dt, cfg.thermal_dt_factor * t_thermal)

        dt = max(dt, cfg.dt_min)
        if _pl is not None:
            _pl.stop()

        totalt += dt

        # --- Snapshot ---
        if timestep in output_times:
            if _pl is not None:
                _pl.start("snapshot")
            inner_idx = _find_inner_index(Sigma, cfg.min_Sigma)
            Sigma_transfer_history.append(float(Sigma[inner_idx]))
            Sigma_history.append(to_host(Sigma).copy())
            Temp_history.append(to_host(T_c).copy())
            H_history.append(to_host(H).copy())
            alpha_history.append(to_host(alpha).copy())
            t_history.append(totalt)
            if _pl is not None:
                _pl.stop()

    # Always capture final state (even if not in output_times)
    if len(Sigma_history) == 1 or t_history[-1] != totalt:
        inner_idx = _find_inner_index(Sigma, cfg.min_Sigma)
        Sigma_transfer_history.append(float(Sigma[inner_idx]))
        Sigma_history.append(to_host(Sigma).copy())
        Temp_history.append(to_host(T_c).copy())
        H_history.append(to_host(H).copy())
        alpha_history.append(to_host(alpha).copy())
        t_history.append(totalt)

    if _pl is not None:
        _pl.save("logs")

    return SimulationResult(
        Sigma_history=Sigma_history,
        Temp_history=Temp_history,
        H_history=H_history,
        alpha_history=alpha_history,
        t_history=t_history,
        Sigma_transfer_history=Sigma_transfer_history,
    )
