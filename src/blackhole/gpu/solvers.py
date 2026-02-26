"""GPU-accelerated solvers for temperature and scale height.

Performance optimizations
-------------------------
- **Fused temperature RawKernel**: the entire per-element secant solve
  (50 iterations × full opacity computation + 7 fallback guesses) runs
  in a single CUDA kernel launch.  Each thread solves one grid point
  independently, eliminating ~500 per-iteration kernel launches and all
  Python-level loop overhead.
- **Analytical scale height**: the pressure balance is a quadratic in H
  and is solved directly via the quadratic formula — no iterative solver.
- **Batched secant fallback**: when CuPy is unavailable, the temperature
  solver uses a batched vectorized secant method with fused
  ElementwiseKernels for the energy-balance residual and hoisted loop
  invariants.

Drop-in replacements for :func:`blackhole.solvers.solve_temperature` and
:func:`blackhole.solvers.solve_scale_height`.
"""

from blackhole.constants import (
    X_HYDROGEN,
    Z_METALS,
    Z_STAR,
    G,
    a_rad,
    c,
    mu,
    r_gas,
)
from blackhole.gpu import HAS_CUPY, get_xp
from blackhole.gpu.disk_physics import kinematic_viscosity, omega
from blackhole.gpu.opacity import kappa_tot

# ---------------------------------------------------------------------------
# Pre-computed opacity constants (embedded in fused kernels & CPU path)
# ---------------------------------------------------------------------------
_KE_COEFF = 0.2 * (1.0 + X_HYDROGEN)
_KK_COEFF = 4e25 * (1.0 + X_HYDROGEN) * (Z_METALS + 0.001)
_KM_VAL = 0.1 * Z_METALS
_KC_COEFF = 2.6e-7 * Z_STAR
_FRAD_COEFF = 2.0 * a_rad * c

# ---------------------------------------------------------------------------
# Fused GPU kernels (created at import time when CuPy is available)
# ---------------------------------------------------------------------------
_energy_kernel = None
_pressure_kernel = None

if HAS_CUPY:
    import cupy as _cp

    _energy_kernel = _cp.ElementwiseKernel(
        'float64 T, float64 rho, float64 Sigma, '
        'float64 f_visc, float64 F_irr',
        'float64 res',
        f'''
        if (T < 1.0 || Sigma <= 0.0 || rho <= 0.0) {{
            res = 1e30;
        }} else {{
            double T2 = T * T;
            double T4 = T2 * T2;

            // kappa_e: electron scattering
            double ke_a = {_KE_COEFF!r} / (1.0 + 2.7e11 * (rho / T2));
            double ke_b = 1.0 / (1.0 + pow(T / 4.5e8, 0.86));
            double ke = ke_a * ke_b;

            // kappa_K: Kramers
            double kk = {_KK_COEFF!r} * rho / pow(T, 3.5);

            // kappa_Hminus
            double khm = 1.1e-25 * sqrt({Z_METALS!r} * rho) * pow(T, 7.7);

            // kappa_rad: harmonic mean blend
            double high = ke + kk;
            double inv_khm = (khm > 1e-300) ? (1.0 / khm) : 1e300;
            double inv_high = (high > 1e-300) ? (1.0 / high) : 1e300;
            double inv_sum_r = inv_khm + inv_high;
            double blend = (inv_sum_r > 1e-300) ? (1.0 / inv_sum_r) : 0.0;
            double kr = {_KM_VAL!r} + blend;

            // kappa_cond
            double rho_sq = rho * rho;
            if (rho_sq < 1e-300) rho_sq = 1e-300;
            double kc = {_KC_COEFF!r}
                * (T2 / rho_sq)
                * (1.0 + pow(rho / 2e6, 2.0 / 3.0));

            // kappa_tot: harmonic mean of kr and kc
            double inv_kr = (kr > 1e-300) ? (1.0 / kr) : 1e300;
            double inv_kc = (kc > 1e-300) ? (1.0 / kc) : 1e300;
            double inv_sum_t = inv_kr + inv_kc;
            double kappa = (inv_sum_t > 1e-300) ? (1.0 / inv_sum_t) : 0.0;

            // f_rad
            double denom = 3.0 * Sigma * kappa;
            if (denom < 1e-300) denom = 1e-300;
            double f_rad_val = ({_FRAD_COEFF!r} * T4) / denom;

            res = 2.0 * f_visc + F_irr - f_rad_val;
        }}
        ''',
        'energy_balance_fused',
    )

    _pressure_kernel = _cp.ElementwiseKernel(
        'float64 H, float64 gas_coeff, float64 p_rad, float64 hydro_coeff',
        'float64 res',
        '''
        if (H < 1.0) {
            res = 1e30;
        } else {
            res = gas_coeff / H + p_rad - hydro_coeff * H;
        }
        ''',
        'pressure_balance_fused',
    )


# ---------------------------------------------------------------------------
# Fused temperature solver RawKernel (entire secant solve per thread)
# ---------------------------------------------------------------------------
_solve_temp_kernel = None

if HAS_CUPY:
    _solve_temp_kernel = _cp.RawKernel(
        r"""
__device__ double energy_balance(
        double T, double rho, double Sigma,
        double f_visc, double F_irr) {
    if (T < 1.0 || Sigma <= 0.0 || rho <= 0.0) return 1e30;

    double T2 = T * T;
    double T4 = T2 * T2;

    // kappa_e: electron scattering
    double ke_a = """
        + repr(_KE_COEFF)
        + r""" / (1.0 + 2.7e11 * (rho / T2));
    double ke_b = 1.0 / (1.0 + pow(T / 4.5e8, 0.86));
    double ke = ke_a * ke_b;

    // kappa_K: Kramers
    double kk = """
        + repr(_KK_COEFF)
        + r""" * rho / pow(T, 3.5);

    // kappa_Hminus
    double khm = 1.1e-25 * sqrt("""
        + repr(Z_METALS)
        + r""" * rho) * pow(T, 7.7);

    // kappa_rad: harmonic mean blend
    double high = ke + kk;
    double inv_khm = (khm > 1e-300) ? (1.0 / khm) : 1e300;
    double inv_high = (high > 1e-300) ? (1.0 / high) : 1e300;
    double inv_sum_r = inv_khm + inv_high;
    double blend = (inv_sum_r > 1e-300) ? (1.0 / inv_sum_r) : 0.0;
    double kr = """
        + repr(_KM_VAL)
        + r""" + blend;

    // kappa_cond
    double rho_sq = rho * rho;
    if (rho_sq < 1e-300) rho_sq = 1e-300;
    double kc = """
        + repr(_KC_COEFF)
        + r""" * (T2 / rho_sq) * (1.0 + pow(rho / 2e6, 2.0 / 3.0));

    // kappa_tot: harmonic mean of kr and kc
    double inv_kr = (kr > 1e-300) ? (1.0 / kr) : 1e300;
    double inv_kc = (kc > 1e-300) ? (1.0 / kc) : 1e300;
    double inv_sum_t = inv_kr + inv_kc;
    double kappa = (inv_sum_t > 1e-300) ? (1.0 / inv_sum_t) : 0.0;

    // f_rad
    double denom = 3.0 * Sigma * kappa;
    if (denom < 1e-300) denom = 1e-300;
    double f_rad = ("""
        + repr(_FRAD_COEFF)
        + r""" * T4) / denom;

    return 2.0 * f_visc + F_irr - f_rad;
}

__device__ bool secant_solve(
        double x0, double rho, double Sigma,
        double f_visc, double F_irr,
        double* result) {
    const double TOL  = 1.48e-8;
    const int MAXITER = 50;
    const double T_LO = 1.0;
    const double T_HI = 2e10;

    double p0 = x0;
    double p1 = p0 * 1.0001 + 1e-4;
    double q0 = energy_balance(p0, rho, Sigma, f_visc, F_irr);
    double q1 = energy_balance(p1, rho, Sigma, f_visc, F_irr);
    double q_init = fabs(q0);

    if (isnan(q0) || isnan(q1)) return false;

    for (int i = 0; i < MAXITER; i++) {
        double dq = q1 - q0;
        if (fabs(dq) == 0.0) break;
        double p = p1 - q1 * (p1 - p0) / dq;
        if (isnan(p) || isinf(p) || p < T_LO || p > T_HI)
            return false;
        if (fabs(p - p1) <= TOL) { p1 = p; break; }
        p0 = p1; q0 = q1; p1 = p;
        q1 = energy_balance(p1, rho, Sigma, f_visc, F_irr);
        if (isnan(q1)) return false;
    }

    // Residual verification (matches CPU secant logic)
    double qf = energy_balance(p1, rho, Sigma, f_visc, F_irr);
    if (isnan(qf) || isinf(qf)) return false;
    if (fabs(qf) > 1e-4 * fmax(q_init, 1.0)) return false;
    if (p1 <= T_LO || p1 >= T_HI) return false;

    *result = p1;
    return true;
}

extern "C" __global__
void solve_temperature_fused(
        const double* __restrict__ Sigma,
        const double* __restrict__ rho,
        const double* __restrict__ f_visc,
        const double* __restrict__ F_irr,
        const double* __restrict__ T_c,
        double* __restrict__ T_out,
        const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    T_out[i] = T_c[i];          // default: keep current
    if (Sigma[i] <= 1e-100) return;

    double result;

    // Primary guess: 1.5x current (matches CPU S-curve branch selection)
    if (secant_solve(T_c[i] * 1.5, rho[i], Sigma[i],
                     f_visc[i], F_irr[i], &result)) {
        T_out[i] = result;
        return;
    }

    // Fallback guesses
    const double guesses[7] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6};
    for (int g = 0; g < 7; g++) {
        if (secant_solve(guesses[g], rho[i], Sigma[i],
                         f_visc[i], F_irr[i], &result)) {
            T_out[i] = result;
            return;
        }
    }
}
""",
        "solve_temperature_fused",
    )


# ---------------------------------------------------------------------------
# CPU-path residual functions (hoisted invariants, no fused kernel)
# ---------------------------------------------------------------------------

def _energy_balance_hoisted(T, rho, Sigma, f_visc, F_irr):
    """Energy-balance residual with pre-computed rho and f_visc."""
    xp = get_xp(T, Sigma)
    bad = (Sigma <= 0.0) | (T < 1.0) | (rho <= 0.0)
    T_safe = xp.where(bad, 1.0, T)
    Sigma_safe = xp.where(bad, 1.0, Sigma)
    rho_safe = xp.where(bad, 1.0, rho)

    kappa = kappa_tot(rho_safe, T_safe)
    denom = 3.0 * Sigma_safe * kappa
    denom = xp.maximum(denom, 1e-300)
    f_rad = (_FRAD_COEFF * T_safe ** 4) / denom

    result = 2.0 * f_visc + F_irr - f_rad
    return xp.where(bad, 1e30, result)


def _pressure_balance_hoisted(H, gas_coeff, p_rad, hydro_coeff):
    """Pressure-balance residual with pre-computed coefficients."""
    xp = get_xp(H)
    bad = H < 1.0
    H_safe = xp.where(bad, 1.0, H)
    result = gas_coeff / H_safe + p_rad - hydro_coeff * H_safe
    return xp.where(bad, 1e30, result)


# ---------------------------------------------------------------------------
# Generic batched secant method
# ---------------------------------------------------------------------------

_SECANT_CHECK_INTERVAL = 10  # iterations between GPU sync checks


def _batched_secant(root_func, x0, args, tol=1.48e-8, maxiter=50,
                    x_min=None, x_max=None):
    """Batched secant method solving N independent roots simultaneously.

    Matches the CPU scalar secant algorithm:

    * Out-of-bounds steps (``p < x_min``, ``p > x_max``, NaN, inf)
      **hard-fail** that point — identical to the CPU's
      ``if p < 1.0 or p > 1e20: return p1, False``.
    * After *maxiter* iterations, a point is accepted if its final
      residual is small relative to the initial residual — the CPU
      does NOT require step-convergence, only the residual check.

    GPU performance note: early-termination checks (``xp.any()``) are
    performed only every *check_interval* iterations instead of every
    iteration, reducing GPU-to-host synchronisation barriers from
    ~2 per iteration to ~2 per *check_interval* iterations.
    Converged/failed points are frozen by ``xp.where`` masks on
    non-check iterations.

    Parameters
    ----------
    root_func : callable
        ``f(x, *args)`` returning an array of residuals (shape N).
    x0 : array, shape (N,)
        Initial guess for each root.
    args : tuple
        Extra arguments forwarded to *root_func*.
    tol : float
        Convergence tolerance (absolute).
    maxiter : int
        Maximum number of secant iterations.
    x_min, x_max : float, optional
        Hard bounds.  If an unclamped step leaves this range the point
        is permanently marked as failed.

    Returns
    -------
    result : array, shape (N,)
        Best root estimate for each element.
    converged : array of bool, shape (N,)
        Whether the final residual is small relative to the initial
        residual and the point was not hard-failed.
    """
    xp = get_xp(x0)

    p0 = x0.copy()
    p1 = p0 * (1.0 + 1e-4) + 1e-4

    q0 = root_func(p0, *args)
    q1 = root_func(p1, *args)

    # Save initial residual for false-convergence detection (matches CPU)
    q_init = xp.abs(q0)

    # NaN in initial evaluations → immediate failure
    init_bad = xp.isnan(q0) | xp.isnan(q1)

    converged = xp.zeros(len(x0), dtype=bool)
    # Points permanently removed from iteration (like CPU ``return p1, False``)
    failed = init_bad.copy()

    for _iter in range(maxiter):
        active = ~converged & ~failed
        # Periodic early-exit check — only every _SECANT_CHECK_INTERVAL
        # iterations to minimise GPU sync barriers.
        if _iter % _SECANT_CHECK_INTERVAL == 0:
            if not xp.any(active):
                break

        dq = q1 - q0
        # Where dq == 0 the secant step is undefined; freeze those points
        safe = xp.abs(dq) > 0.0
        dq_safe = xp.where(safe, dq, 1.0)
        p_raw = p1 - q1 * (p1 - p0) / dq_safe
        # Freeze points where dq was zero
        p_raw = xp.where(safe, p_raw, p1)

        # --- Hard-fail out-of-bounds (matches CPU early return) ---
        oob = xp.isnan(p_raw) | xp.isinf(p_raw)
        if x_min is not None:
            oob = oob | (p_raw < x_min)
        if x_max is not None:
            oob = oob | (p_raw > x_max)
        newly_failed = oob & active
        failed = failed | newly_failed

        # For non-failed active points, use the new value
        p_new = xp.where(~failed & active, p_raw, p1)

        # Check convergence (only on non-failed, active points)
        newly_converged = (xp.abs(p_new - p1) <= tol) & active & ~failed
        converged = converged | newly_converged

        # Update for next iteration (only active, non-failed points)
        still_active = ~converged & ~failed
        p0 = xp.where(still_active, p1, p0)
        q0 = xp.where(still_active, q1, q0)
        p1 = xp.where(still_active, p_new, p1)

        # Periodic early-exit check before expensive root_func call
        if _iter % _SECANT_CHECK_INTERVAL == _SECANT_CHECK_INTERVAL - 1:
            if not xp.any(still_active):
                break

        q1_new = root_func(p1, *args)
        # NaN in residual → fail that point
        nan_q = xp.isnan(q1_new) & still_active
        failed = failed | nan_q
        q1 = xp.where(still_active & ~nan_q, q1_new, q1)

    # --- Residual verification (matches CPU _secant_temperature logic) ---
    # The CPU accepts ANY point (even without step-convergence) if the
    # final residual is small.  Step-convergence only terminates the loop
    # early — it is not required for acceptance.
    q_final = root_func(p1, *args)
    residual_ok = xp.abs(q_final) <= 1e-4 * xp.maximum(q_init, 1.0)
    nan_final = xp.isnan(q_final) | xp.isinf(q_final)
    accepted = residual_ok & ~nan_final & ~failed

    return p1, accepted


# ---------------------------------------------------------------------------
# Temperature solver
# ---------------------------------------------------------------------------

def solve_temperature(H, Sigma, r, T_c, alpha, M_star, F_irr=None):
    """Solve for midplane temperature at each grid point (vectorized).

    Drop-in replacement for :func:`blackhole.solvers.solve_temperature`.
    On GPU, launches a single fused CUDA kernel where each thread runs the
    full secant solve (including fallback guesses) for one grid point.
    On CPU, uses the batched vectorized secant method.

    Parameters
    ----------
    H : array
        Scale height array (cm).
    Sigma : array
        Surface density array (g/cm^2).
    r : array
        Radius array (cm).
    T_c : array
        Current midplane temperature array (K).
    alpha : array or float
        Viscosity parameter (scalar or per-cell).
    M_star : float
        Central object mass (g).
    F_irr : array, optional
        Irradiation flux at each point.  If ``None``, zero everywhere.

    Returns
    -------
    array
        Updated temperature array (does not mutate input).
    """
    xp = get_xp(H, Sigma, r, T_c)

    T_new = T_c.copy()
    alpha_arr = xp.broadcast_to(
        xp.asarray(alpha, dtype=xp.float64), Sigma.shape
    ).copy()
    F_irr_arr = (
        xp.zeros_like(Sigma) if F_irr is None
        else xp.asarray(F_irr, dtype=xp.float64)
    )

    # Mask: only solve where Sigma is significant
    active = Sigma > 1e-100

    if not xp.any(active):
        return T_new

    # Extract active subset
    H_a = H[active]
    Sig_a = Sigma[active]
    r_a = r[active]
    Tc_a = T_c[active]
    al_a = alpha_arr[active]
    fi_a = F_irr_arr[active]

    # --- Hoist invariants ---
    rho_a = Sig_a / (2.0 * H_a)
    nu_a = kinematic_viscosity(H_a, r_a, al_a, M_star)
    om_a = omega(r_a, M_star)
    fv_a = (9.0 / 8.0) * nu_a * Sig_a * om_a ** 2

    N_active = len(Sig_a)

    # --- GPU path: fused RawKernel (one launch, full solve per thread) ---
    if _solve_temp_kernel is not None and xp.__name__ == 'cupy':
        import numpy as np

        T_out = Tc_a.copy()
        block = 256
        grid = (N_active + block - 1) // block
        _solve_temp_kernel(
            (grid,), (block,),
            (Sig_a, rho_a, fv_a, fi_a, Tc_a, T_out, np.int32(N_active)),
        )
        T_new[active] = T_out
        return T_new

    # --- CPU path: batched secant with fused ElementwiseKernel fallback ---
    T_MIN = 1e0
    T_MAX = 2e10

    _use_fused = _energy_kernel is not None and xp.__name__ == 'cupy'

    def _root_T(T, rho_v, Sigma_v, fv_v, fi_v):
        if _use_fused:
            return _energy_kernel(T, rho_v, Sigma_v, fv_v, fi_v)
        return _energy_balance_hoisted(T, rho_v, Sigma_v, fv_v, fi_v)

    args = (rho_a, Sig_a, fv_a, fi_a)

    # Primary guess: 1.5x current value (matches CPU S-curve branch selection)
    x0 = Tc_a * 1.5
    result, converged = _batched_secant(
        _root_T, x0, args, x_min=T_MIN, x_max=T_MAX,
    )

    # Accept converged results that are in range
    good = converged & (result > T_MIN) & (result < T_MAX)

    # For points that failed, try fallback guesses
    need_fallback = ~good
    if xp.any(need_fallback):
        fallback_guesses = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        for guess_val in fallback_guesses:
            if not xp.any(need_fallback):
                break
            x0_fb = xp.full(int(xp.sum(need_fallback)), guess_val, dtype=xp.float64)
            args_fb = (
                rho_a[need_fallback],
                Sig_a[need_fallback],
                fv_a[need_fallback],
                fi_a[need_fallback],
            )
            res_fb, conv_fb = _batched_secant(
                _root_T, x0_fb, args_fb, x_min=T_MIN, x_max=T_MAX,
            )
            good_fb = conv_fb & (res_fb > T_MIN) & (res_fb < T_MAX)
            if xp.any(good_fb):
                fb_indices = xp.where(need_fallback)[0]
                success_indices = fb_indices[good_fb]
                result[success_indices] = res_fb[good_fb]
                good[success_indices] = True
                need_fallback = ~good

    # Write results back
    T_active = Tc_a.copy()
    T_active[good] = result[good]
    T_new[active] = T_active

    return T_new


# ---------------------------------------------------------------------------
# Scale-height solver
# ---------------------------------------------------------------------------

def solve_scale_height(H, Sigma, r, T_c, M_star):
    """Solve for scale height at each grid point (direct quadratic solution).

    The hydrostatic pressure balance

        P_gas / H  +  P_rad  =  hydro_coeff * H

    is a quadratic in H:

        hydro_coeff * H^2  -  P_rad * H  -  gas_coeff  =  0

    whose unique positive root is computed directly via the quadratic
    formula, replacing the iterative secant solver with a single array
    operation.  Works with NumPy and CuPy arrays.

    Parameters
    ----------
    H : array
        Current scale height array (cm) — used only for inactive cells.
    Sigma : array
        Surface density array (g/cm^2).
    r : array
        Radius array (cm).
    T_c : array
        Midplane temperature array (K).
    M_star : float
        Central object mass (g).

    Returns
    -------
    array
        Updated scale height array (does not mutate input).
    """
    xp = get_xp(H, Sigma, r, T_c)

    H_new = H.copy()
    H_MIN = 1e7
    H_MAX = 2e10

    active = Sigma > 1e-100

    if not xp.any(active):
        return H_new

    H_a = H[active]
    Sig_a = Sigma[active]
    r_a = r[active]
    Tc_a = T_c[active]

    # Quadratic coefficients: hydro * H^2 - p_rad * H - gas_coeff = 0
    gas_coeff = (r_gas * Sig_a * Tc_a) / (mu * 2.0)
    p_rad = (1.0 / 3.0) * a_rad * Tc_a ** 4
    hydro_coeff = 0.5 * Sig_a * (G * M_star / r_a ** 3)

    # Discriminant is always positive (all terms > 0 for active cells)
    disc = p_rad ** 2 + 4.0 * hydro_coeff * gas_coeff
    H_sol = (p_rad + xp.sqrt(disc)) / (2.0 * hydro_coeff)

    # Keep old H where root falls outside physical bounds (matches CPU)
    in_range = (H_sol > H_MIN) & (H_sol < H_MAX)
    H_new[active] = xp.where(in_range, H_sol, H_a)

    return H_new
