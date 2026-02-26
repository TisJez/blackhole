"""Batched vectorized secant solvers — pure array ops, no JIT.

The core innovation: instead of looping N grid points with scalar secant
solves, we run one secant iteration across ALL N points simultaneously
via array operations.  This makes the solver CuPy-compatible and yields
significant speedups at large N.

Drop-in replacements for :func:`blackhole.solvers.solve_temperature` and
:func:`blackhole.solvers.solve_scale_height`.
"""

from blackhole.constants import G, a_rad, c, mu, r_gas
from blackhole.gpu import get_xp
from blackhole.gpu.disk_physics import kinematic_viscosity, omega
from blackhole.gpu.opacity import kappa_bath

# ---------------------------------------------------------------------------
# Vectorized residual functions
# ---------------------------------------------------------------------------

def _energy_balance_vec(T, H, Sigma, R, alpha, M_star, F_irr):
    """Vectorized energy-balance residual for temperature solver.

    Returns ``2*F_visc + F_irr - F_rad`` element-wise.  Matches the CPU
    ``_energy_balance_jit`` guard: returns 1e30 where T < 1, H <= 0, or
    Sigma <= 0 to create the same artificial sign change for the secant
    method.
    """
    xp = get_xp(T, H, Sigma, R)

    # Guard: match CPU _energy_balance_jit behaviour
    bad = (H <= 0.0) | (Sigma <= 0.0) | (T < 1.0)

    # Use safe values for computation (avoid div-by-zero), result is
    # overwritten by 1e30 where bad==True.
    T_safe = xp.where(bad, 1.0, T)
    H_safe = xp.where(bad, 1.0, H)
    Sigma_safe = xp.where(bad, 1.0, Sigma)

    nu = kinematic_viscosity(H_safe, R, alpha, M_star)
    om = omega(R, M_star)
    f_visc = (9.0 / 8.0) * nu * Sigma_safe * om ** 2

    kappa = kappa_bath(H_safe, Sigma_safe, T_safe)
    denom = 3.0 * Sigma_safe * kappa
    denom = xp.maximum(denom, 1e-300)
    f_rad = (2.0 * a_rad * c * T_safe ** 4) / denom

    result = 2.0 * f_visc + F_irr - f_rad
    # Apply guard: return 1e30 for invalid inputs (matches CPU)
    return xp.where(bad, 1e30, result)


def _pressure_balance_vec(H, Sigma, R, T, M_star):
    """Vectorized pressure-balance residual for scale-height solver.

    Returns ``pressure - pressure_2`` element-wise.  Matches the CPU
    ``_pressure_balance_jit`` guard: returns 1e30 where H < 1, Sigma <= 0,
    or T <= 0.
    """
    xp = get_xp(H, Sigma, R, T)

    bad = (H < 1.0) | (Sigma <= 0.0) | (T <= 0.0)

    H_safe = xp.where(bad, 1.0, H)
    Sigma_safe = xp.where(bad, 1.0, Sigma)
    T_safe = xp.where(bad, 1.0, T)

    p_gas = (r_gas * Sigma_safe * T_safe) / (mu * 2.0 * H_safe)
    p_rad = (1.0 / 3.0) * a_rad * T_safe ** 4
    p1 = p_gas + p_rad
    p2 = 0.5 * Sigma_safe * H_safe * (G * M_star / R ** 3)
    result = p1 - p2
    return xp.where(bad, 1e30, result)


# ---------------------------------------------------------------------------
# Generic batched secant method
# ---------------------------------------------------------------------------

def _batched_secant(root_func, x0, args, tol=1.48e-8, maxiter=50,
                    x_min=None, x_max=None):
    """Batched secant method solving N independent roots simultaneously.

    Matches the CPU scalar secant algorithm exactly:

    * Out-of-bounds steps (``p < x_min``, ``p > x_max``, NaN, inf)
      **hard-fail** that point — identical to the CPU's
      ``if p < 1.0 or p > 1e20: return p1, False``.
    * After *maxiter* iterations, a point is accepted if its final
      residual is small relative to the initial residual — the CPU
      does NOT require step-convergence, only the residual check.

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

    for _ in range(maxiter):
        active = ~converged & ~failed

        if not xp.any(active):
            break

        dq = q1 - q0
        # Where dq == 0 the secant step is undefined; freeze those points
        safe = xp.abs(dq) > 0.0
        dq_safe = xp.where(safe, dq, xp.ones_like(dq))
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

    Drop-in replacement for :func:`blackhole.solvers.solve_temperature`
    using the batched secant method.  Works with NumPy and CuPy arrays.

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

    T_MIN = 1e0
    T_MAX = 2e10

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

    def _root_T(T, H_v, Sig_v, r_v, al_v, M_s, fi_v):
        return _energy_balance_vec(T, H_v, Sig_v, r_v, al_v, M_s, fi_v)

    args = (H_a, Sig_a, r_a, al_a, M_star, fi_a)

    # Primary guess: 1.5x current value
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
                H_a[need_fallback],
                Sig_a[need_fallback],
                r_a[need_fallback],
                al_a[need_fallback],
                M_star,
                fi_a[need_fallback],
            )
            res_fb, conv_fb = _batched_secant(
                _root_T, x0_fb, args_fb, x_min=T_MIN, x_max=T_MAX,
            )
            good_fb = conv_fb & (res_fb > T_MIN) & (res_fb < T_MAX)
            if xp.any(good_fb):
                # Write back successes
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
    """Solve for scale height at each grid point (vectorized).

    Drop-in replacement for :func:`blackhole.solvers.solve_scale_height`
    using the batched secant method.  Works with NumPy and CuPy arrays.

    Parameters
    ----------
    H : array
        Current scale height array (cm).
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

    def _root_H(Hv, Sig_v, r_v, T_v, M_s):
        return _pressure_balance_vec(Hv, Sig_v, r_v, T_v, M_s)

    args = (Sig_a, r_a, Tc_a, M_star)

    # Primary guess: 1.5x current value
    x0 = H_a * 1.5
    result, converged = _batched_secant(
        _root_H, x0, args, x_min=H_MIN, x_max=H_MAX,
    )

    good = converged & (result > H_MIN) & (result < H_MAX)

    # Fallback guesses
    need_fallback = ~good
    if xp.any(need_fallback):
        fallback_guesses = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
        for guess_val in fallback_guesses:
            if not xp.any(need_fallback):
                break
            x0_fb = xp.full(int(xp.sum(need_fallback)), guess_val, dtype=xp.float64)
            args_fb = (
                Sig_a[need_fallback],
                r_a[need_fallback],
                Tc_a[need_fallback],
                M_star,
            )
            res_fb, conv_fb = _batched_secant(
                _root_H, x0_fb, args_fb, x_min=H_MIN, x_max=H_MAX,
            )
            good_fb = conv_fb & (res_fb > H_MIN) & (res_fb < H_MAX)
            if xp.any(good_fb):
                fb_indices = xp.where(need_fallback)[0]
                success_indices = fb_indices[good_fb]
                result[success_indices] = res_fb[good_fb]
                good[success_indices] = True
                need_fallback = ~good

    H_active = H_a.copy()
    H_active[good] = result[good]
    H_new[active] = H_active

    return H_new
