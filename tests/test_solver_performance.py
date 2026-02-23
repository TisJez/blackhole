"""Performance tests for blackhole.solvers.

Verifies that the JIT'd secant-method solvers meet wall-clock time budgets
for the grid sizes used in production simulations (N=103).

Each test warms up the solver (JIT compilation on first call), then times
a batch of repeated calls to measure steady-state throughput.
"""

import time

import numpy as np

from blackhole.constants import M_sun
from blackhole.solvers import (
    _secant_scale_height,
    _secant_temperature,
    _solve_scale_height_jit,
    _solve_temperature_jit,
    solve_scale_height,
    solve_temperature,
)

M_STAR = 10.0 * M_sun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_disk(N):
    """Create a physically realistic disk state for benchmarking."""
    r = np.linspace(5e8, 4.2e11, N)
    Sigma = np.full(N, 1e2)
    H = np.full(N, 1e8)
    T_c = np.full(N, 1e4)
    alpha = np.full(N, 0.1)
    F_irr = np.zeros(N)
    return r, Sigma, H, T_c, alpha, F_irr


def _time_calls(func, args, n_calls):
    """Time n_calls invocations of func(*args), return total seconds."""
    start = time.perf_counter()
    for _ in range(n_calls):
        func(*args)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Single-point secant method timing
# ---------------------------------------------------------------------------

class TestSecantPerformance:
    """Time the low-level secant root-finders on individual grid points."""

    def test_secant_temperature_throughput(self):
        """A single secant temperature solve should complete in < 0.5 ms."""
        h_i, sig_i, r_i, alpha_i, f_irr_i = 1e8, 1e2, 1e10, 0.1, 0.0

        # Warmup (JIT compilation)
        _secant_temperature(1e4, h_i, sig_i, r_i, alpha_i, M_STAR, f_irr_i)

        n_calls = 1000
        elapsed = _time_calls(
            _secant_temperature,
            (1e4, h_i, sig_i, r_i, alpha_i, M_STAR, f_irr_i),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 0.5, (
            f"Secant temperature solve took {per_call_ms:.3f} ms/call "
            f"(budget: 0.5 ms)"
        )

    def test_secant_scale_height_throughput(self):
        """A single secant scale-height solve should complete in < 0.5 ms."""
        sig_i, r_i, t_i = 1e2, 1e10, 1e5

        # Warmup
        _secant_scale_height(1e8, sig_i, r_i, t_i, M_STAR)

        n_calls = 1000
        elapsed = _time_calls(
            _secant_scale_height,
            (1e8, sig_i, r_i, t_i, M_STAR),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 0.5, (
            f"Secant scale-height solve took {per_call_ms:.3f} ms/call "
            f"(budget: 0.5 ms)"
        )


# ---------------------------------------------------------------------------
# Full grid solver timing (N=103, production size)
# ---------------------------------------------------------------------------

class TestSolveTemperaturePerformance:
    """Benchmark solve_temperature at production grid size."""

    N = 103  # production grid size used by simulation notebooks

    def _setup_disk(self):
        return _make_disk(self.N)

    def test_jit_temperature_under_budget(self):
        """JIT'd temperature solver for N=103 should finish in < 30 ms."""
        r, Sigma, H, T_c, alpha, F_irr = self._setup_disk()

        # Warmup
        solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)

        n_calls = 100
        elapsed = _time_calls(
            solve_temperature,
            (H, Sigma, r, T_c, alpha, M_STAR),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 30.0, (
            f"solve_temperature(N={self.N}) took {per_call_ms:.1f} ms/call "
            f"(budget: 30 ms)"
        )

    def test_jit_temperature_with_irradiation(self):
        """Temperature solver with F_irr should stay under 50 ms.

        Irradiation shifts the energy balance, causing more fallback guesses
        to be tried per cell, so this path is slower than the base case.
        """
        r, Sigma, H, T_c, alpha, F_irr = self._setup_disk()
        F_irr = np.full(self.N, 1e10)

        # Warmup
        solve_temperature(H, Sigma, r, T_c, alpha, M_STAR, F_irr=F_irr)

        n_calls = 100
        elapsed = _time_calls(
            solve_temperature,
            (H, Sigma, r, T_c, alpha, M_STAR, F_irr),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 50.0, (
            f"solve_temperature(N={self.N}, F_irr) took {per_call_ms:.1f} ms "
            f"(budget: 50 ms)"
        )

    def test_internal_jit_loop_under_budget(self):
        """The raw _solve_temperature_jit loop should be < 30 ms for N=103."""
        r, Sigma, H, T_c, alpha, F_irr = self._setup_disk()
        alpha_arr = np.full(self.N, 0.1)
        T_new = T_c.copy()

        # Warmup
        _solve_temperature_jit(H, Sigma, r, T_c, alpha_arr, M_STAR, F_irr, T_new)

        n_calls = 100
        elapsed = _time_calls(
            _solve_temperature_jit,
            (H, Sigma, r, T_c, alpha_arr, M_STAR, F_irr, T_new),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 30.0, (
            f"_solve_temperature_jit(N={self.N}) took {per_call_ms:.1f} ms "
            f"(budget: 30 ms)"
        )


class TestSolveScaleHeightPerformance:
    """Benchmark solve_scale_height at production grid size."""

    N = 103

    def _setup_disk(self):
        return _make_disk(self.N)

    def test_jit_scale_height_under_budget(self):
        """JIT'd scale-height solver for N=103 should finish in < 30 ms."""
        r, Sigma, H, T_c, _, _ = self._setup_disk()

        # Warmup
        solve_scale_height(H, Sigma, r, T_c, M_STAR)

        n_calls = 100
        elapsed = _time_calls(
            solve_scale_height,
            (H, Sigma, r, T_c, M_STAR),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 30.0, (
            f"solve_scale_height(N={self.N}) took {per_call_ms:.1f} ms/call "
            f"(budget: 30 ms)"
        )

    def test_internal_jit_loop_under_budget(self):
        """The raw _solve_scale_height_jit loop should be < 20 ms for N=103."""
        r, Sigma, H, T_c, _, _ = self._setup_disk()
        H_new = H.copy()

        # Warmup
        _solve_scale_height_jit(H, Sigma, r, T_c, M_STAR, H_new)

        n_calls = 100
        elapsed = _time_calls(
            _solve_scale_height_jit,
            (H, Sigma, r, T_c, M_STAR, H_new),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 20.0, (
            f"_solve_scale_height_jit(N={self.N}) took {per_call_ms:.1f} ms "
            f"(budget: 20 ms)"
        )


# ---------------------------------------------------------------------------
# Scipy fallback timing (custom opacity path)
# ---------------------------------------------------------------------------

class TestScipyFallbackPerformance:
    """Benchmark the scipy.optimize.newton fallback path (custom opacity)."""

    N = 103

    def test_scipy_fallback_under_budget(self):
        """Scipy fallback with custom opacity should finish in < 500 ms.

        This path is slower than JIT because scipy.optimize.newton has
        per-call Python overhead, but should still be bounded.
        """
        r, Sigma, H, T_c, alpha, _ = _make_disk(self.N)

        def constant_opacity(H, Sigma, T):
            return 0.4

        # Warmup
        solve_temperature(
            H, Sigma, r, T_c, alpha, M_STAR, opacity_func=constant_opacity,
        )

        n_calls = 10
        elapsed = _time_calls(
            solve_temperature,
            (H, Sigma, r, T_c, alpha, M_STAR, None, constant_opacity),
            n_calls,
        )
        per_call_ms = (elapsed / n_calls) * 1e3
        assert per_call_ms < 500.0, (
            f"solve_temperature(scipy, N={self.N}) took {per_call_ms:.1f} ms "
            f"(budget: 500 ms)"
        )


# ---------------------------------------------------------------------------
# Simulation-scale timing (100 timesteps)
# ---------------------------------------------------------------------------

class TestSimulationTimestepBudget:
    """Verify solver throughput at simulation scale.

    A simulation calls solve_temperature + solve_scale_height once per
    timestep, typically for 100k timesteps.  This test runs 100 timesteps
    and extrapolates to verify the total solver time stays within budget.
    """

    N = 103
    N_TIMESTEPS = 100

    def test_combined_solver_throughput(self):
        """100 timesteps of (T + H solve) should take < 3 s.

        Extrapolated: 100k timesteps < ~50 min total solver time.
        """
        r, Sigma, H, T_c, alpha, F_irr = _make_disk(self.N)

        # Warmup both solvers
        solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
        solve_scale_height(H, Sigma, r, T_c, M_STAR)

        start = time.perf_counter()
        for _ in range(self.N_TIMESTEPS):
            T_c = solve_temperature(H, Sigma, r, T_c, alpha, M_STAR)
            H = solve_scale_height(H, Sigma, r, T_c, M_STAR)
        elapsed = time.perf_counter() - start

        per_step_ms = (elapsed / self.N_TIMESTEPS) * 1e3
        extrapolated_100k_min = (per_step_ms * 1e5) / 6e4
        assert elapsed < 3.0, (
            f"{self.N_TIMESTEPS} timesteps took {elapsed:.2f} s "
            f"({per_step_ms:.1f} ms/step, "
            f"extrapolated 100k steps: {extrapolated_100k_min:.0f} min). "
            f"Budget: 3 s for {self.N_TIMESTEPS} steps."
        )
