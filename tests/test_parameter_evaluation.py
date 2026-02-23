"""Tests for blackhole.parameter_evaluation."""

from blackhole.constants import M_sun
from blackhole.parameter_evaluation import EvaluationResult, ParameterEvaluation


class TestEvaluationResult:
    def test_valid_when_both_ok(self):
        r = EvaluationResult(
            mass_deposition_ok=True,
            instability_ok=True,
            valid=True,
            dt_cfl=1.0, dt_used=1.0, dt_max=10.0,
            deposition_margin=10.0,
            sigma_ss=100.0, sigma_max=50.0,
            instability_ratio=2.0, t_viscous=1e6,
        )
        assert r.valid is True

    def test_invalid_when_deposition_fails(self):
        r = EvaluationResult(
            mass_deposition_ok=False,
            instability_ok=True,
            valid=False,
            dt_cfl=1.0, dt_used=100.0, dt_max=10.0,
            deposition_margin=0.1,
            sigma_ss=100.0, sigma_max=50.0,
            instability_ratio=2.0, t_viscous=1e6,
        )
        assert r.valid is False
        assert r.mass_deposition_ok is False

    def test_invalid_when_instability_fails(self):
        r = EvaluationResult(
            mass_deposition_ok=True,
            instability_ok=False,
            valid=False,
            dt_cfl=1.0, dt_used=1.0, dt_max=10.0,
            deposition_margin=10.0,
            sigma_ss=1.0, sigma_max=50.0,
            instability_ratio=0.02, t_viscous=1e6,
        )
        assert r.valid is False
        assert r.instability_ok is False

    def test_valid_equals_both_and(self):
        for dep_ok in [True, False]:
            for inst_ok in [True, False]:
                r = EvaluationResult(
                    mass_deposition_ok=dep_ok,
                    instability_ok=inst_ok,
                    valid=dep_ok and inst_ok,
                    dt_cfl=1.0, dt_used=1.0, dt_max=10.0,
                    deposition_margin=10.0,
                    sigma_ss=1.0, sigma_max=1.0,
                    instability_ratio=1.0, t_viscous=1e6,
                )
                assert r.valid == (dep_ok and inst_ok)


class TestParameterEvaluation:
    """Test with actual notebook parameters."""

    # --- Known-good configurations ---

    def test_wd_valid(self):
        """WD simulation: M_star=1 M_sun, M_dot=5e16, dt_mult=10."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=10, dt_floor=200)
        assert result.valid is True
        assert result.mass_deposition_ok is True
        assert result.instability_ok is True
        assert result.instability_ratio > 1.0
        assert result.deposition_margin > 10

    def test_bh_base_valid(self):
        """BH base: M_star=9 M_sun, M_dot=3e17, dt_mult=30."""
        pe = ParameterEvaluation(
            M_star=9 * M_sun, R_1=5e8, R_K=2.2e11, R_N=4.2e11,
            M_dot=3e17,
        )
        result = pe.evaluate(dt_mult=30, dt_floor=200)
        assert result.valid is True
        assert result.mass_deposition_ok is True
        assert result.instability_ok is True
        assert result.instability_ratio > 0.5

    def test_bh_noeffects_valid(self):
        """BH noeffects: M_star=9 M_sun, M_dot=1e17, dt_mult=30."""
        pe = ParameterEvaluation(
            M_star=9 * M_sun, R_1=5e8, R_K=2.2e11, R_N=4.2e11,
            M_dot=1e17,
        )
        result = pe.evaluate(dt_mult=30, dt_floor=200)
        assert result.valid is True
        assert result.instability_ratio > 0.3

    def test_sgr_a_valid(self):
        """Sgr A*: M_star=4.3e6 M_sun, M_dot=1e22, alpha_cold=0.02."""
        pe = ParameterEvaluation(
            M_star=4.3e6 * M_sun, R_1=4e12, R_K=1e15, R_N=2e15,
            M_dot=1e22, alpha_cold=0.02,
        )
        result = pe.evaluate(dt_mult=300, dt_floor=1e5, dt_cap=2e9)
        assert result.valid is True
        assert result.mass_deposition_ok is True
        assert result.instability_ok is True
        assert result.instability_ratio > 0.5

    # --- Known-broken configurations ---

    def test_bh_wrong_mass_instability_fails(self):
        """BH with M_star=1 M_sun (wrong) → instability_ok=False."""
        pe = ParameterEvaluation(
            M_star=1 * M_sun, R_1=5e8, R_K=2.2e11, R_N=4.2e11,
            M_dot=3e17,
        )
        result = pe.evaluate(dt_mult=30, dt_floor=200)
        assert result.instability_ok is False
        assert result.valid is False
        assert result.instability_ratio < 0.3

    def test_sgr_a_low_mdot_instability_fails(self):
        """Sgr A* with M_dot=1e17 (too low) → instability_ok=False."""
        pe = ParameterEvaluation(
            M_star=4.3e6 * M_sun, R_1=4e12, R_K=1e15, R_N=2e15,
            M_dot=1e17, alpha_cold=0.02,
        )
        result = pe.evaluate(dt_mult=300, dt_floor=1e5, dt_cap=2e9)
        assert result.instability_ok is False
        assert result.valid is False

    def test_excessive_dt_mult_deposition_fails(self):
        """Any sim with huge dt_mult → mass_deposition_ok=False."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=1e6)
        assert result.mass_deposition_ok is False
        assert result.valid is False
        assert result.deposition_margin < 1.0

    # --- Diagnostic value tests ---

    def test_deposition_margin_matches_ratio(self):
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=10, dt_floor=200)
        assert abs(result.deposition_margin - result.dt_max / result.dt_used) < 1e-10

    def test_instability_ratio_matches(self):
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=10, dt_floor=200)
        assert abs(result.instability_ratio - result.sigma_ss / result.sigma_max) < 1e-10

    def test_dt_floor_applied(self):
        """dt_floor raises dt_used when CFL*mult is too small."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=0.001, dt_floor=1e6)
        assert result.dt_used == 1e6

    def test_dt_cap_applied(self):
        """dt_cap limits dt_used when CFL*mult is too large."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=1e6, dt_cap=100.0)
        assert result.dt_used == 100.0

    def test_positive_diagnostics(self):
        """All diagnostic values should be positive."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        result = pe.evaluate(dt_mult=10)
        assert result.dt_cfl > 0
        assert result.dt_used > 0
        assert result.dt_max > 0
        assert result.sigma_ss > 0
        assert result.sigma_max > 0
        assert result.t_viscous > 0

    def test_multiple_evaluations_same_system(self):
        """evaluate() can be called multiple times on the same system."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16,
        )
        r1 = pe.evaluate(dt_mult=10)
        r2 = pe.evaluate(dt_mult=1e6)
        # Same system → same sigma values
        assert r1.sigma_ss == r2.sigma_ss
        assert r1.sigma_max == r2.sigma_max
        # Different dt → different deposition results
        assert r1.mass_deposition_ok is True
        assert r2.mass_deposition_ok is False

    def test_custom_instability_threshold(self):
        """Custom threshold changes the instability check."""
        pe = ParameterEvaluation(
            M_star=9 * M_sun, R_1=5e8, R_K=2.2e11, R_N=4.2e11,
            M_dot=1e17, instability_threshold=0.3,
        )
        result_low = pe.evaluate(dt_mult=30, dt_floor=200)
        assert result_low.instability_ok is True

        pe_strict = ParameterEvaluation(
            M_star=9 * M_sun, R_1=5e8, R_K=2.2e11, R_N=4.2e11,
            M_dot=1e17, instability_threshold=0.9,
        )
        result_strict = pe_strict.evaluate(dt_mult=30, dt_floor=200)
        # BH noeffects has ratio ~0.45, so strict threshold of 0.9 fails
        assert result_strict.instability_ok is False

    def test_grid_construction_matches_notebook(self):
        """Grid size = N + N_n (matching notebook convention)."""
        pe = ParameterEvaluation(
            M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
            M_dot=5e16, N=100, N_n=3,
        )
        assert pe.N == 103
        assert len(pe.X) == 103
        assert len(pe.nu_array) == 103
