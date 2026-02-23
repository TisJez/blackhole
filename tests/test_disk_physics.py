"""Tests for blackhole.disk_physics."""

import numpy as np

from blackhole.constants import M_sun
from blackhole.disk_physics import (
    Marr,
    R_func,
    S_factor,
    Sigma_from_S,
    X_func,
    density,
    kinematic_viscosity,
    omega,
    pressure,
    pressure_2,
    scale_height,
)

M_STAR = 10 * M_sun


class TestCoordinateTransforms:
    def test_X_R_roundtrip(self):
        r = np.array([1e10, 1e12, 1e14])
        assert np.allclose(R_func(X_func(r)), r)

    def test_X_func_positive(self):
        assert X_func(1e10) > 0

    def test_array_input(self):
        r = np.logspace(8, 14, 20)
        result = X_func(r)
        assert result.shape == (20,)


class TestOmega:
    def test_positivity(self):
        assert omega(1e10, M_STAR) > 0

    def test_decreases_with_r(self):
        assert omega(1e10, M_STAR) > omega(1e12, M_STAR)

    def test_array_input(self):
        r = np.array([1e10, 1e11, 1e12])
        result = omega(r, M_STAR)
        assert result.shape == (3,)


class TestKinematicViscosity:
    def test_positivity(self):
        val = kinematic_viscosity(1e8, 1e10, 0.1, M_STAR)
        assert val > 0

    def test_proportional_to_alpha(self):
        nu1 = kinematic_viscosity(1e8, 1e10, 0.1, M_STAR)
        nu2 = kinematic_viscosity(1e8, 1e10, 0.2, M_STAR)
        assert abs(nu2 / nu1 - 2.0) < 1e-10


class TestDensity:
    def test_positivity(self):
        assert density(1e8, 1e2) > 0

    def test_formula(self):
        H, Sigma = 1e8, 1e2
        assert abs(density(H, Sigma) - Sigma / (2 * H)) < 1e-30


class TestPressure:
    def test_positivity(self):
        assert pressure(1e8, 1e2, 1e4) > 0

    def test_pressure_2_positivity(self):
        assert pressure_2(1e8, 1e2, 1e10, M_STAR) > 0


class TestScaleHeight:
    def test_positivity(self):
        p = pressure(1e8, 1e2, 1e4)
        val = scale_height(1e2, p, 1e10, M_STAR)
        assert val > 0


class TestSurfaceDensityHelpers:
    def test_S_factor(self):
        X = np.array([1.0, 2.0, 3.0])
        Sigma = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_equal(S_factor(X, Sigma), X * Sigma)

    def test_Sigma_from_S_roundtrip(self):
        X = np.array([1.0, 2.0, 3.0])
        Sigma = np.array([10.0, 20.0, 30.0])
        S = S_factor(X, Sigma)
        np.testing.assert_allclose(Sigma_from_S(S, X), Sigma)

    def test_Marr_positivity(self):
        X = np.array([1e6, 2e6, 3e6])
        Sigma = np.array([10.0, 20.0, 30.0])
        dX = 1e6
        result = Marr(X, Sigma, dX)
        assert np.all(result > 0)
