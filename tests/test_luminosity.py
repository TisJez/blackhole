"""Tests for blackhole.luminosity."""

import numpy as np

from blackhole.constants import M_sun
from blackhole.luminosity import L_rad, L_rad_array, T_eff

M_STAR = 10 * M_sun


class TestLRad:
    def test_positive(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        dr = r[1] - r[0]
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = L_rad(Sigma, H, alpha, r, dr, M_STAR)
        assert result > 0

    def test_returns_scalar(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        dr = r[1] - r[0]
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = L_rad(Sigma, H, alpha, r, dr, M_STAR)
        assert np.isscalar(result)

    def test_increases_with_sigma(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        dr = r[1] - r[0]
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        L1 = L_rad(np.full(N, 1e2), H, alpha, r, dr, M_STAR)
        L2 = L_rad(np.full(N, 2e2), H, alpha, r, dr, M_STAR)
        assert L2 > L1


class TestLRadArray:
    def test_shape_matches_input(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        dr = r[1] - r[0]
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = L_rad_array(Sigma, H, alpha, r, dr, M_STAR)
        assert result.shape == (N,)

    def test_positive(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        dr = r[1] - r[0]
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = L_rad_array(Sigma, H, alpha, r, dr, M_STAR)
        assert np.all(result > 0)

    def test_sum_equals_L_rad(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        dr = r[1] - r[0]
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        total = L_rad(Sigma, H, alpha, r, dr, M_STAR)
        arr = L_rad_array(Sigma, H, alpha, r, dr, M_STAR)
        assert abs(np.sum(arr) - total) / total < 1e-10


class TestTEff:
    def test_positive_and_finite(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = T_eff(Sigma, H, alpha, r, M_STAR)
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_shape_matches_input(self):
        N = 50
        r = np.linspace(1e9, 1e12, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = T_eff(Sigma, H, alpha, r, M_STAR)
        assert result.shape == (N,)

    def test_zero_at_inner_edge(self):
        """T_eff should be zero at r = r_in (1 - sqrt(r_in/r) = 0)."""
        N = 50
        r = np.linspace(1e9, 1e12, N)
        Sigma = np.full(N, 1e2)
        H = np.full(N, 1e8)
        alpha = np.full(N, 0.1)
        result = T_eff(Sigma, H, alpha, r, M_STAR)
        assert result[0] == 0.0
