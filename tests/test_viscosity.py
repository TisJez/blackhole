"""Tests for blackhole.viscosity."""

import numpy as np

from blackhole.constants import ALPHA_COLD, ALPHA_HOT
from blackhole.viscosity import alpha_visc


class TestAlphaVisc:
    def test_cold_limit(self):
        """At low T, alpha -> alpha_cold."""
        val = alpha_visc(1e3)
        assert abs(val - ALPHA_COLD) / ALPHA_COLD < 0.01

    def test_hot_limit(self):
        """At high T, alpha -> alpha_hot."""
        val = alpha_visc(1e6)
        assert abs(val - ALPHA_HOT) / ALPHA_HOT < 0.01

    def test_bounded(self):
        T = np.logspace(2, 8, 100)
        vals = alpha_visc(T)
        assert np.all(vals >= ALPHA_COLD * 0.99)
        assert np.all(vals <= ALPHA_HOT * 1.01)

    def test_monotonically_increasing(self):
        T = np.logspace(3, 6, 200)
        vals = alpha_visc(T)
        assert np.all(np.diff(vals) >= 0)

    def test_array_vectorization(self):
        T = np.array([1e3, 1e4, 1e5, 1e6])
        result = alpha_visc(T)
        assert result.shape == (4,)

    def test_transition_region(self):
        """At the critical T (~25000 K), alpha should be roughly midway (in log)."""
        val = alpha_visc(2.5e4)
        log_mid = 0.5 * (np.log(ALPHA_COLD) + np.log(ALPHA_HOT))
        assert abs(np.log(val) - log_mid) / abs(log_mid) < 0.1

    def test_custom_alpha_values(self):
        val = alpha_visc(1e3, alpha_cold=0.02, alpha_hot=0.3)
        assert abs(val - 0.02) / 0.02 < 0.01
