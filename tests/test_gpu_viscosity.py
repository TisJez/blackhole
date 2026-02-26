"""Tests for blackhole.gpu.viscosity — validates against CPU JIT'd viscosity."""

import numpy as np

from blackhole import viscosity as cpu_visc
from blackhole.gpu import viscosity as gpu_visc


class TestAlphaVisc:
    def test_cold_regime(self):
        """In the cold regime (T << T_crit), alpha should be close to alpha_cold."""
        T = np.array([1e3])
        result = gpu_visc.alpha_visc(T)
        np.testing.assert_allclose(result, 0.04, rtol=1e-3)

    def test_hot_regime(self):
        """In the hot regime (T >> T_crit), alpha should be close to alpha_hot."""
        T = np.array([1e6])
        result = gpu_visc.alpha_visc(T)
        np.testing.assert_allclose(result, 0.2, rtol=1e-3)

    def test_array_match(self):
        T = np.logspace(2, 8, 100)
        cpu = np.array([cpu_visc.alpha_visc(t) for t in T])
        gpu = gpu_visc.alpha_visc(T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)

    def test_transition_monotonic(self):
        """Alpha should increase monotonically with temperature."""
        T = np.logspace(3, 6, 100)
        result = gpu_visc.alpha_visc(T)
        assert np.all(np.diff(result) >= 0)

    def test_custom_params(self):
        T = np.logspace(3, 6, 50)
        cpu = np.array([cpu_visc.alpha_visc(t, alpha_cold=0.01, alpha_hot=0.3)
                        for t in T])
        gpu = gpu_visc.alpha_visc(T, alpha_cold=0.01, alpha_hot=0.3)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)
