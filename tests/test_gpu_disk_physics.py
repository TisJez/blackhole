"""Tests for blackhole.gpu.disk_physics — validates against CPU JIT'd physics."""

import numpy as np

from blackhole import disk_physics as cpu_dp
from blackhole.constants import M_sun
from blackhole.gpu import disk_physics as gpu_dp

M_STAR = 10 * M_sun


class TestXFunc:
    def test_array_match(self):
        r = np.logspace(8, 12, 50)
        cpu = cpu_dp.X_func(r)
        gpu = gpu_dp.X_func(r)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestRFunc:
    def test_array_match(self):
        x = np.linspace(1e3, 1e6, 50)
        cpu = cpu_dp.R_func(x)
        gpu = gpu_dp.R_func(x)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestRoundTrip:
    def test_X_R_roundtrip(self):
        r = np.logspace(8, 12, 50)
        np.testing.assert_allclose(gpu_dp.R_func(gpu_dp.X_func(r)), r, rtol=1e-12)


class TestOmega:
    def test_array_match(self):
        R = np.logspace(8, 12, 50)
        cpu = cpu_dp.omega(R, M_STAR)
        gpu = gpu_dp.omega(R, M_STAR)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKinematicViscosity:
    def test_array_match(self):
        H = np.logspace(7, 9, 50)
        R = np.logspace(9, 12, 50)
        alpha = np.full(50, 0.1)
        cpu = cpu_dp.kinematic_viscosity(H, R, alpha, M_STAR)
        gpu = gpu_dp.kinematic_viscosity(H, R, alpha, M_STAR)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestDensity:
    def test_array_match(self):
        H = np.logspace(7, 9, 50)
        Sigma = np.logspace(-2, 3, 50)
        cpu = cpu_dp.density(H, Sigma)
        gpu = gpu_dp.density(H, Sigma)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestPressure:
    def test_array_match(self):
        H = np.logspace(7, 9, 50)
        Sigma = np.logspace(-2, 3, 50)
        T = np.logspace(3, 7, 50)
        cpu = cpu_dp.pressure(H, Sigma, T)
        gpu = gpu_dp.pressure(H, Sigma, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestPressure2:
    def test_array_match(self):
        H = np.logspace(7, 9, 50)
        Sigma = np.logspace(-2, 3, 50)
        R = np.logspace(9, 12, 50)
        cpu = cpu_dp.pressure_2(H, Sigma, R, M_STAR)
        gpu = gpu_dp.pressure_2(H, Sigma, R, M_STAR)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestSFactor:
    def test_array_match(self):
        X = np.linspace(1e3, 1e5, 50)
        Sigma = np.logspace(-2, 3, 50)
        cpu = cpu_dp.S_factor(X, Sigma)
        gpu = gpu_dp.S_factor(X, Sigma)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestSigmaFromS:
    def test_array_match(self):
        X = np.linspace(1e3, 1e5, 50)
        S = np.logspace(1, 8, 50)
        cpu = cpu_dp.Sigma_from_S(S, X)
        gpu = gpu_dp.Sigma_from_S(S, X)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestMarr:
    def test_array_match(self):
        X = np.linspace(1e3, 1e5, 50)
        Sigma = np.logspace(-2, 3, 50)
        dX = X[1] - X[0]
        cpu = cpu_dp.Marr(X, Sigma, dX)
        gpu = gpu_dp.Marr(X, Sigma, dX)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)
