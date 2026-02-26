"""Tests for blackhole.gpu.opacity — validates against CPU JIT'd opacity."""

import numpy as np

from blackhole import opacity as cpu_opacity
from blackhole.gpu import opacity as gpu_opacity


class TestKappaE:
    def test_scalar_match(self):
        rho, T = 1e-7, 1e5
        cpu = cpu_opacity.kappa_e(rho, T)
        gpu = gpu_opacity.kappa_e(np.float64(rho), np.float64(T))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)

    def test_array_match(self):
        rho = np.logspace(-10, -3, 50)
        T = np.logspace(3, 8, 50)
        cpu = np.array([cpu_opacity.kappa_e(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_e(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaK:
    def test_scalar_match(self):
        rho, T = 1e-7, 1e5
        cpu = cpu_opacity.kappa_K(rho, T)
        gpu = gpu_opacity.kappa_K(np.float64(rho), np.float64(T))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)

    def test_array_match(self):
        rho = np.logspace(-10, -3, 50)
        T = np.logspace(3, 8, 50)
        cpu = np.array([cpu_opacity.kappa_K(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_K(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaHminus:
    def test_scalar_match(self):
        rho, T = 1e-7, 1e4
        cpu = cpu_opacity.kappa_Hminus(rho, T)
        gpu = gpu_opacity.kappa_Hminus(np.float64(rho), np.float64(T))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)

    def test_array_match(self):
        rho = np.logspace(-10, -3, 50)
        T = np.logspace(3, 5, 50)
        cpu = np.array([cpu_opacity.kappa_Hminus(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_Hminus(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaMol:
    def test_match(self):
        cpu = cpu_opacity.kappa_mol()
        gpu = gpu_opacity.kappa_mol()
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaRad:
    def test_array_match(self):
        rho = np.logspace(-10, -3, 50)
        T = np.logspace(3, 8, 50)
        cpu = np.array([cpu_opacity.kappa_rad(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_rad(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaCond:
    def test_array_match(self):
        rho = np.logspace(-10, -3, 50)
        T = np.logspace(3, 8, 50)
        cpu = np.array([cpu_opacity.kappa_cond(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_cond(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaTot:
    def test_array_match(self):
        rho = np.logspace(-10, -3, 50)
        T = np.logspace(3, 8, 50)
        cpu = np.array([cpu_opacity.kappa_tot(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_tot(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestKappaBath:
    def test_array_match(self):
        H = np.logspace(7, 9, 50)
        Sigma = np.logspace(-2, 3, 50)
        T = np.logspace(3, 7, 50)
        cpu = np.array([cpu_opacity.kappa_bath(h, s, t)
                        for h, s, t in zip(H, Sigma, T)])
        gpu = gpu_opacity.kappa_bath(H, Sigma, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)


class TestLegacyOpacities:
    def test_kappa_ff(self):
        rho = np.logspace(-10, -3, 20)
        T = np.logspace(4, 8, 20)
        cpu = np.array([cpu_opacity.kappa_ff(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_ff(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)

    def test_kappa_simple(self):
        rho = np.logspace(-10, -3, 20)
        T = np.logspace(4, 8, 20)
        cpu = np.array([cpu_opacity.kappa_simple(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_simple(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)

    def test_kappa_bf(self):
        rho = np.logspace(-10, -3, 20)
        T = np.logspace(4, 8, 20)
        cpu = np.array([cpu_opacity.kappa_bf(r, t) for r, t in zip(rho, T)])
        gpu = gpu_opacity.kappa_bf(rho, T)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-12)
