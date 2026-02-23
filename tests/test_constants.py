"""Tests for blackhole.constants."""

from blackhole import constants as C


def test_fundamental_constants_positive():
    for val in [C.G, C.c, C.k_B, C.m_p, C.sigma_SB, C.a_rad, C.r_gas, C.M_sun]:
        assert val > 0


def test_speed_of_light_order_of_magnitude():
    assert 2e10 < C.c < 4e10


def test_solar_mass_order_of_magnitude():
    assert 1e33 < C.M_sun < 3e33


def test_alpha_ordering():
    assert 0 < C.ALPHA_COLD < C.ALPHA_HOT < 1


def test_hydrogen_fraction():
    assert 0 < C.X_HYDROGEN < 1
    assert abs(C.X_HYDROGEN + C.Z_METALS - 1.0) < 1e-10
