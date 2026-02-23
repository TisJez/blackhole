"""Shared test fixtures for the blackhole package."""

import numpy as np
import pytest

from blackhole.constants import M_sun


@pytest.fixture
def ten_solar_mass():
    """10 solar-mass black hole."""
    return 10.0 * M_sun


@pytest.fixture
def rho_array():
    """Log-spaced density array (g/cm^3)."""
    return np.logspace(-12, -4, 50)


@pytest.fixture
def T_array():
    """Log-spaced temperature array (K)."""
    return np.logspace(2, 8, 50)
