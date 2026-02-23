"""CGS physical constants — single source of truth.

Canonical values taken from the GPU_timedep notebook (most precise).
All units are CGS (g, cm, s, K, erg).
"""

# Fundamental constants
G = 6.67430e-8              # Gravitational constant (cm^3 g^-1 s^-2)
c = 2.99792458e10           # Speed of light (cm/s)
k_B = 1.380649e-16          # Boltzmann constant (erg/K)
m_p = 1.6726219e-24         # Proton mass (g)
sigma_SB = 5.670374419e-5   # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
a_rad = 7.5657e-15          # Radiation density constant (erg cm^-3 K^-4)
r_gas = 8.31446261815324e7  # Molar gas constant (erg mol^-1 K^-1)

# Astrophysical constants
M_sun = 1.989e33            # Solar mass (g)

# Default disk parameters
mu = 0.5                    # Mean molecular weight (ionised pure hydrogen)
ALPHA_COLD = 0.04           # Cold-state alpha viscosity (standard DIM)
ALPHA_HOT = 0.2             # Hot-state alpha viscosity

# Opacity composition defaults
X_HYDROGEN = 0.96           # Hydrogen mass fraction
Z_METALS = 0.04             # Metal mass fraction (1 - X)
Z_STAR = 5e-10              # Effective Z for conductive opacity

# Critical accretion rate
M_DOT_CRIT = 1.5e17         # Critical mass accretion rate (g/s)
