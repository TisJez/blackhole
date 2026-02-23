# Black Hole X-ray Outburst

Python project for modelling X-ray outbursts from stellar compact objects (black holes and white dwarfs). Uses numerical methods to solve accretion disk equations and visualize outburst behavior.
Written for the Durham University 2024 MPhys Master's Project by Jezreel Penuliar: "Modelling X-Ray Outbursts in Black Hole Accretion Disks" (Supervisor: Prof. Chris Done)

## Project Structure

```
blackhole/
├── pyproject.toml
├── src/
│   ├── blackhole/          # Python package (installable via pip)
│   │   ├── __init__.py     # gpu_jit decorator (CUDA → CPU numba → passthrough)
│   │   ├── constants.py    # CGS physical constants
│   │   ├── opacity.py      # Opacity regimes (electron, Kramers, H-, conduction)
│   │   ├── viscosity.py    # Temperature-dependent alpha viscosity
│   │   ├── steady_state.py # Shakura-Sunyaev steady-state disk structure
│   │   ├── disk_physics.py # Core disk physics (pressure, density, scale height)
│   │   ├── irradiation.py  # Irradiation feedback (Flux_irr, Sigma_max/min)
│   │   ├── evolution.py    # Surface density time-stepping & mass addition
│   │   ├── solvers.py      # Newton solvers for temperature & scale height
│   │   └── luminosity.py   # Radiative luminosity & effective temperature
│   ├── notebooks/          # Jupyter notebooks (simulations & analysis)
│   ├── graphs/             # Output plots from notebooks
│   └── data/               # CSV simulation data (git-ignored)
├── tests/                  # Unit tests (119 tests)
└── .github/workflows/      # CI/CD pipelines
```

## Getting Started

See [SETUP.md](SETUP.md) for installation instructions, CUDA/GPU setup, and running notebooks.

## Package Modules

| Module | Description |
|--------|-------------|
| `constants` | CGS physical constants — single source of truth |
| `opacity` | Multiple opacity regimes (electron scattering, Kramers, H-minus, molecular, conduction) |
| `viscosity` | Temperature-dependent alpha viscosity with hot/cold disk states |
| `steady_state` | Shakura-Sunyaev steady-state disk structure (3 regions) |
| `disk_physics` | Core disk physics (pressure, density, scale height, coordinate transforms) |
| `irradiation` | Irradiation feedback, critical surface densities and temperatures (DIM S-curve) |
| `evolution` | Surface density time-stepping, mass addition, tidal torques, disk evaporation |
| `solvers` | Newton-method solvers for energy balance (temperature) and hydrostatic equilibrium (scale height) |
| `luminosity` | Radiative luminosity diagnostics and effective temperature profiles |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `opacity_formulae.ipynb` | Opacity regime calculations (electron scattering, bound-free, H-minus, molecular, conduction) |
| `diskequations_SS_bath_params.ipynb` | Steady-state Shakura-Sunyaev disk equations |
| `diskeqs_ss_2.ipynb` | Extended steady-state disk equations |
| `alpha-t dependence.ipynb` | Temperature-dependent alpha viscosity model |
| `bh_model_comparison_graphs.ipynb` | Comparison graphs between models |
| `GPU_timedep_newopacity_sagittarius_a_alpha_2.ipynb` | GPU-accelerated time-dependent simulation (Sgr A*) |
| `Outburst_graphs.ipynb` | Outburst visualization from pre-computed simulation data |

## Technology Stack

- **Python 3.12+**
- **NumPy/SciPy** - Scientific computing and numerical methods
- **Numba/CUDA** - GPU acceleration for time-dependent simulations
- **CuPy** - GPU array operations
- **Matplotlib** - Visualization
- **Pandas** - Data processing

## Git Workflow

Branch strategy: `feature` → `dev` → `main`
