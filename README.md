# Black Hole X-ray Outburst

Python project for modelling X-ray outbursts from stellar compact objects (black holes and white dwarfs). Uses numerical methods to solve accretion disk equations and visualize outburst behavior.

Written for the Durham University 2024 MPhys Master's Project by Jezreel Penuliar: "Modelling X-Ray Outbursts in Black Hole Accretion Disks" (Supervisor: Prof. Chris Done)

## Project Structure

```
blackhole/
├── pyproject.toml
├── src/
│   ├── blackhole/           # Python package (installable via pip)
│   │   ├── __init__.py      # gpu_jit decorator, get_xp() CuPy/NumPy dispatch
│   │   ├── constants.py     # CGS physical constants
│   │   ├── opacity.py       # Opacity regimes (electron, Kramers, H-, conduction)
│   │   ├── viscosity.py     # Temperature-dependent alpha viscosity
│   │   ├── steady_state.py  # Shakura-Sunyaev steady-state disk structure
│   │   ├── disk_physics.py  # Core disk physics (pressure, density, scale height)
│   │   ├── irradiation.py   # Irradiation feedback (Flux_irr, Sigma_max/min)
│   │   ├── evolution.py     # Surface density time-stepping & mass addition
│   │   ├── solvers.py       # Newton solvers for temperature & scale height
│   │   ├── luminosity.py    # Radiative luminosity & effective temperature
│   │   └── cr_solvers.py    # Critical-regime (CR) steady-state disk solvers
│   ├── notebooks/           # Jupyter notebooks (simulations & analysis)
│   ├── graphs/              # Output plots from notebooks
│   └── data/                # CSV simulation data (git-ignored)
├── tests/                   # Unit tests (172 tests)
└── .github/workflows/       # CI/CD pipelines
```

## Getting Started

See [SETUP.md](SETUP.md) for installation instructions, CUDA/GPU setup, and running notebooks.

### Quick install

```bash
git clone https://github.com/TisJez/blackhole.git
cd blackhole

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -e ".[dev]"          # CPU-only (dev/CI)
pip install -e ".[gpu,dev]"      # With GPU (CuPy, Numba CUDA)
```

### Verify

```bash
pytest                           # 172 tests
ruff check .                     # Lint
```

## Package Modules

All modules support transparent CuPy/NumPy array dispatch via `get_xp()`. Pass CuPy arrays in to get GPU-accelerated computation; pass NumPy arrays for CPU.

| Module | Description |
|--------|-------------|
| `constants` | CGS physical constants (G, c, k_B, sigma_SB, M_sun, etc.) |
| `opacity` | Multiple opacity regimes (electron scattering, Kramers, H-minus, molecular, conduction, bound-free) |
| `viscosity` | Temperature-dependent alpha viscosity with smooth hot/cold transition |
| `steady_state` | Shakura-Sunyaev steady-state disk structure (inner/middle/outer regions with boundary functions) |
| `disk_physics` | Core disk physics (pressure, density, scale height, coordinate transforms X/R, Keplerian angular velocity) |
| `irradiation` | X-ray irradiation feedback: flux, epsilon parameter, critical surface densities (Sigma_max/min), critical temperatures (T_c_max/min), irradiation-modified alpha viscosity |
| `evolution` | Surface density time-stepping via viscous diffusion, mass addition at transfer radius, tidal torques, disk evaporation |
| `solvers` | Newton-method solvers for energy balance (midplane temperature) and hydrostatic equilibrium (scale height) |
| `luminosity` | Radiative luminosity diagnostics (total and per-annulus) and effective temperature profiles |
| `cr_solvers` | Critical-regime steady-state disk structure: radiation pressure, Keplerian omega, scale height, coupled rho-T solver via Levenberg-Marquardt |

## Notebooks

The notebooks are organized into three categories: steady-state analysis, time-dependent simulations, and post-processing/visualization.

### Steady-State Analysis

These notebooks explore the equilibrium disk structure and the physics of individual components (opacity, viscosity, S-curves). They run quickly and do not require pre-computed data.

| Notebook | Description |
|----------|-------------|
| `opacity_constants` | Opacity regime calculations: electron scattering, bound-free, H-minus, molecular, and conduction opacities plotted as functions of temperature and density |
| `viscosity_temperature_dependence` | Temperature-dependent alpha viscosity model with irradiation effects: plots the smooth alpha(T) transition and irradiation-modified critical temperatures |
| `steady_state_disk_structure` | Full Shakura-Sunyaev steady-state disk structure with 3 regions (inner/middle/outer): solves for temperature, density, scale height, and pressure as functions of radius |
| `steady_state_disk_subplots` | Extended steady-state disk variable plots: multi-panel visualizations of radial disk profiles using the `Variable_plot` helper |
| `opacity_model_comparison` | Comparison of standard SS73 and Critical-Regime (CR) steady-state models: T-Sigma S-curves, temperature vs radius, scale height vs radius, with fixed and variable alpha |

### Time-Dependent Simulations

Each notebook runs a full disk instability simulation from a fresh initial disk. They produce CSV history files in `src/data/` that can be visualized by the plotting notebooks. All simulations use the same 4-cell structure: (1) parameters & grid setup, (2) timestep config, (3) fresh disk initialization, (4) main simulation loop with CSV I/O.

| Notebook | Object | M_star | R_in (cm) | R_out (cm) | M_dot (g/s) | Irradiation | Evaporation | CSV suffix | Timesteps |
|----------|--------|--------|-----------|------------|-------------|-------------|-------------|------------|-----------|
| `wd_timedep_simulation` | White dwarf | 1 M_sun | 5e8 | 8e10 | 5e16 | No | No | `_wd` | 15,001 |
| `bh_timedep_simulation` | BH base | 1 M_sun | 5e8 | 4.2e11 | 3e17 | No | No | `_bh` | 25,001 |
| `bh_noeffects_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | No | No | `_bhne` | 25,001 |
| `bh_irradiation_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | Yes | No | `_ir` | 25,001 |
| `bh_evaporation_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | No | Yes | `_ev` | 25,001 |
| `bh_iradevap_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | Yes | Yes | `_irev` | 25,001 |
| `sgr_a_timedep_simulation` | Sgr A* SMBH | 4.3e6 M_sun | 4e12 | 2e15 | 1e17 | Yes | Yes | _(none)_ | 25,001 |

Each simulation saves 6 CSV files to `src/data/`:
- `Sigma_history_bath_array{suffix}.csv` - Surface density snapshots
- `Temp_history_bath_array{suffix}.csv` - Midplane temperature snapshots
- `H_history_bath_array{suffix}.csv` - Scale height snapshots
- `alpha_history_bath_array{suffix}.csv` - Alpha viscosity snapshots
- `t_history_bath_array{suffix}.csv` - Simulation time at each snapshot
- `Sigma_transfer_history_bath_array{suffix}.csv` - Inner-edge surface density history

### Visualization & Post-Processing

These notebooks load pre-computed CSV data and produce publication-quality plots. Run the relevant simulation notebook first to generate the data.

| Notebook | Description | Requires data from |
|----------|-------------|--------------------|
| `outburst_lightcurves` | Multi-model comparison: mass accretion rate, luminosity, outer radius vs time for WD, BH, and SMBH models. Includes animations and outburst period analysis. | All simulation notebooks |
| `sgr_a_outburst_plots` | Sgr A* focused: 3-panel plot of mass accretion rate, luminosity, and outer radius vs time (years) | `sgr_a_timedep_simulation` |

## Physics Overview

The codebase implements the Disk Instability Model (DIM) for accretion disks around compact objects:

1. **Opacity** - Multiple opacity regimes determine how efficiently the disk radiates: electron scattering (high-T), Kramers/bound-free (intermediate), H-minus (low-T), molecular, and conduction
2. **Alpha viscosity** - The Shakura-Sunyaev alpha prescription with a smooth temperature-dependent transition between cold (alpha ~ 0.04) and hot (alpha ~ 0.2) states, creating the thermal-viscous instability
3. **Time-dependent evolution** - Viscous diffusion of surface density on a 1D radial grid (X = 2*sqrt(R) coordinates), with mass addition at the transfer radius, tidal torques from the companion, and optional disk evaporation
4. **Irradiation** - X-ray heating from the inner disk modifies the critical temperatures for the hot/cold transition, stabilizing the outer disk
5. **Evaporation** - Removal of mass from the inner disk due to coronal evaporation above the Eddington rate

Physical units are CGS throughout the codebase.

## Technology Stack

- **Python 3.12+**
- **NumPy/SciPy** - Scientific computing and numerical methods
- **CuPy** - GPU array operations (optional, transparent dispatch via `get_xp()`)
- **Numba/CUDA** - GPU JIT compilation for time-dependent simulations (optional)
- **Matplotlib** - Visualization
- **Pandas** - Data I/O for simulation histories

## Git Workflow

Branch strategy: `feature` → `dev` → `main`

- CI runs on pull requests to `dev` and `main`
- Deploy runs on push to `main`
