# Black Hole X-ray Outburst

Python project for modelling X-ray outbursts from stellar compact objects (black holes and white dwarfs). Uses numerical methods to solve accretion disk equations and visualize outburst behavior. Includes a **parameter evaluation tool** that validates simulation parameters before committing to multi-hour runs.

Written for the Durham University 2024 MPhys Master's Project by Jezreel Penuliar: "Modelling X-Ray Outbursts in Black Hole Accretion Disks" (Supervisor: Prof. Chris Done)

## Project Structure

```
blackhole/
├── pyproject.toml
├── src/
│   ├── blackhole/           # Python package (installable via pip)
│   │   ├── __init__.py      # cpu_jit / gpu_jit decorators (numba JIT with fallback)
│   │   ├── constants.py     # CGS physical constants
│   │   ├── opacity.py       # Opacity regimes (electron, Kramers, H-, conduction) — JIT'd
│   │   ├── viscosity.py     # Temperature-dependent alpha viscosity — JIT'd
│   │   ├── steady_state.py  # Shakura-Sunyaev steady-state disk structure — JIT'd
│   │   ├── disk_physics.py  # Core disk physics (pressure, density, scale height) — JIT'd
│   │   ├── irradiation.py   # Irradiation feedback (Flux_irr, Sigma_max/min)
│   │   ├── evolution.py     # Surface density time-stepping & mass addition
│   │   ├── solvers.py       # Newton solvers for temperature & scale height (JIT'd root functions)
│   │   ├── luminosity.py    # Radiative luminosity & effective temperature
│   │   ├── cr_solvers.py    # Critical-regime (CR) steady-state disk solvers
│   │   └── parameter_evaluation.py  # Pre-flight parameter checks for DIM simulations
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

pip install -e ".[dev]"          # CPU-only with numba JIT (dev/CI)
pip install -e ".[gpu,dev]"      # With GPU (CuPy, Numba CUDA)
```

### Verify

```bash
pytest                           # 172 tests
ruff check .                     # Lint
```

### Evaluate your parameters

Before running a simulation, open the **[parameter evaluation notebook](src/notebooks/parameter_evaluation.ipynb)** to check that your configuration will produce outbursts:

```bash
jupyter notebook src/notebooks/parameter_evaluation.ipynb
```

Edit the parameter cell (M_star, R_K, R_N, M_dot, dt_mult, etc.), run all cells, and check for **PASS**. The notebook will:
- Verify mass deposition won't be silently skipped (dt constraint)
- Verify the disk is unstable enough to outburst (DIM criterion)
- Compare your config against all 7 reference simulations
- Sweep dt_mult and M_dot to show the safe parameter ranges

Recommended workflow: **Evaluate parameters** → **Run simulation** → **Visualize results**

## Performance

Leaf physics functions (opacity, viscosity, disk physics, steady-state) are compiled to machine code via numba's `@cpu_jit` decorator, providing significant speedup for time-dependent simulations. The Newton solvers use JIT-compiled root functions with scipy's secant method for robust convergence near the S-curve thermal instability transition.

With GPU libraries installed (`pip install -e ".[gpu,dev]"`), a typical 1-million-timestep simulation completes in **~10–20 minutes**. On CPU-only, the same simulation can take **multiple hours to over a day** depending on grid size and the number of timesteps.

When numba is not installed, `@cpu_jit` falls back to a transparent passthrough — all code runs correctly in pure Python.

## Package Modules

| Module | Description | JIT |
|--------|-------------|-----|
| `constants` | CGS physical constants (G, c, k_B, sigma_SB, M_sun, etc.) | — |
| `opacity` | Multiple opacity regimes (electron scattering, Kramers, H-minus, molecular, conduction, bound-free) | Yes |
| `viscosity` | Temperature-dependent alpha viscosity with smooth hot/cold transition | Yes |
| `steady_state` | Shakura-Sunyaev steady-state disk structure (inner/middle/outer regions with boundary functions) | Yes |
| `disk_physics` | Core disk physics (pressure, density, scale height, coordinate transforms X/R, Keplerian angular velocity) | Yes |
| `irradiation` | X-ray irradiation feedback: flux, epsilon parameter, critical surface densities (Sigma_max/min), critical temperatures (T_c_max/min), irradiation-modified alpha viscosity | No |
| `evolution` | Surface density time-stepping via viscous diffusion (explicit or Crank-Nicolson implicit), mass addition at transfer radius, tidal torques, disk evaporation | No |
| `solvers` | Newton-method solvers for energy balance (midplane temperature) and hydrostatic equilibrium (scale height) | Root fns |
| `luminosity` | Radiative luminosity diagnostics (total and per-annulus) and effective temperature profiles | No |
| `cr_solvers` | Critical-regime steady-state disk structure: radiation pressure, Keplerian omega, scale height, coupled rho-T solver via Levenberg-Marquardt | No |
| `parameter_evaluation` | Pre-flight parameter checks: mass deposition constraint and DIM instability criterion | No |

## Notebooks

The recommended workflow is: **(1) evaluate parameters** → **(2) run simulation** → **(3) visualize results**. Steady-state notebooks can be run independently at any time.

### Parameter Evaluation (start here)

| Notebook | Description |
|----------|-------------|
| `parameter_evaluation` | **Start here.** Interactive pre-flight checks for your simulation parameters. Edit M_star, R_K, R_N, M_dot, and dt_mult, then run to verify mass deposition and DIM instability constraints. Includes a comparison table of all 7 reference configurations and parameter sensitivity sweeps (dt_mult and M_dot) with plots. |

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

| Notebook | Object | M_star | R_in (cm) | R_out (cm) | M_dot (g/s) | Irradiation | Evaporation | CSV suffix | dt multiplier | Timesteps |
|----------|--------|--------|-----------|------------|-------------|-------------|-------------|------------|---------------|-----------|
| `wd_timedep_simulation` | White dwarf | 1 M_sun | 5e8 | 8e10 | 5e16 | No | No | `_wd` | 10x | 1,000,001 |
| `bh_timedep_simulation` | BH base | 9 M_sun | 5e8 | 4.2e11 | 3e17 | No | No | `_bh` | 30x | 1,500,001 |
| `bh_noeffects_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | No | No | `_bhne` | 30x | 1,500,001 |
| `bh_irradiation_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | Yes | No | `_ir` | 20–30x | 500,001 |
| `bh_evaporation_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | No | Yes | `_ev` | 30–100x | 500,001 |
| `bh_iradevap_timedep_simulation` | BH (9 M_sun) | 9 M_sun | 5e8 | 4.2e11 | 1e17 | Yes | Yes | `_irev` | 30x | 500,001 |
| `sgr_a_timedep_simulation` | Sgr A* SMBH | 4.3e6 M_sun | 4e12 | 2e15 | 1e22 | Yes | Yes | _(none)_ | 10–300x (clamped) | 500,001 |

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
| `wd_outburst_plots` | White dwarf 3-panel plot: mass accretion rate, luminosity, and outer radius vs time (years) | `wd_timedep_simulation` |
| `bh_outburst_plots` | BH base 3-panel plot: mass accretion rate, luminosity, and outer radius vs time (years) | `bh_timedep_simulation` |
| `bh_noeffects_outburst_plots` | BH (9 M_sun) no-effects 3-panel plot: mass accretion rate, luminosity, and outer radius vs time (years) | `bh_noeffects_timedep_simulation` |
| `bh_irradiation_outburst_plots` | BH (9 M_sun) irradiation 3-panel plot: mass accretion rate, luminosity, and outer radius vs time (years) | `bh_irradiation_timedep_simulation` |
| `bh_evaporation_outburst_plots` | BH (9 M_sun) evaporation 3-panel plot: mass accretion rate, luminosity, and outer radius vs time (years) | `bh_evaporation_timedep_simulation` |
| `bh_iradevap_outburst_plots` | BH (9 M_sun) irradiation+evaporation 3-panel plot: mass accretion rate, luminosity, and outer radius vs time (years) | `bh_iradevap_timedep_simulation` |
| `sgr_a_outburst_plots` | Sgr A* focused: 3-panel plot of mass accretion rate, luminosity, and outer radius vs time (years) | `sgr_a_timedep_simulation` |

## Physics Overview

The codebase implements the Disk Instability Model (DIM) for accretion disks around compact objects:

1. **Opacity** - Multiple opacity regimes determine how efficiently the disk radiates: electron scattering (high-T), Kramers/bound-free (intermediate), H-minus (low-T), molecular, and conduction
2. **Alpha viscosity** - The Shakura-Sunyaev alpha prescription with a smooth temperature-dependent transition between cold (alpha ~ 0.04) and hot (alpha ~ 0.2) states, creating the thermal-viscous instability
3. **Time-dependent evolution** - Viscous diffusion of surface density on a 1D radial grid (X = 2*sqrt(R) coordinates), with mass addition at the transfer radius, tidal torques from the companion, and optional disk evaporation
4. **Irradiation** - X-ray heating from the inner disk modifies the critical temperatures for the hot/cold transition, stabilizing the outer disk
5. **Evaporation** - Removal of mass from the inner disk due to coronal evaporation above the Eddington rate

Physical units are CGS throughout the codebase.

### Parameter Evaluation

The `ParameterEvaluation` class provides pre-flight checks for simulation parameters, verifying two critical constraints before committing to a multi-hour run:

1. **Mass deposition** — `dt_used < dt_max`: if violated, `add_mass` silently skips deposition and the disk stays empty.
2. **DIM instability** — `Sigma_ss / Sigma_max > threshold`: if the cold-disk steady-state surface density is too far below the S-curve critical density, the disk is permanently stable and will never outburst.

**Usage:**

```python
from blackhole.constants import M_sun
from blackhole.parameter_evaluation import ParameterEvaluation

# Set up the physical system (WD example)
pe = ParameterEvaluation(
    M_star=M_sun, R_1=5e8, R_K=2.1e10, R_N=8e10,
    M_dot=5e16,
)

# Test a timestep configuration
result = pe.evaluate(dt_mult=10, dt_floor=200)

print(f"Valid: {result.valid}")
print(f"Mass deposition OK: {result.mass_deposition_ok} (margin: {result.deposition_margin:.1f}x)")
print(f"Instability OK: {result.instability_ok} (ratio: {result.instability_ratio:.2f})")
print(f"dt_used={result.dt_used:.0f} s, dt_max={result.dt_max:.0f} s")
print(f"Sigma_ss={result.sigma_ss:.0f}, Sigma_max={result.sigma_max:.0f} g/cm^2")
print(f"Viscous timescale: {result.t_viscous:.2e} s")
```

The constructor sets up the physical system (grid, cold-disk viscosity); `evaluate()` tests a specific timestep strategy, so you can compare multiple `dt_mult` / `dt_floor` / `dt_cap` combinations on the same system. The `EvaluationResult` dataclass contains:

- `valid` — `True` if both constraints are satisfied
- `mass_deposition_ok` / `instability_ok` — individual constraint results
- `deposition_margin` — `dt_max / dt_used` (>1 means safe)
- `instability_ratio` — `Sigma_ss / Sigma_max` (>0.3 default threshold for outbursts)
- `dt_cfl`, `dt_used`, `dt_max` — timestep diagnostics (s)
- `sigma_ss`, `sigma_max` — surface density diagnostics (g/cm^2)
- `t_viscous` — viscous timescale at the circularisation radius (s)

### Numerical Stability

Explicit finite-difference schemes for diffusion equations are subject to the **Courant-Friedrichs-Lewy (CFL) condition**: the timestep must satisfy `dt < 0.5 * X_0² * dX² / (12 * max(nu))`, where `nu` is the kinematic viscosity. If `dt` exceeds this limit, numerical errors grow exponentially and the solution diverges. This is problematic for accretion disk simulations where viscosity varies by orders of magnitude during outbursts, forcing very small timesteps and long runtimes.

The viscous diffusion step in `evolve_surface_density` supports both explicit and implicit time integration, controlled by the `theta` parameter:

| Scheme | `theta` | Stability | Accuracy | Use case |
|--------|---------|-----------|----------|----------|
| Forward Euler (explicit) | 0.0 | Conditional (CFL limited) | O(dt) | Small timesteps, simple cases |
| Crank-Nicolson | 0.5 | **Unconditionally stable** | O(dt²) | All production simulations |
| Backward Euler (implicit) | 1.0 | Unconditionally stable | O(dt) | Maximum damping |

All time-dependent simulation notebooks use **Crank-Nicolson (`theta=0.5`)** with timestep multipliers (10x for WD, 20–30x for BH, 10–300x for Sgr A* clamped to dt_max) to cover sufficient physical time for outburst cycles. The implicit scheme solves a tridiagonal system via `scipy.linalg.solve_banded` at O(N) cost per timestep — the same computational cost as the explicit scheme — while allowing timesteps well beyond the CFL stability limit.

Tidal torques and evaporation are applied as explicit operator-split source terms after the implicit diffusion step, as they are small corrections that do not drive the CFL constraint.

### Mass Deposition Constraint

The `add_mass` function deposits mass at the outer disk edge with angular momentum conservation. To prevent unphysical surface density jumps, it enforces a safety cap: deposited Sigma must satisfy `Sigma_deposit < 200 g/cm²`. This imposes an **upper bound on dt**:

The deposited surface density scales as:

```
Sigma_deposit ~ 4 * M_dot * dt / (pi * X_K^2 * dX * X_K)
```

where `X_K = 2*sqrt(R_K)` is the X-coordinate of the transfer radius. Requiring `Sigma_deposit < 200` gives:

```
dt < dt_max = 200 * pi * X_K^3 * dX / (4 * M_dot)
```

If dt exceeds this limit, `add_mass` silently skips deposition and the disk remains empty. This constraint is independent of the CFL condition — it limits the maximum rather than minimum timestep.

| Simulation | X_K | dX | M_dot (g/s) | dt_max (s) | dt_used (s) | Margin |
|------------|-----|-----|-------------|------------|-------------|--------|
| WD | 2.9e5 | 5.2e3 | 5e16 | 4.0e5 | 8.4e3 | 47x |
| BH base | 9.4e5 | 1.3e4 | 3e17 | 5.4e6 | 4.8e4 | 112x |
| BH (9 M_sun) | 9.4e5 | 1.3e4 | 1e17 | 1.6e7 | 1.6e5 | 100x |
| Sgr A* | 6.3e7 | 8.5e5 | 1e22 | 3.4e9 | 2e9 | 1.7x |

For stellar-mass systems (WD and BH), the CFL timestep is naturally well below dt_max — the constraint is automatically satisfied. For **Sgr A*** with M_dot = 1e22 g/s, dt_max drops to ~3.4e9 s. The notebook clamps dt to `[1e5, 2e9]` to stay safely below this limit while covering ~2.4 viscous timescales in 500k steps.

### Tidal Torque Scaling

Tidal torques from a binary companion remove angular momentum from the outer disk at a rate proportional to `(r / a_1)^n_1`, where `a_1` is the binary separation and `n_1 = 5`. The tidal torque surface density loss per timestep is:

```
dSigma_tid = dt * cw * nu * Sigma * (r / a_1)^5 / (2 * pi)
```

This term grows **extremely steeply** with `r / a_1`. The tidal truncation radius (where torques become significant) must lie well outside the disk: `R_N / a_1 << 1`. If `R_N / a_1 >= 1`, the torque at the outer edge overwhelms viscous mass transport and prevents surface density from building up to the instability threshold.

| Simulation | R_N (cm) | a_1 (cm) | R_N / a_1 | (R_N/a_1)^5 | Status |
|------------|----------|----------|-----------|-------------|--------|
| WD | 8e10 | 2.1e11 | 0.38 | 7.9e-3 | OK |
| BH (9 M_sun) | 4.2e11 | 2.2e12 | 0.19 | 2.5e-4 | OK |
| BH base | 4.2e11 | 2.2e12 | 0.19 | 2.5e-4 | OK |
| Sgr A* | 2e15 | 1.5e16 | 0.13 | 3.7e-5 | OK |

For stellar-mass binaries, `a_1` is the orbital separation — typically 3–10x the disk outer radius. For Sgr A* (fed by stellar winds at the Galactic Centre, not Roche lobe overflow), `a_1` is set large enough that tidal torques are a negligible perturbation at the outer edge.

### DIM Instability Criterion

For the Disk Instability Model (DIM) to produce outbursts, the mass transfer rate must be high enough that the cold-disk steady-state surface density at the circularisation radius exceeds the critical surface density `Sigma_max` (the upper limit of the cold branch on the S-curve):

```
Sigma_ss = M_dot / (3 * pi * nu_cold)
Sigma_max ~ 10.8 * alpha_cold^(-0.84) * (M/M_sun)^(-0.37) * r10^1.11
```

If `Sigma_ss < Sigma_max`, the disk remains permanently in the cold stable state and never outbursts. The ratio `Sigma_ss / Sigma_max` determines the outburst behavior:

| Simulation | M_star | M_dot (g/s) | Sigma_ss (g/cm²) | Sigma_max (g/cm²) | Ratio | Outbursts? |
|------------|--------|-------------|------------------|-------------------|-------|------------|
| WD | 1 M_sun | 5e16 | ~600 | ~400 | ~1.5 | Yes |
| BH base | 9 M_sun | 3e17 | ~2900 | ~2200 | ~1.3 | Yes |
| BH (9 M_sun) | 9 M_sun | 1e17 | ~980 | ~2200 | ~0.45 | Yes (local accumulation) |
| Sgr A* | 4.3e6 M_sun | 1e22 | ~4.4e5 | ~3.6e5 | ~1.2 | Yes |

Note: Even when `Sigma_ss / Sigma_max < 1` (as in the BH 9 M_sun case), local mass accumulation at the deposition radius can exceed `Sigma_max` before viscous spreading redistributes the mass, triggering outbursts.

## Technology Stack

- **Python 3.12+**
- **NumPy/SciPy** - Scientific computing and numerical methods
- **Numba** - CPU JIT compilation for physics functions (optional, falls back to pure Python)
- **CuPy** - GPU array operations (optional, for large grid sizes N > 10,000)
- **Numba CUDA** - GPU JIT compilation for time-dependent simulations (optional)
- **Matplotlib** - Visualization
- **Pandas** - Data I/O for simulation histories

## Git Workflow

Branch strategy: `feature` → `dev` → `main`

- CI runs on pull requests to `dev` and `main`
- Deploy runs on push to `main`
