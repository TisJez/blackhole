# Setup & Installation

## Prerequisites

- **Python 3.12+** ([python.org](https://www.python.org/downloads/))
- **Git** ([git-scm.com](https://git-scm.com/))
- **NVIDIA GPU** with up-to-date drivers (optional, for CUDA acceleration)
- **CUDA Toolkit 13.x** ([developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)) (optional)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/TisJez/blackhole.git
cd blackhole
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (cmd)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

**CPU-only (dev/CI):**

```bash
pip install -e ".[dev]"
```

This installs:
- **Core**: numpy, scipy, matplotlib, pandas
- **Dev**: pytest, jupyter, ruff

**With GPU (requires NVIDIA CUDA):**

```bash
pip install -e ".[gpu,dev]"
```

This additionally installs: numba, numba-cuda, cupy-cuda12x

### 4. Verify the installation

```bash
# CPU verification
python -c "from blackhole.constants import G; print('Package OK, G =', G)"

# Run tests
pytest                     # 244 tests

# Lint
ruff check .

# GPU verification (optional)
python -c "import cupy as cp; print('CuPy version:', cp.__version__)"
```

## CUDA / GPU Setup

GPU acceleration is optional. The package works fully on CPU. GPU support is provided via CuPy (array dispatch) and Numba CUDA (JIT compilation).

### Check your GPU

```bash
nvidia-smi
```

This should show your GPU model, driver version, and supported CUDA version.

### Install the CUDA Toolkit

1. Go to [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Select your OS and architecture
3. Download and run the installer

The Python CUDA libraries (`numba-cuda`, `nvidia-nvvm`, etc.) are installed automatically via pip as part of the `numba-cuda[cu13]` dependency.

### Verify CUDA

```bash
python -c "from numba import cuda; print('CUDA available:', cuda.is_available()); cuda.detect()"
```

### Verify CuPy

```bash
python -c "import cupy as cp; print('CuPy version:', cp.__version__); print('CUDA runtime:', cp.cuda.runtime.runtimeGetVersion())"
```

### Note on numba-cuda

The CUDA support in Numba is provided by the separate `numba-cuda` package (maintained by NVIDIA), not the old built-in `numba.cuda`. The import path is still `from numba import cuda` -- `numba-cuda` patches itself into the Numba namespace automatically.

## Windows CUDA DLL Resolution

On Windows, NVIDIA pip packages (e.g. `cupy-cuda12x`) store CUDA shared libraries in separate `site-packages/nvidia/*/bin/` directories. The `blackhole.gpu` package automatically registers these directories via `os.add_dll_directory()` at import time, so CUDA libraries (cusparse, cusolver, etc.) load correctly even when the system CUDA Toolkit version differs from the CuPy build target. No manual `PATH` configuration is needed.

## CuPy/NumPy Array Dispatch

All physics modules support transparent GPU acceleration via `get_xp()` in `blackhole/__init__.py`. When you pass CuPy arrays to any function, it automatically uses CuPy operations. When you pass NumPy arrays, it uses NumPy. No code changes needed:

```python
import numpy as np
from blackhole.opacity import kappa_e

# CPU path (NumPy)
T = np.array([1e4, 1e5, 1e6])
result = kappa_e(T)  # uses numpy internally

# GPU path (CuPy) — same function, same call
import cupy as cp
T_gpu = cp.array([1e4, 1e5, 1e6])
result_gpu = kappa_e(T_gpu)  # uses cupy internally
```

## Running Notebooks

### From the command line

```bash
jupyter notebook notebooks/
```

### From IntelliJ / PyCharm

1. Install the **Jupyter** plugin (Settings > Plugins)
2. Open any `.ipynb` file — it opens in the built-in Jupyter editor
3. Select your Python interpreter (the `.venv` you created) from the kernel dropdown
4. Run cells with Shift+Enter

### Notebook execution order

The simulation notebooks are independent — each initializes a fresh disk. Run any simulation notebook to generate its CSV data files, then run a visualization notebook to plot the results.

**Typical workflow:**

1. Run a simulation notebook (e.g., `simulations/sgr_a.ipynb`) — generates CSV files in `data/`
2. Run a plotting notebook (e.g., `visualization/sgr_a.ipynb`) — reads the CSVs and produces plots
3. Steady-state notebooks (`steady_state/opacity_regimes`, `steady_state/alpha_viscosity`, etc.) are self-contained and can be run independently

## Running Tests

```bash
pytest                    # Run all 244 tests
pytest tests/ -v          # Verbose output
pytest -k "test_opacity"  # Run specific tests by name
```

Tests marked `@requires_cupy` are automatically skipped on machines without CuPy installed.

## Linting

```bash
ruff check .              # Check for errors
ruff check --fix .        # Auto-fix import sorting etc.
```

## Project Structure

```
blackhole/
├── pyproject.toml
├── README.md
├── SETUP.md
├── CLAUDE.md
├── src/
│   └── blackhole/                              # Python package
│       ├── __init__.py                         # cpu_jit, gpu_jit, get_xp()
│       ├── constants.py                        # CGS physical constants
│       ├── opacity.py                          # Opacity regimes
│       ├── viscosity.py                        # Alpha viscosity
│       ├── steady_state.py                     # SS73 steady-state
│       ├── disk_physics.py                     # Core disk physics
│       ├── irradiation.py                      # Irradiation feedback
│       ├── evolution.py                        # Time-stepping
│       ├── solvers.py                          # Newton solvers
│       ├── luminosity.py                       # Luminosity & T_eff
│       ├── cr_solvers.py                       # CR steady-state
│       ├── parameter_evaluation.py             # Pre-flight parameter checks
│       └── gpu/                                # GPU-accelerated subpackage
│           ├── __init__.py                     # Array dispatch (get_xp, to_device, to_host); Windows NVIDIA DLL registration
│           ├── opacity.py                      # Vectorized opacity
│           ├── disk_physics.py                 # Vectorized disk physics
│           ├── viscosity.py                    # Vectorized alpha viscosity
│           ├── solvers.py                      # Batched secant solvers
│           ├── evolution.py                    # Vectorized evolution; GPU-native sparse tridiagonal solve
│           ├── luminosity.py                   # Vectorized luminosity
│           └── simulation.py                   # GPU simulation orchestrator
├── notebooks/                                  # Jupyter notebooks
│   ├── steady_state/                           # Equilibrium disk analysis
│   │   ├── opacity_regimes.ipynb               # Opacity regimes
│   │   ├── alpha_viscosity.ipynb               # Alpha-T model
│   │   ├── disk_structure.ipynb                # SS73 disk structure
│   │   ├── disk_subplots.ipynb                 # SS73 subplots
│   │   └── model_comparison.ipynb              # SS73 vs CR comparison
│   ├── simulations/                            # Time-dependent simulations
│   │   ├── wd.ipynb                            # White dwarf sim
│   │   ├── bh_base.ipynb                       # BH base sim
│   │   ├── bh_noeffects.ipynb                  # BH no-effects sim
│   │   ├── bh_irradiation.ipynb                # BH irradiation sim
│   │   ├── bh_evaporation.ipynb                # BH evaporation sim
│   │   ├── bh_iradevap.ipynb                   # BH irrad+evap sim
│   │   ├── sgr_a.ipynb                         # Sgr A* SMBH sim
│   │   ├── gpu_bh_base.ipynb                   # GPU BH base sim
│   │   ├── gpu_bh_iradevap.ipynb               # GPU BH irrad+evap sim
│   │   └── gpu_sgr_a.ipynb                     # GPU Sgr A* SMBH sim
│   ├── visualization/                          # Outburst plots & lightcurves (11)
│   │   ├── lightcurves.ipynb                   # Multi-model plots
│   │   ├── wd.ipynb                            # WD outburst plot
│   │   ├── bh_base.ipynb                       # BH base outburst plot
│   │   ├── bh_noeffects.ipynb                  # BH no-effects outburst plot
│   │   ├── bh_irradiation.ipynb                # BH irradiation outburst plot
│   │   ├── bh_evaporation.ipynb                # BH evaporation outburst plot
│   │   ├── bh_iradevap.ipynb                   # BH irrad+evap outburst plot
│   │   ├── sgr_a.ipynb                         # Sgr A* outburst plot
│   │   ├── gpu_bh_base.ipynb                   # GPU BH base outburst plot
│   │   ├── gpu_bh_iradevap.ipynb               # GPU BH irrad+evap outburst plot
│   │   └── gpu_sgr_a.ipynb                     # GPU Sgr A* outburst plot
│   └── tools/                                  # Utility notebooks
│       └── parameter_evaluation.ipynb          # Parameter pre-flight checks
├── graphs/                                     # Output plots
├── data/                                       # CSV data (git-ignored)
├── tests/                                      # 244 unit tests
└── .github/workflows/                          # CI/CD
```

## Git Workflow

Branch strategy: `feature` → `dev` → `main`

- Create feature branches off `dev`
- Open PRs targeting `dev`
- CI (lint + tests) runs on PRs to `dev` and `main`
- Release by merging `dev` → `main` via PR
