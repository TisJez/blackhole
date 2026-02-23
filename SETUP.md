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
pytest                     # 172 tests

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
jupyter notebook src/notebooks/
```

### From IntelliJ / PyCharm

1. Install the **Jupyter** plugin (Settings > Plugins)
2. Open any `.ipynb` file — it opens in the built-in Jupyter editor
3. Select your Python interpreter (the `.venv` you created) from the kernel dropdown
4. Run cells with Shift+Enter

### Notebook execution order

The simulation notebooks are independent — each initializes a fresh disk. Run any simulation notebook to generate its CSV data files, then run a visualization notebook to plot the results.

**Typical workflow:**

1. Run a simulation notebook (e.g., `sgr_a_timedep_simulation.ipynb`) — generates CSV files in `src/data/`
2. Run a plotting notebook (e.g., `sgr_a_outburst_plots.ipynb`) — reads the CSVs and produces plots
3. Steady-state notebooks (`opacity_constants`, `viscosity_temperature_dependence`, etc.) are self-contained and can be run independently

## Running Tests

```bash
pytest                    # Run all 172 tests
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
│   ├── blackhole/                              # Python package
│   │   ├── __init__.py                         # gpu_jit, get_xp()
│   │   ├── constants.py                        # CGS physical constants
│   │   ├── opacity.py                          # Opacity regimes
│   │   ├── viscosity.py                        # Alpha viscosity
│   │   ├── steady_state.py                     # SS73 steady-state
│   │   ├── disk_physics.py                     # Core disk physics
│   │   ├── irradiation.py                      # Irradiation feedback
│   │   ├── evolution.py                        # Time-stepping
│   │   ├── solvers.py                          # Newton solvers
│   │   ├── luminosity.py                       # Luminosity & T_eff
│   │   └── cr_solvers.py                       # CR steady-state
│   ├── notebooks/                              # Jupyter notebooks
│   │   ├── opacity_constants.ipynb             # Opacity regimes
│   │   ├── viscosity_temperature_dependence.ipynb  # Alpha-T model
│   │   ├── steady_state_disk_structure.ipynb   # SS73 disk structure
│   │   ├── steady_state_disk_subplots.ipynb    # SS73 subplots
│   │   ├── opacity_model_comparison.ipynb      # SS73 vs CR comparison
│   │   ├── wd_timedep_simulation.ipynb         # White dwarf sim
│   │   ├── bh_timedep_simulation.ipynb         # BH base sim
│   │   ├── bh_noeffects_timedep_simulation.ipynb   # BH no-effects sim
│   │   ├── bh_irradiation_timedep_simulation.ipynb # BH irradiation sim
│   │   ├── bh_evaporation_timedep_simulation.ipynb # BH evaporation sim
│   │   ├── bh_iradevap_timedep_simulation.ipynb    # BH irrad+evap sim
│   │   ├── sgr_a_timedep_simulation.ipynb      # Sgr A* SMBH sim
│   │   ├── outburst_lightcurves.ipynb          # Multi-model plots
│   │   └── sgr_a_outburst_plots.ipynb          # Sgr A* plots
│   ├── graphs/                                 # Output plots
│   └── data/                                   # CSV data (git-ignored)
├── tests/                                      # 172 unit tests
└── .github/workflows/                          # CI/CD
```

## Git Workflow

Branch strategy: `feature` → `dev` → `main`

- Create feature branches off `dev`
- Open PRs targeting `dev`
- CI (lint + tests) runs on PRs to `dev` and `main`
- Release by merging `dev` → `main` via PR
