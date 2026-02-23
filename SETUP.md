# Setup & Installation

## Prerequisites

- **Python 3.12+** ([python.org](https://www.python.org/downloads/))
- **Git** ([git-scm.com](https://git-scm.com/))
- **NVIDIA GPU** with up-to-date drivers (for CUDA acceleration)
- **CUDA Toolkit 13.x** ([developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads))

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

**With GPU (requires NVIDIA CUDA):**

```bash
pip install -e ".[dev]"
```

This installs:
- **Core**: numpy, scipy, matplotlib, pandas, numba, numba-cuda, cupy-cuda12x
- **Dev**: pytest, jupyter, ruff

**Without GPU (CPU-only dev/CI):**

```bash
pip install -e ".[cpu,dev]"
```

This installs only the CPU-compatible packages (numpy, scipy, matplotlib, pandas, pytest, jupyter, ruff).

### 4. Verify the installation

```bash
# CPU-only verification
python -c "from blackhole.constants import G; print('Package OK, G =', G)"

# Full GPU verification
python -c "import numpy, scipy, matplotlib, pandas, numba, cupy; print('All packages OK')"
```

## CUDA / GPU Setup

GPU acceleration is used for the time-dependent disk simulations. This requires an NVIDIA GPU with the CUDA Toolkit installed.

### Check your GPU

```bash
nvidia-smi
```

This should show your GPU model, driver version, and supported CUDA version.

### Install the CUDA Toolkit

1. Go to [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Select your OS and architecture
3. Download and run the installer

The Python CUDA libraries (`numba-cuda`, `nvidia-nvvm`, etc.) are installed automatically via pip as part of the `numba-cuda[cu13]` dependency. No separate Python-side CUDA setup is needed.

### Verify CUDA

```bash
python -c "from numba import cuda; print('CUDA available:', cuda.is_available()); cuda.detect()"
```

Expected output should list your GPU and show `[SUPPORTED]`.

### Test a CUDA kernel

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = x[i] + y[i]

n = 1000
x = np.ones(n, dtype=np.float32)
y = np.ones(n, dtype=np.float32)
out = np.zeros(n, dtype=np.float32)

add_kernel[1, n](x, y, out)
print(out)  # should print array of 2.0s
```

### Verify CuPy

```bash
python -c "import cupy as cp; print('CuPy version:', cp.__version__); print('CUDA runtime:', cp.cuda.runtime.runtimeGetVersion())"
```

Expected output should show the CuPy version and CUDA runtime version number.

### Note on numba-cuda

The CUDA support in Numba is provided by the separate `numba-cuda` package (maintained by NVIDIA), not the old built-in `numba.cuda`. The import path is still `from numba import cuda` -- `numba-cuda` patches itself into the Numba namespace automatically.

## Running Without an NVIDIA GPU

The project is designed for NVIDIA GPUs, but can be adapted to run on CPU-only machines.

### 1. Install without GPU packages

Skip the GPU-specific dependencies (`numba-cuda`, `cupy-cuda12x`) by installing manually:

```bash
pip install numpy scipy matplotlib pandas numba
pip install -e ".[dev]" --no-deps
```

Or install everything and just ignore the GPU packages that fail — `pip install -e ".[dev]"` will still install the CPU-compatible packages even if `cupy-cuda12x` errors out.

### 2. Adapt imports in notebooks

In each notebook, replace the CuPy import with a NumPy-backed shim so that all `cp.*` calls fall back to NumPy on CPU:

```python
try:
    import cupy as cp
except ImportError:
    import numpy as cp
    cp.asnumpy = lambda x: x  # no-op since arrays are already NumPy
```

Place this right after `import numpy as np` in each notebook's first cell.

### 3. Numba JIT functions

The `@jit(target_backend='cuda')` functions in `GPU_timedep*.ipynb` will not work without a CUDA GPU. To run those notebooks on CPU, change the decorator to plain `@jit(nopython=True)`:

```python
# GPU version (requires NVIDIA GPU)
@jit(target_backend='cuda', nopython=True)
def my_func(...):

# CPU-only version
@jit(nopython=True)
def my_func(...):
```

### 4. Which notebooks work on CPU

| Notebook | CPU-only? | Notes |
|----------|-----------|-------|
| `opacity_formulae.ipynb` | Yes | Only needs the CuPy import shim |
| `diskequations_SS_bath_params.ipynb` | Yes | Only needs the CuPy import shim |
| `diskeqs_ss_2.ipynb` | Yes | Only needs the CuPy import shim |
| `alpha-t dependence.ipynb` | Yes | Only needs the CuPy import shim |
| `bh_model_comparison_graphs.ipynb` | Yes | Only needs the CuPy import shim |
| `Outburst_graphs.ipynb` | Yes | Only needs the CuPy import shim (loads pre-computed CSV data) |
| `GPU_timedep*.ipynb` | Partial | Needs both the CuPy import shim **and** JIT decorator changes. Will run but significantly slower without GPU acceleration. |

## Running Notebooks

```bash
jupyter notebook src/notebooks/
```

## Linting

```bash
ruff check .
```

## Running Tests

```bash
pytest                # Run all 119 tests
pytest tests/ -v      # Verbose output
```

## Project Structure

```
blackhole/
├── pyproject.toml
├── src/
│   ├── blackhole/          # Python package (installable via pip)
│   │   ├── __init__.py     # gpu_jit decorator (CUDA → CPU numba → passthrough)
│   │   ├── constants.py    # CGS physical constants
│   │   ├── opacity.py      # Opacity regimes
│   │   ├── viscosity.py    # Temperature-dependent alpha viscosity
│   │   ├── steady_state.py # Shakura-Sunyaev steady-state disk structure
│   │   ├── disk_physics.py # Core disk physics
│   │   ├── irradiation.py  # Irradiation feedback
│   │   ├── evolution.py    # Surface density time-stepping
│   │   ├── solvers.py      # Newton solvers (temperature & scale height)
│   │   └── luminosity.py   # Radiative luminosity & effective temperature
│   ├── notebooks/          # Jupyter notebooks (simulations & analysis)
│   ├── graphs/             # Output plots from notebooks
│   └── data/               # CSV simulation data (git-ignored)
├── tests/                  # Unit tests (119 tests)
└── .github/workflows/      # CI/CD pipelines
```

## Git Workflow

Branch strategy: `feature` → `dev` → `main`

- Create feature branches off `dev`
- Open PRs targeting `dev`
- CI (lint + tests) runs on PRs to `dev` and `main`
- Release by merging `dev` → `main` via PR
