# Black Hole X-ray Outburst

Python project for modelling X-ray outbursts from stellar compact objects (black holes and white dwarfs). Uses numerical methods to solve accretion disk equations and visualize outburst behavior.

## Project Structure

```
blackhole/
├── pyproject.toml          # Project config, dependencies, tool settings
├── src/
│   ├── notebooks/          # Jupyter notebooks (simulations & analysis)
│   ├── graphs/             # Output plots from notebooks
│   └── data/               # CSV simulation data (git-ignored)
├── tests/                  # Test directory
└── .github/workflows/      # CI/CD pipelines
```

## Getting Started

See [SETUP.md](SETUP.md) for installation instructions, CUDA/GPU setup, and running notebooks.

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

- **Python 3.14**
- **NumPy/SciPy** - Scientific computing and numerical methods
- **Numba/CUDA** - GPU acceleration for time-dependent simulations
- **CuPy** - GPU array operations
- **Matplotlib** - Visualization
- **Pandas** - Data processing

## Git Workflow

Branch strategy: `feature` → `dev` → `main`
