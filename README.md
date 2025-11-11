# JAX-CTSM: Community Terrestrial Systems Model in JAX

A differentiable, GPU-accelerated reimplementation of CTSM ecosystem dynamics in JAX.

## Overview

This project translates CTSM's biogeochemical processes from Fortran to JAX, enabling:
- **Automatic differentiation** for parameter optimization and sensitivity analysis
- **GPU acceleration** for large-scale simulations
- **Vectorization** for ensemble modeling and uncertainty quantification
- **Pure functional** design for reproducibility and composability

## Project Structure

```
jax-ctsm/
├── src/
│   └── jax_ctsm/
│       ├── core/           # Core data structures and spatial hierarchy
│       ├── physics/        # Physical processes (maintenance respiration, etc.)
│       ├── params/         # Parameter definitions and loaders
│       └── utils/          # Utilities and helpers
├── tests/                  # Unit and integration tests
├── examples/               # Example notebooks and scripts
├── configs/                # Configuration files
└── docs/                   # Documentation
```

## Installation

```bash
# Create conda environment
conda create -n jax-ctsm python=3.10
conda activate jax-ctsm

# Install dependencies
pip install -e .
```

## Current Implementation Status

### ✅ Proof of Concept: Maintenance Respiration
- Complete spatial hierarchy (gridcell → landunit → column → patch)
- Temperature-dependent respiration calculations
- Leaf, root, stem, and reproductive tissue respiration
- Fully tested against Fortran CTSM output

### 🚧 In Progress
- Allocation processes
- Phenology
- Soil biogeochemistry

## Quick Start

```python
import jax.numpy as jnp
from jax_ctsm.core.hierarchy import PatchState, ColumnState
from jax_ctsm.physics.maintenance_respiration import calculate_maintenance_respiration
from jax_ctsm.params.respiration import RespirationParams

# Create sample patch state
patch = PatchState(
    nitrogen_state=NitrogenState(
        leafn=jnp.array([50.0]),
        frootn=jnp.array([30.0]),
        livestemn=jnp.array([20.0]),
    ),
    temperature=jnp.array([298.15]),  # 25°C in Kelvin
    lai_sun=jnp.array([2.5]),
    lai_shade=jnp.array([1.5]),
)

# Calculate maintenance respiration
params = RespirationParams()
mr_fluxes = calculate_maintenance_respiration(patch, params)

print(f"Leaf MR: {mr_fluxes.leaf_mr[0]:.6f} gC/m2/s")
print(f"Root MR: {mr_fluxes.froot_mr[0]:.6f} gC/m2/s")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/jax_ctsm tests/

# Run specific test
pytest tests/physics/test_maintenance_respiration.py -v
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[📖 Getting Started](QUICKSTART.md)** - Quick start guide for new users
- **[📚 Documentation Index](docs/README.md)** - Complete documentation overview
- **[🏗️ Design Documents](docs/design/)** - Architecture and implementation details
- **[📖 User Guides](docs/guides/)** - In-depth guides (spatial hierarchy, etc.)
- **[📊 Reports](docs/reports/)** - Validation results and development reports
- **[🤝 Contributing](CONTRIBUTING.md)** - Development guidelines

### Key Documents

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Quick start guide for new users |
| [docs/guides/spatial_hierarchy.md](docs/guides/spatial_hierarchy.md) | Understanding CTSM's spatial structure |
| [docs/design/driver_design.md](docs/design/driver_design.md) | Driver architecture and Fortran comparison |
| [docs/reports/validation_results.md](docs/reports/validation_results.md) | Validation against Fortran CTSM |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute to the project |

## Citation

If you use this code, please cite:
- Original CTSM: [Lawrence et al. 2019](https://doi.org/10.1029/2018MS001583)
- This JAX implementation: [TBD]

## License

BSD-3-Clause (same as CTSM)

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
