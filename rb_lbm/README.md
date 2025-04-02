# Simulating Rayleigh-Benard convection with Lattice Boltzmann methods (rb_lbm)

A Rayleigh-Benard convection simulation library using Lattice Boltzmann methods built using JAX.

## Features

- Efficient Lattice Boltzmann Method (LBM) implementation using JAX
- Rayleigh-Benard convection simulation capabilities
- Visualizations of fluid dynamics simulations
- Built on the core functionality of RLLBM (see here: https://github.com/hlasco/rllbm/tree/master)

## Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- JAX and related dependencies (optional but recommended)

### Installation steps

```bash
# Clone the repository
git clone https://github.com/jtbuch/rayleigh-benard-lbm.git
cd rayleigh-benard-lbm

# Install in development mode (with all dependencies including JAX)
pip install -e ".[default]"

# Or install without JAX dependencies (basic install)
pip install -e .

# Verify installation by running tests
pytest
```

### Google Colab support

ShittyBird includes special handling for Google Colab environments. When running in Colab:

1. The package will automatically detect JAX, JAXLIB, and other core dependencies that are pre-installed in Colab
2. Examples will use optimized parameters suitable for Colab's environment
3. To use in a Colab notebook:

```python
# Clone the repository
!git clone https://github.com/jtbuch/rayleigh-benard-lbm.git
%cd rayleigh-benard-lbm

# Install with pip
!pip install -e .

# Run examples
%run examples/rayleigh_benard_simulation.py
```

## Quick Start

Simulate Rayleigh-Benard convection:

```python
import jax.numpy as jnp
import jax
from core import Simulation, Domain, FluidLattice
from simulations import rayleigh_benard

# Configure the simulation
config = {
    "n": 96,            # Grid size
    "pr": 0.71,         # Prandtl number
    "ra": 1e6,          # Rayleigh number
    "buoy": 0.0001,     # Buoyancy
    "save_video": True
}

# Run the simulation
result = rayleigh_benard(**config)
```

## Examples

Check the `examples` directory for more simulation examples:

- `rayleigh_benard_simulation.py`: Basic Rayleigh-Benard convection setup

## Project structure

The project is organized as follows:

- `core/`: Core LBM implementation with boundary conditions, stencils, and simulation components
- `utils/`: Utility functions for visualization and environment detection
- `simulations/`: Specific simulation setups like Rayleigh-Benard convection
- `examples/`: Example scripts to demonstrate usage
- `tests/`: Unit and integration tests

## Development

### Unit tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_rayleigh_benard.py

# Run linting
ruff check .

# Run type checking
mypy .
```

### Running examples

```bash
# Run the Rayleigh-Benard simulation example
python examples/rayleigh_benard_simulation.py
```

## License

MIT License
