# ShittyBird

A fluid dynamics simulation library built on JAX focused on the Lattice Boltzmann Method.

## Features

- Efficient Lattice Boltzmann Method (LBM) implementation using JAX
- Rayleigh-Benard convection simulation capabilities
- Visualizations of fluid dynamics simulations
- Built on the core functionality of RLLBM (see here: https://github.com/hlasco/rllbm/tree/master)

## Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/jtbuch/shitty_bird.git
cd shitty_bird

# Install with Poetry
poetry install

# Verify installation by running tests
poetry run pytest
```

If you encounter any dependency issues, specifically with IPython, you can install it manually:

```bash
poetry add ipython@^8.0.0
```

## Quick Start

Simulate Rayleigh-Benard convection:

```python
import jax.numpy as jnp
import jax
from shitty_bird.core import run_simulation
from shitty_bird.simulations import rayleigh_benard

# Configure the simulation
config = {
    "n": 96,            # Grid size
    "pr": 0.71,         # Prandtl number
    "ra": 1e6,          # Rayleigh number
    "buoy": 0.0001,     # Buoyancy
    "save_video": True
}

# Run the simulation
run_simulation(rayleigh_benard, config)
```

## Examples

Check the `examples` directory for more simulation examples:

- `rayleigh_benard_simulation.py`: Basic Rayleigh-Benard convection setup

## Development

### Setting Up Development Environment

```bash
# Install all dependencies including development dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Quality Assurance

```bash
# Run all tests
poetry run pytest

# Run a specific test file
poetry run pytest tests/test_rayleigh_benard.py

# Run linting
poetry run ruff check .

# Run type checking
poetry run mypy src/
```

### Running Examples

```bash
# Run the Rayleigh-Benard simulation example
poetry run python examples/rayleigh_benard_simulation.py
```

## License

MIT License
