# ShittyBird

A fluid dynamics simulation library built on JAX focused on the Lattice Boltzmann Method.

## Features

- Efficient Lattice Boltzmann Method (LBM) implementation using JAX
- Rayleigh-Benard convection simulation capabilities
- Visualizations of fluid dynamics simulations
- Built on the core functionality of RLLBM (see here: https://github.com/hlasco/rllbm/tree/master)

## Installation

```bash
# Clone the repository
git clone https://github.com/jtbuch/shitty_bird.git
cd shitty_bird

# Install with Poetry
poetry install
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

```bash
# Install development dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Run type checking
poetry run mypy src/
```

## License

MIT License
