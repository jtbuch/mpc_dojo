"""Core simulation module for ShittyBird."""

from typing import Dict, Any, Callable

import jax
import jax.numpy as jnp

from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning
import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class Diagnostic:
    """Store and log diagnostic data from simulations."""
    
    def __init__(self, name: str, values: jnp.ndarray) -> None:
        """Initialize a diagnostic object with simulation values.
        
        Args:
            name: The name of the diagnostic
            values: The array of values to compute statistics for
        """
        self.name = name
        self.min = jnp.min(values)
        self.max = jnp.max(values)
        self.mean = jnp.mean(values)
        self.std = jnp.std(values)
    
    def __str__(self) -> str:
        """Return a string representation of the diagnostic."""
        return (f"{self.name}: min={self.min:.4e}, max={self.max:.4e}, "
                f"mean={self.mean:.4e}, std={self.std:.4e}")


def run_simulation(simulation_func: Callable, config: Dict[str, Any]) -> None:
    """Run a simulation with the given configuration.
    
    Args:
        simulation_func: The simulation function to run
        config: Configuration parameters for the simulation
    """
    simulation_func(**config)