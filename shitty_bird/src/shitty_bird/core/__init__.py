"""Core functionality for ShittyBird."""

from .simulation import run_simulation
from . import lbm

__all__ = ["simulation", "run_simulation", "lbm"]