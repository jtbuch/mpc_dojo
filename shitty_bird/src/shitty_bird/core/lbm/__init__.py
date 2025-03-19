"""Lattice Boltzmann Method implementation."""

from .boundary import Boundary, BoundaryDict, BounceBackBoundary, InletBoundary, OutletBoundary, apply_boundary_conditions
from .collisions import collide
from .lattice import Lattice, CoupledLattices, FluidLattice, ThermalFluidLattice
from .simulation import LBMState, Simulation, Domain
from .stencil import Stencil, D1Q3, D2Q5, D2Q9
from .stream import stream

__all__ = [
    "Boundary",
    "BoundaryDict",
    "BounceBackBoundary",
    "InletBoundary",
    "OutletBoundary",
    "apply_boundary_conditions",
    "collide",
    "Lattice",
    "CoupledLattices",
    "FluidLattice",
    "ThermalFluidLattice", 
    "LBMState",
    "Simulation",
    "Domain",
    "Stencil",
    "D1Q3",
    "D2Q5",
    "D2Q9",
    "stream",
]