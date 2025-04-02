"""Tests for Rayleigh-Benard convection simulation."""

import numpy as np
import pytest
from simulations.rayleigh_benard import RayleighBenardSimulation, get_diagnostics


def test_rayleigh_benard_init():
    """Test initialization of Rayleigh-Benard simulation."""
    sim = RayleighBenardSimulation(n=16, pr=0.71, ra=1e4, buoy=0.0001)
    
    # Check that the simulation was initialized correctly
    assert sim.n == 16
    assert sim.pr == 0.71
    assert sim.ra == 1e4
    assert sim.buoy == 0.0001
    
    # Check that history has initial state
    assert len(sim.history) == 1
    
    # Check that fluid state is initialized
    assert not np.isnan(sim.sim.fluid_state.T).any()
    assert not np.isnan(sim.sim.fluid_state.u).any()
    assert not np.isnan(sim.sim.fluid_state.rho).any()


def test_rayleigh_benard_diagnostics():
    """Test diagnostic functions for Rayleigh-Benard simulation."""
    sim = RayleighBenardSimulation(n=16, pr=0.71, ra=1e4, buoy=0.0001)
    
    # Get diagnostics
    diags = get_diagnostics(sim.sim)
    
    # Check that diagnostics were computed correctly
    assert len(diags) == 4
    assert diags[0].name == "temperature"
    assert diags[1].name == "density"
    assert diags[2].name == "velocity_x"
    assert diags[3].name == "velocity_y"
    
    # Check that diagnostics have expected values
    for diag in diags:
        assert not np.isnan(diag.min)
        assert not np.isnan(diag.max)
        assert not np.isnan(diag.mean)
        assert not np.isnan(diag.std)


def test_rayleigh_benard_short_run():
    """Test running Rayleigh-Benard simulation for a few steps."""
    sim = RayleighBenardSimulation(n=16, pr=0.71, ra=1e4, buoy=0.0001)
    
    # Run for a few steps
    sim.run(steps=10, save_frequency=5, display_frequency=0, progress_bar=False)
    
    # Check that history has been updated
    assert len(sim.history) == 3  # Initial + 2 saved states
    
    # Check that simulation time has advanced
    assert sim.current_step == 10
    assert sim.current_time > 0