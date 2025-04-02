"""Visualization utilities for ShittyBird simulations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter

def fig_constructor(sim):
    """Create a figure for visualizing the simulation.
    
    Args:
        sim: The simulation object
        
    Returns:
        Dictionary containing figure components
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    
    temperature = sim.fluid_state.T[:, :, 0].T
    x = sim.x
    y = sim.y
    vmax = 0.5

    img = plt.imshow(
        temperature,
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="RdBu_r",
        norm=SymLogNorm(0.1*vmax, vmin=-vmax, vmax=vmax),
        origin="lower",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    formatter = LogFormatter(10, labelOnlyBase=False)
    cbar = fig.colorbar(img, ax=ax, format=formatter)
    cbar.set_label("Fluid Temperature")

    fig.tight_layout(pad=2)
    fig.canvas.draw()
    
    fig_data = {
        "figure": fig,
        "image": img,
        "axes": ax,
    }
    return fig_data

def fig_updater(fig_data, sim):
    """Update the figure with new simulation data.
    
    Args:
        fig_data: Dictionary containing figure components
        sim: The simulation object
        
    Returns:
        Updated figure data dictionary
    """
    temperature = sim.fluid_state.T[:, :, 0].T
    fig_data["image"].set_data(temperature)
    return fig_data

def plot_state(sim, title=None):
    """Plot the current state of the simulation with multiple views.
    
    Args:
        sim: The simulation object
        title: Optional title for the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot temperature
    temperature = sim.fluid_state.T[:, :, 0].T
    vmax = 0.5
    img1 = axes[0].imshow(
        temperature,
        extent=[sim.x.min(), sim.x.max(), sim.y.min(), sim.y.max()],
        cmap="RdBu_r",
        norm=SymLogNorm(0.1*vmax, vmin=-vmax, vmax=vmax),
        origin="lower",
    )
    axes[0].set_title("Temperature")
    plt.colorbar(img1, ax=axes[0])
    
    # Plot velocity magnitude
    velocity = np.sqrt(sim.fluid_state.u[:, :, 0]**2 + sim.fluid_state.u[:, :, 1]**2).T
    img2 = axes[1].imshow(
        velocity,
        extent=[sim.x.min(), sim.x.max(), sim.y.min(), sim.y.max()],
        cmap="viridis",
        origin="lower",
    )
    axes[1].set_title("Velocity Magnitude")
    plt.colorbar(img2, ax=axes[1])
    
    # Plot velocity field with streamlines
    X, Y = np.meshgrid(sim.x, sim.y, indexing='ij')
    u = sim.fluid_state.u[:, :, 0].T
    v = sim.fluid_state.u[:, :, 1].T
    
    axes[2].streamplot(X.T, Y.T, u, v, density=1.5, color='k', linewidth=1)
    axes[2].set_title("Velocity Field")
    
    if title:
        fig.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    return fig