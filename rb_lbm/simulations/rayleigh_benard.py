"""Rayleigh-Benard convection simulation module for ShittyBird."""

from typing import Dict, List, Optional, Any

import jax
import jax.numpy as jnp
import numpy as np
import core as lbm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output, display

from tqdm.rich import trange
import warnings
from tqdm import TqdmExperimentalWarning

from utils.visualization import fig_constructor, fig_updater, plot_state

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
        return (f"{self.name}: min={self.min:.3e}, max={self.max:.3e}, "
                f"mean={self.mean:.3e}, std={self.std:.3e}")


def get_diagnostics(sim: lbm.Simulation) -> List[Diagnostic]:
    """Get diagnostic metrics from simulation state.
    
    Args:
        sim: The simulation object
        
    Returns:
        List of diagnostic objects with simulation metrics
    """
    return [
        Diagnostic("temperature", sim.fluid_state.T),
        Diagnostic("density", sim.fluid_state.rho),
        Diagnostic("velocity_x", sim.fluid_state.u[..., 0]),
        Diagnostic("velocity_y", sim.fluid_state.u[..., 1]),
    ]


class RayleighBenardSimulation:
    """Rayleigh-Benard convection simulation using LBM."""
    
    def __init__(
        self, 
        n: int = 64, 
        pr: float = 0.71, 
        ra: float = 1e7, 
        buoy: float = 0.0001, 
        seed: int = 0
    ) -> None:
        """
        Initialize the Rayleigh-Benard simulation
        
        Args:
            n: Grid resolution (n x n)
            pr: Prandtl number
            ra: Rayleigh number
            buoy: Buoyancy coefficient
            seed: Random seed for initial conditions
        """
        self.n = n
        self.pr = pr
        self.ra = ra
        self.buoy = buoy
        self.seed = seed
        
        # Initialize simulation
        self.setup_simulation()
        
        # Store history
        self.history = []
        self.save_state()
        
    def setup_simulation(self) -> None:
        """Setup the simulation domain, lattice, and boundary conditions"""
        nx = ny = self.n
        
        # Create domain
        domain = lbm.Domain(shape=(nx, ny), bounds=(0.0, 1.0, 0.0, 1.0))
        
        # Calculate physical parameters
        dx = domain.dx
        dt = (self.buoy * dx) ** 0.5
        
        # Collision parameters
        viscosity = (self.pr / self.ra) ** 0.5 * dt / dx**2
        kappa = viscosity / self.pr
        
        # Calculate timescales
        self.convection_timescale = 1
        self.dt = dt
        
        # Set relaxation parameters
        omegas = {
            "FluidLattice": 1.0 / (3 * viscosity + 0.5),
            "ThermalLattice": 1.0 / (3 * kappa + 0.5),
        }
        
        # Instantiate the lattice
        lattice = lbm.ThermalFluidLattice(
            fluid_stencil=lbm.D2Q9,
            thermal_stencil=lbm.D2Q5,
            buoyancy=jnp.array([0, self.buoy]),
        )
        
        # Create simulation
        self.sim = lbm.Simulation(domain, lattice, omegas)
        key = jax.random.PRNGKey(self.seed)
        
        # Set initial conditions
        self.sim.set_initial_conditions(
            rho=jnp.ones((nx, ny, 1)),
            T=jax.random.uniform(key, (nx, ny, 1), minval=-0.05, maxval=0.05),
            u=jnp.zeros((nx, ny, 2)),
        )
        
        # Set boundary conditions
        # Fluid: No-slip at top and bottom walls
        self.sim.set_boundary_conditions(
            lbm.BounceBackBoundary(
                "walls", self.sim.bottom | self.sim.top
            ),
            "FluidLattice",
        )
        
        # Thermal: Fixed temperature at top and bottom walls
        self.sim.set_boundary_conditions(lbm.InletBoundary("bot", self.sim.bottom), "ThermalLattice")
        self.sim.set_boundary_conditions(lbm.InletBoundary("top", self.sim.top), "ThermalLattice")
        
        # Hot bottom wall, cold top wall
        self.sim.update_boundary_condition("bot", {"m": 0.5}, "ThermalLattice")
        self.sim.update_boundary_condition("top", {"m": -0.5}, "ThermalLattice")
        
        # Current time
        self.current_step = 0
        self.current_time = 0.0
    
    def save_state(self) -> None:
        """Save the current state of the simulation"""
        # Convert JAX arrays to numpy for storage
        state = {
            'step': self.current_step,
            'time': self.current_time,
            'temperature': np.array(self.sim.fluid_state.T),
            'velocity': np.array(self.sim.fluid_state.u),
            'density': np.array(self.sim.fluid_state.rho)
        }
        self.history.append(state)
    
    def run(
        self, 
        steps: int, 
        save_frequency: int = 50, 
        display_frequency: Optional[int] = None, 
        progress_bar: bool = True
    ) -> None:
        """Run the simulation for the specified number of steps
        
        Args:
            steps: Number of steps to simulate
            save_frequency: How often to save states (in steps)
            display_frequency: How often to display progress (in steps)
            progress_bar: Whether to show a progress bar
        """
        if display_frequency is None:
            display_frequency = save_frequency
            
        if progress_bar:
            iterator = trange(steps)
        else:
            iterator = range(steps)
            
        for i in iterator:
            self.sim.step()
            self.current_step += 1
            self.current_time += self.dt
            
            # Check for NaNs
            if jnp.isnan(self.sim.fluid_state.T).any():
                print(f"NaNs detected, stopping simulation at step {self.current_step}")
                break
                
            # Save state
            if i % save_frequency == 0:
                self.save_state()
                
            # Display current state
            if display_frequency > 0 and i % display_frequency == 0:
                if not progress_bar:  # Only clear output if not using progress bar
                    clear_output(wait=True)
                fig = plot_state(self.sim, f"Step: {self.current_step}, Time: {self.current_time:.3f}")
                display(fig)
                plt.close(fig)
                diags = get_diagnostics(self.sim)
                for diag in diags:
                    print(f"  {diag}")
    
    def plot_state_at_index(self, idx: int = -1) -> plt.Figure:
        """Plot a specific state from the simulation history
        
        Args:
            idx: Index of the state to plot
            
        Returns:
            Matplotlib figure
        """
        state = self.history[idx]
        time = state['time']
        step = state['step']

        # Create an updated fluid state
        new_fluid_state = self.sim.fluid_state._replace(
            T=jnp.array(state['temperature']),
            u=jnp.array(state['velocity']),
            rho=jnp.array(state['density'])
        )
        
        # Create a temporary simulation with this state
        temp_sim = self.sim.copy_with_fluid_state(new_fluid_state)
        
        # Generate the plot
        fig = plot_state(temp_sim, f"Step: {step}, Time: {time:.3f}")
        return fig

    def create_matplotlib_animation(self, output_file: str = "rayleigh_benard.mp4", fps: int = 15) -> None:
        """Create an animation using matplotlib.animation
        
        Args:
            output_file: Path to save the animation
            fps: Frames per second for the animation
        """
        # Use the first history state to set up the initial simulation state
        first_state = self.history[0]
        new_fluid_state = self.sim.fluid_state._replace(
            T=jnp.array(first_state['temperature']),
            u=jnp.array(first_state['velocity']),
            rho=jnp.array(first_state['density'])
        )
        temp_sim = self.sim.copy_with_fluid_state(new_fluid_state)
        
        # Create the initial figure
        fig_data = fig_constructor(temp_sim)
        fig = fig_data["figure"]
        ax = fig_data["axes"]
        
        # Define the update function for each frame
        def update(frame_index):
            state = self.history[frame_index]
            # Create a temporary simulation copy with updated fluid state
            new_fluid_state = self.sim.fluid_state._replace(
                T=jnp.array(state['temperature']),
                u=jnp.array(state['velocity']),
                rho=jnp.array(state['density'])
            )
            temp_sim = self.sim.copy_with_fluid_state(new_fluid_state)
            # Update the figure
            fig_updater(fig_data, temp_sim)
            # Update title with step and time
            step = state.get('step', frame_index)
            time_val = state.get('time', 0)
            ax.set_title(f"Step: {step}  Time: {time_val:.3f}")
            # Return the updated image for blitting
            return (fig_data["image"],)
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.history), blit=True, interval=1000/fps)
        
        # Save the animation
        if output_file.lower().endswith('.gif'):
            ani.save(output_file, writer='imagemagick', fps=fps)
        else:
            ani.save(output_file, writer='ffmpeg', fps=fps)
        
        plt.close(fig)
        print(f"Animation saved to {output_file}")


def rayleigh_benard(
    n: int = 96, 
    pr: float = 0.71, 
    ra: float = 1e6, 
    buoy: float = 0.0001,
    steps: int = 5000,
    run_time: Optional[float] = None,
    save_frequency: int = 50,
    display_frequency: int = 200,
    save_video: bool = True,
    video_fps: int = 15,
    seed: int = 0,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run a Rayleigh-Benard convection simulation.
    
    Args:
        n: Grid size (n x n)
        pr: Prandtl number
        ra: Rayleigh number
        buoy: Buoyancy parameter
        steps: Number of steps to simulate
        run_time: Total simulation time (overrides steps if provided)
        save_frequency: How often to save states
        display_frequency: How often to display simulation state
        save_video: Whether to save video output
        video_fps: Frames per second for video output
        seed: Random seed for initial conditions
        output_file: Path for output video file
        
    Returns:
        Dictionary with simulation results
    """
    print(f"Starting Rayleigh-Benard simulation:")
    print(f"  Grid: {n}x{n}")
    print(f"  Prandtl number (Pr): {pr}")
    print(f"  Rayleigh number (Ra): {ra}")
    print(f"  Buoyancy: {buoy}")
    
    # Create and initialize the simulation
    rb_sim = RayleighBenardSimulation(n=n, pr=pr, ra=ra, buoy=buoy, seed=seed)
    
    # If run_time is provided, calculate steps
    if run_time is not None:
        convection_timescale = 1
        steps = int(run_time * convection_timescale / rb_sim.dt)
    
    # Run the simulation
    rb_sim.run(
        steps=steps, 
        save_frequency=save_frequency, 
        display_frequency=display_frequency, 
        progress_bar=True
    )
    
    # Create animation if requested
    if save_video:
        if output_file is None:
            output_file = f"rb_n{n}_pr{pr:.1e}_ra{ra:.1e}_buoy{buoy:.1e}.mp4"
        rb_sim.create_matplotlib_animation(output_file=output_file, fps=video_fps)
    
    # Return results
    return {
        "simulation": rb_sim,
        "history": rb_sim.history,
        "parameters": {
            "n": n,
            "pr": pr,
            "ra": ra,
            "buoy": buoy,
            "steps": steps,
            "seed": seed
        }
    }