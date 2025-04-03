#!/usr/bin/env python
"""
Example script demonstrating Rayleigh-Benard convection simulation.

This example sets up and runs a Rayleigh-Benard convection simulation
with different parameters to explore flow regimes.
"""

import argparse

from simulations import rayleigh_benard
from utils import is_in_colab


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Rayleigh-Benard convection simulation"
    )
    
    parser.add_argument(
        "--grid", "-g", type=int, default=64,
        help="Grid size (n x n)"
    )
    parser.add_argument(
        "--prandtl", "-p", type=float, default=0.71,
        help="Prandtl number (0.71 for air, 7 for water)"
    )
    parser.add_argument(
        "--rayleigh", "-r", type=float, default=1e6,
        help="Rayleigh number (> 1708 for convection)"
    )
    parser.add_argument(
        "--buoyancy", "-b", type=float, default=0.0001,
        help="Buoyancy coefficient"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=5000,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output filename for video"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Disable video generation"
    )
    
    return parser.parse_args()


def main():
    """Run the simulation with command line parameters."""
    args = parse_args()
    
    print("""
Rayleigh-Benard Simulation Parameters:
--------------------------------------
Grid size:        {grid} x {grid}
Prandtl number:   {prandtl}
Rayleigh number:  {rayleigh}
Buoyancy:         {buoyancy}
Steps:            {steps}
Save video:       {save_video}
Random seed:      {seed}
Environment:      {env}
    """.format(
        grid=args.grid,
        prandtl=args.prandtl,
        rayleigh=args.rayleigh,
        buoyancy=args.buoyancy,
        steps=args.steps,
        save_video=not args.no_video,
        seed=args.seed,
        env="Google Colab" if is_in_colab() else "Local"
    ))
    
    # Run the simulation
    result = rayleigh_benard(
        n=args.grid,
        pr=args.prandtl,
        ra=args.rayleigh,
        buoy=args.buoyancy,
        steps=args.steps,
        save_video=not args.no_video,
        output_file=args.output,
        seed=args.seed
    )
    
    # Print final simulation statistics
    sim = result["simulation"]
    final_state = sim.history[-1]
    print("\nSimulation completed:")
    print(f"  Final time: {final_state['time']:.3f}")
    print(f"  Total steps: {final_state['step']}")
    
    return result


if __name__ == "__main__":
    main()