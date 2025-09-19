import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('..')))
from mpc.configs import *
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio


# Parse command line arguments
recompute_interval = int(sys.argv[1])
n_planning = int(sys.argv[2])
obs_noise = float(sys.argv[3])
obs_mu = float(sys.argv[4])
act_noise = float(sys.argv[5])
act_mu = float(sys.argv[6])
control_model = sys.argv[7]  # 'rl' or 'predictive'
value = sys.argv[8]  # 'env' or 'rl'
rl_model = sys.argv[9]  # Name of the RL model to use

print(f"Running simulation with recompute_interval={recompute_interval}, n_planning={n_planning}, obs_noise={obs_noise}, obs_mu={obs_mu}, act_noise={act_noise}, act_mu={act_mu}")

# Define configuration
config = {
    'planning_width': 200,
    'reward_type': 'continuous',
    'obs_noise_mu': np.array([obs_mu, obs_mu, obs_mu*0.1, obs_mu*0.1]),
    'obs_noise_sigma': np.array([obs_noise, obs_noise, obs_noise*0.1, obs_noise*0.1]),
    'act_noise_mu': np.array([act_mu]),
    'act_noise_sigma': np.array([act_noise]),
    'n_episodes': 5,
    'time_steps': 3000,
    'action_cost': 0.2,
    'control_model': control_model,
    'recompute_intervals': [recompute_interval],  # Use the command line argument
    'n_planning_values': [n_planning],             # Use the command line argument
    'value': value,
    'rl_model': rl_model  # Use the command line argument
}

# Run the simulation
run_simulations(config)

# No need to analyze results here as we'll do that separately
print(f"Simulation completed with recompute_interval={recompute_interval}, n_planning={n_planning}")
