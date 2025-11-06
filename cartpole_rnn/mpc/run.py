import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
import os 
import do_mpc
from casadi import vertcat
from scipy.linalg import solve_discrete_are
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..')))
from mpc.Cartpole_Control_Utils import *
import time

# Parse command line arguments
length_ratio = float(sys.argv[1])
recompute_interval = int(sys.argv[2])
wind_mu = float(sys.argv[3])
wind_sigma = float(sys.argv[4])
world_model = sys.argv[5]
controller = sys.argv[6]

print(f"Running simulation with length_ratio={length_ratio} and recompute_interval={recompute_interval} and wind_mu={wind_mu} and wind_sigma={wind_sigma} and world_model={world_model}")

# Create 'Results' folder if it doesn't exist
results_folder = "../results/PerformanceResults/"

# Make sure the folders exist
for folder in [results_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Lists for timing results
results_records = []

#------------------------------------------------------------------------------------------------------------------------------------------------
# Train/evaluate the MPC model
#------------------------------------------------------------------------------------------------------------------------------------------------
start_time = time.time()
mpc_results = evaluate_mpc_controllers(
        controller=controller,
        world_model=world_model,
        horizons=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        recompute_intervals=[recompute_interval],
        dt = 0.02,
        length_ratios=[length_ratio],
        wind_mus=[wind_mu],
        wind_sigmas=[wind_sigma],
        results_folder="../results/PerformanceResults/",
        episode_length=2000,
        num_episodes=5,
        init_angles=[0.0, 0.10, 0.15, 0.20])


