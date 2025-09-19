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
length_ratio = int(sys.argv[1])
recompute_interval = int(sys.argv[2])

print(f"Running simulation with length_ratio={length_ratio} and recompute_interval={recompute_interval}")

# Create 'Results' folder if it doesn't exist
models_folder = "../results/Trained_Models/"
results_folder = "../results/PerformanceResults/"
videos_folder = "../results/Videos/"

# Make sure the folders exist
for folder in [models_folder, results_folder, videos_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Lists for timing results
results_records = []

#------------------------------------------------------------------------------------------------------------------------------------------------
# Train/evaluate the MPC model
#------------------------------------------------------------------------------------------------------------------------------------------------
start_time = time.time()
mpc_results = evaluate_mpc_controllers(
        horizons=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        recompute_intervals=[recompute_interval],
        length_ratios=[length_ratio],
        results_folder="../results/PerformanceResults/",
        episode_length=5,
        num_episodes=2,
        linear=False,
    )