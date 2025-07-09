import numpy as np
from mpc.configs import *
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import time
import os
import torch
from stable_baselines3 import TD3



def simulate_one_model(bird, n_episodes, time_steps, recompute_interval, obs_noise_mu=np.array([1.0, 0.0, 0.0, 0.0]), obs_noise_sigma=np.array([0.0, 0.0, 0.0, 0.0]), act_noise_mu=np.array([0.0]), act_noise_sigma=np.array([0.0]), reward_type='continuous', action_cost = 0.0):
    """Simulate bird."""

    simulation_data = {}
    np.random.seed(2024)

    for epi in range(n_episodes):
        episode_seed = 54 + epi
        rng = np.random.default_rng(episode_seed)  # One RNG per episode

        env = gym.make("InvertedPendulum-v5", render_mode="rgb_array", reset_noise_scale=0.01) 
        obs, _ = env.reset(seed=episode_seed)

        env.action_space.seed(episode_seed)
        env.observation_space.seed(episode_seed)

        # writer = imageio.get_writer("output.gif", fps=30)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        observations = np.zeros((time_steps, obs_dim))
        actions = np.zeros((time_steps, act_dim))
        rewards = np.zeros(time_steps)

        # We'll initialize reward components later after the first env.step() gives us `info`
        reward_components = {}
        reward_names = None

        observations[0] = obs  # store initial observation at time 0

        for t in range(time_steps):

            if t % recompute_interval == 0: # This is the step at which we recompute the action
                        within_step = 0
                        action_trajectory = bird.mujoco_policy(env, obs)
                        action_trajectory = np.asarray(action_trajectory).flatten()
                        action = action_trajectory[within_step]
            else:
                        within_step += 1
                        action = action_trajectory[within_step]
           
            # Add action noise
            action += rng.normal(act_noise_mu, act_noise_sigma)

            # action += np.random.default_rng(episode_seed + t + 1).normal(act_noise_mu, act_noise_sigma)
            
            # Save the action before adding noise
            actions[t] = action            

            obs, reward, done, truncated, info = env.step(action)

            # If the model is discrete keep reward as is, otherwise save obs[1] as reward
            if reward_type == 'discrete':
                reward = reward - action_cost * np.sum(np.square(action[0]))  
            elif reward_type == 'continuous':
                reward = -np.abs(obs[1]) -action_cost * np.sum(np.square(action[0]))
                # print(f"Observation at time {t}: {obs[1]}")
                # print(f"Action cost at time {t}: {np.sum(np.square(action[0]))}")
                # print(f"Reward at time {t}: {reward}")
                # print(f"Action at time {t}: {action[0]}")

            # Add observation noise
            obs += rng.normal(obs_noise_mu, obs_noise_sigma)

            # obs += np.random.default_rng(episode_seed + t + 2).normal(obs_noise_mu, obs_noise_sigma)

            # Save the true observation before adding noise
            observations[t] = obs
            rewards[t] = reward


            # Initialize reward_components dict once we see the reward names
            if t == 0 and info:
                reward_names = list(info.keys())
                reward_components = {
                    name: np.zeros(time_steps) for name in reward_names
                }

            # Record reward components if they exist
            for name in reward_components:
                reward_components[name][t] = info.get(name, 0.0)

            # if epi == n_episodes - 1:
                # frame = env.render()
                # writer.append_data(np.asarray(frame).astype(np.uint8))

        # writer.close()
        env.close()

        episode_data = {}

        for i in range(obs_dim):
            episode_data[f"state_{i+1}"] = observations[:, i]
        for i in range(act_dim):
            episode_data[f"action_{i+1}"] = actions[:, i]

        episode_data["reward"] = rewards
        for name, values in reward_components.items():
            episode_data[f"reward_{name}"] = values
        episode_data["reward_names"] = reward_names

        simulation_data[f"episode_{epi+1}"] = episode_data

    return simulation_data

def ensure_dir_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def run_simulation(interval, n_planning, config):
    """Run a simulation with the given parameters."""
    print(f"Running simulation with recompute_interval={interval}, n_planning={n_planning}")
    
    # Create agent with specified parameters
    bird = MPCShittyBird(
        n_actions=10,
        planning_width=config['planning_width'],
        n_planning=n_planning,
        reward_type=config['reward_type'],
        model_type=config['model_type'],
        action_cost=config['action_cost'],
        control_model=config['control_model'],
        value=config['value'],
        rl_model=config['rl_model']

    )
    
    # Run simulation
    simulation_data = simulate_one_model(
        bird,
        n_episodes=config['n_episodes'],
        time_steps=config['time_steps'],
        recompute_interval=interval,
        obs_noise_mu=config['obs_noise_mu'],
        obs_noise_sigma=config['obs_noise_sigma'],
        act_noise_mu=config['act_noise_mu'],
        act_noise_sigma=config['act_noise_sigma'],
        reward_type=config['reward_type'],
        action_cost=config['action_cost']
    )
    
    # Process results
    episode_rewards = []
    episode_terminations = []
    
    for episode_data in simulation_data.values():
        angle = episode_data["state_2"]
        
        # Compute reward
        reward = -np.mean(np.abs(angle))
        episode_rewards.append(reward)
        
        # Compute termination step
        exceed_indices = np.where(np.abs(angle) > 1.5)[0]
        if len(exceed_indices) > 0:
            termination_step = int(exceed_indices[0])
        else:
            termination_step = len(angle)
        episode_terminations.append(termination_step)
    
    return {
        'interval': interval,
        'n_planning': n_planning,
        'rewards': episode_rewards,
        'terminations': episode_terminations,
        'raw_data': simulation_data,  # Keep raw data for detailed analysis
        'config': config  # Save config with results
    }

def load_models_data(models_dir):
    """Load all model results from the models directory."""
    results_list = []
    
    # Find all model result files
    for filename in os.listdir(models_dir):
        if filename.startswith('model_interval') and filename.endswith('.pkl'):
            filepath = os.path.join(models_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    model_result = pickle.load(f)
                    results_list.append(model_result)
                    print(f"Loaded model from {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return results_list

def compute_stats(results_list):
    """Compute statistics from a list of simulation results."""
    # Group results by interval
    results_with_stats = {}
    termination_with_stats = {}
    
    for result in results_list:
        interval = result['interval']
        n_planning = result['n_planning']
        
        if interval not in results_with_stats:
            results_with_stats[interval] = {
                "n_planning": [],
                "means": [],
                "std_errors": []
            }
            termination_with_stats[interval] = {
                "n_planning": [],
                "means": [],
                "std_errors": []
            }
        
        # Add to stats
        results_with_stats[interval]["n_planning"].append(n_planning)
        results_with_stats[interval]["means"].append(np.mean(result['rewards']))
        results_with_stats[interval]["std_errors"].append(stats.sem(result['rewards']))
        
        termination_with_stats[interval]["n_planning"].append(n_planning)
        termination_with_stats[interval]["means"].append(np.mean(result['terminations']))
        termination_with_stats[interval]["std_errors"].append(stats.sem(result['terminations']))
    
    # Sort by n_planning for each interval
    for interval in results_with_stats:
        sort_idx = np.argsort(results_with_stats[interval]["n_planning"])
        results_with_stats[interval]["n_planning"] = [results_with_stats[interval]["n_planning"][i] for i in sort_idx]
        results_with_stats[interval]["means"] = [results_with_stats[interval]["means"][i] for i in sort_idx]
        results_with_stats[interval]["std_errors"] = [results_with_stats[interval]["std_errors"][i] for i in sort_idx]
        
        termination_with_stats[interval]["n_planning"] = [termination_with_stats[interval]["n_planning"][i] for i in sort_idx]
        termination_with_stats[interval]["means"] = [termination_with_stats[interval]["means"][i] for i in sort_idx]
        termination_with_stats[interval]["std_errors"] = [termination_with_stats[interval]["std_errors"][i] for i in sort_idx]
    
    return results_with_stats, termination_with_stats

def plot_results(results_with_stats, termination_with_stats, config):
    """Plot the results."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    # Plot 1: Average Reward
    for interval, stats_dict in results_with_stats.items():
        x = np.array(stats_dict["n_planning"])
        y = np.array(stats_dict["means"])
        yerr = np.array(stats_dict["std_errors"])
        jitter = (interval - np.mean(list(results_with_stats.keys()))) * 0.05
        x_jittered = x + jitter
        axs[0].errorbar(x_jittered, y, yerr=yerr, marker='o', capsize=4, label=f'Interval {interval}')
    
    axs[0].set_title('Average Reward')
    axs[0].set_xlabel('n_planning')
    axs[0].set_ylabel('Avg Reward (± SEM)')
    axs[0].set_ylim(-2,0.1)
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: Termination Step
    for interval, stats_dict in termination_with_stats.items():
        x = np.array(stats_dict["n_planning"])
        y = np.array(stats_dict["means"])
        yerr = np.array(stats_dict["std_errors"])
        jitter = (interval - np.mean(list(termination_with_stats.keys()))) * 0.05
        x_jittered = x + jitter
        axs[1].errorbar(x_jittered, y, yerr=yerr, marker='s', capsize=4, label=f'Interval {interval}')
    
    axs[1].set_title('Termination Step')
    axs[1].set_xlabel('n_planning')
    axs[1].set_ylabel('Step where |angle| > 1.5 (± SEM)')
    axs[1].set_ylim(0, config['time_steps'] + 5)
    axs[1].grid(True)
    axs[1].legend()
    axs[1].axhline(y=config['time_steps'], color='r', linestyle='--', label='Max Time Steps')
    
    title = (
        f'Reward and Termination Step across Planning & Recompute Intervals\n'
        f'Obs Noise_mu={config["obs_noise_mu"]}, Obs Noise_sigma={config["obs_noise_sigma"]}\n'
        f'Act Noise_mu={config["act_noise_mu"]}, Act Noise_sigma={config["act_noise_sigma"]}\n'
        f'Act cost={config["action_cost"]}, Sampling method={config["control_model"]}\n'
        f'Plan Width={config["planning_width"]}, Episodes={config["n_episodes"]}, Timesteps={config["time_steps"]}\n'
        f'Value={config["value"]}' + (f', RL Model={config["rl_model"]}' if config["value"] == 'rl' else '')
    )

    plt.suptitle(title, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def run_simulations(config):
    """
    Run simulations based on the given configuration.
    
    Automatically detects whether to run a single simulation or multiple based on the config.
    
    Args:
        config: Configuration dictionary with experiment parameters
        
    Returns:
        timestamp: Timestamp string for the simulation run
    """
    # Setup directory structure
    results_dir = os.path.join('..', 'results')
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir_exists(results_dir)
    ensure_dir_exists(models_dir)
    
    # Create timestamp for file naming
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Get intervals and planning values from config
    intervals_to_run = config['recompute_intervals']
    planning_values_to_run = config['n_planning_values']
    
    # Print appropriate message based on number of simulations
    if len(intervals_to_run) == 1 and len(planning_values_to_run) == 1:
        interval = intervals_to_run[0]
        n_planning = planning_values_to_run[0]
        print(f"Running single simulation with interval={interval}, n_planning={n_planning}")
    else:
        print(f"Starting simulations with {len(intervals_to_run)} intervals and {len(planning_values_to_run)} planning values")
    
    # Track start time
    start_time = time.time()
    
    # Run simulations
    for interval in intervals_to_run:
        for n_planning in planning_values_to_run:
            if n_planning >= interval:  # Skip invalid configuration
                # Define model file path
                model_file = os.path.join(models_dir, f'model_interval{interval}_planning{n_planning}_obsnoise{config["obs_noise_sigma"][0]}_obsmu{config["obs_noise_mu"][0]}_actnoise{config["act_noise_sigma"][0]}_actmu{config["act_noise_mu"][0]}_time{timestamp}_model{config["control_model"]}_value{config["value"]}_rl_model{config["rl_model"]}.pkl')
                
                # Run the simulation for this configuration
                result = run_simulation(interval, n_planning, config)
                
                # Save individual model result
                with open(model_file, 'wb') as f:
                    pickle.dump(result, f)
                print(f"Model result saved to {model_file}")
    
    print(f"Simulations completed in {time.time() - start_time:.2f} seconds")
    return timestamp



def analyze_and_plot_results(config=None, timestamp=None):
    """
    Load ALL simulation results from the models directory, compute statistics, and generate plots.
    
    Args:
        config: Optional dictionary that can include:
            - filtering criteria (e.g., 'obs_noise_mu' to filter results)
            - plotting configuration (e.g., 'planning_width', 'reward_type')
        timestamp: Optional timestamp for naming output files (if None, generate a new one)
        
    Returns:
        fig: The matplotlib figure object
        results_with_stats: Statistics for reward data
        termination_with_stats: Statistics for termination data
    """
    # Setup directory paths
    results_dir = os.path.join('..', 'results')
    models_dir = os.path.join(results_dir, 'models')
    ensure_dir_exists(results_dir)
    
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Load all model results from the models directory
    results_list = load_models_data(models_dir)
    
    if not results_list:
        print("No model results found in the models directory.")
        return None, None, None

    # Initialize filter criteria and custom plotting config
    filter_criteria = {}
    custom_plotting_config = {}
    
    if config is not None:
        for key, value in config.items():
            # Determine if the key is for filtering or for configuration
            if key in ['obs_noise_mu', 'obs_noise_sigma', 'act_noise_mu', 'act_noise_sigma', 'control_model', 'rl_model', 'value', 'n_episodes', 'time_steps']:  # Extend as needed
                filter_criteria[key] = value
            else:
                custom_plotting_config[key] = value

    # Filter results based on filter_criteria
    if filter_criteria:
        filtered_results = []
        for result in results_list:
            match = True
            for key, value in filter_criteria.items():
                if not np.array_equal(result['config'][key], value):
                    match = False
                    break
            if match:
                filtered_results.append(result)
    else:
        filtered_results = results_list

    if not filtered_results:
        print("No model results found with the specified configuration.")
        return None, None, None

    # Extract configuration from the first filtered model result
    config = filtered_results[0]['config'].copy()
    
    # Update plotting configuration parameters if not overridden
    config.update(custom_plotting_config)

    # Save combined results
    # combined_results_file = os.path.join(results_dir, f"combined_results_{timestamp}.pkl")
    # with open(combined_results_file, 'wb') as f:
    #     pickle.dump(filtered_results, f)
    # print(f"Combined results saved to {combined_results_file}")
    
    # Compute statistics
    results_with_stats, termination_with_stats = compute_stats(filtered_results)
    
    # Plot results
    fig = plot_results(results_with_stats, termination_with_stats, config)
    
    # Save plot
    plot_filename = os.path.join(results_dir, f"simulation_plot_{timestamp}.png")
    fig.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")
    
    return fig, results_with_stats, termination_with_stats

