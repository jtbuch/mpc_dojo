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
           
            # Save the action before adding noise
            actions[t] = action            

            obs, reward, done, truncated, info = env.step(action)

            # If the model is discrete keep reward as is, otherwise save obs[1] as reward
            if reward_type == 'discrete':
                reward = reward 
            elif reward_type == 'continuous':
                # Base reward for staying alive
                reward = 1.0
                
                # Angle reward (most important - stay upright)
                angle_reward = np.exp(-5 * obs[1]**2)
                reward += 2.0 * angle_reward
                
                # Position reward (stay centered)
                position_reward = np.exp(-0.5 * obs[1]**2)
                reward += 0.5 * position_reward
                
                # Stability reward (minimize velocities)
                velocity_penalty = 0.1 * (obs[2]**2 + obs[3]**2)
                reward -= velocity_penalty
                
                # Large penalty for falling
                if done:
                    reward -= 10.0
                
                # Bonus for being very stable
                if abs(obs[1]) < 0.1 and abs(obs[3]) < 0.1:
                    reward += 0.5

            # Add observation noise
            obs += rng.normal(obs_noise_mu, obs_noise_sigma)

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
    # Group results by interval and rl_model
    results_with_stats = {}
    termination_with_stats = {}
    
    for result in results_list:
        interval = result['interval']
        n_planning = result['n_planning']
        rl_model = result['config']['rl_model']  # Get RL model from config
        
        # Create nested structure: rl_model -> interval
        if rl_model not in results_with_stats:
            results_with_stats[rl_model] = {}
            termination_with_stats[rl_model] = {}
            
        if interval not in results_with_stats[rl_model]:
            results_with_stats[rl_model][interval] = {
                "n_planning": [],
                "means": [],
                "std_errors": []
            }
            termination_with_stats[rl_model][interval] = {
                "n_planning": [],
                "means": [],
                "std_errors": []
            }
        
        # Add to stats
        results_with_stats[rl_model][interval]["n_planning"].append(n_planning)
        results_with_stats[rl_model][interval]["means"].append(np.mean(result['rewards']))
        results_with_stats[rl_model][interval]["std_errors"].append(stats.sem(result['rewards']))
        
        termination_with_stats[rl_model][interval]["n_planning"].append(n_planning)
        termination_with_stats[rl_model][interval]["means"].append(np.mean(result['terminations']))
        termination_with_stats[rl_model][interval]["std_errors"].append(stats.sem(result['terminations']))
    
    # Sort by n_planning for each rl_model and interval
    for rl_model in results_with_stats:
        for interval in results_with_stats[rl_model]:
            sort_idx = np.argsort(results_with_stats[rl_model][interval]["n_planning"])
            results_with_stats[rl_model][interval]["n_planning"] = [results_with_stats[rl_model][interval]["n_planning"][i] for i in sort_idx]
            results_with_stats[rl_model][interval]["means"] = [results_with_stats[rl_model][interval]["means"][i] for i in sort_idx]
            results_with_stats[rl_model][interval]["std_errors"] = [results_with_stats[rl_model][interval]["std_errors"][i] for i in sort_idx]
            
            termination_with_stats[rl_model][interval]["n_planning"] = [termination_with_stats[rl_model][interval]["n_planning"][i] for i in sort_idx]
            termination_with_stats[rl_model][interval]["means"] = [termination_with_stats[rl_model][interval]["means"][i] for i in sort_idx]
            termination_with_stats[rl_model][interval]["std_errors"] = [termination_with_stats[rl_model][interval]["std_errors"][i] for i in sort_idx]
    
    return results_with_stats, termination_with_stats

def plot_results(results_with_stats, termination_with_stats, config):
    """Plot the results with support for multiple RL models."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    # Define color maps for different RL models
    rl_models = list(results_with_stats.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(rl_models)))
    color_map = dict(zip(rl_models, colors))
    
    # Plot 1: Average Reward
    for rl_model, model_results in results_with_stats.items():
        base_color = color_map[rl_model]
        
        for i, (interval, stats_dict) in enumerate(model_results.items()):
            x = np.array(stats_dict["n_planning"])
            y = np.array(stats_dict["means"])
            yerr = np.array(stats_dict["std_errors"])
            
            # Add jitter based on interval
            jitter = (interval - np.mean(list(model_results.keys()))) * 0.05
            x_jittered = x + jitter
            
            # Create label combining RL model and interval
            label = f'{rl_model}'
            
            # Use different line styles for different intervals of the same model
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            axs[0].errorbar(x_jittered, y, yerr=yerr, marker='o', capsize=4, 
                          label=label, color=base_color, linestyle=linestyle, linewidth=2)
    
    axs[0].set_title('Average Reward')
    axs[0].set_xlabel('n_planning')
    axs[0].set_ylabel('Avg Reward (± SEM)')
    axs[0].set_ylim(-2, 0.1)
    axs[0].set_xlim(0, 100)
    axs[0].grid(True)
    
    # Add legend with title for plot 1
    legend1 = axs[0].legend(title='RL Training', title_fontsize=10)
    legend1.get_title().set_fontweight('bold')
    
    # Plot 2: Termination Step
    for rl_model, model_results in termination_with_stats.items():
        base_color = color_map[rl_model]
        
        for i, (interval, stats_dict) in enumerate(model_results.items()):
            x = np.array(stats_dict["n_planning"])
            y = np.array(stats_dict["means"])
            yerr = np.array(stats_dict["std_errors"])
            
            # Add jitter based on interval
            jitter = (interval - np.mean(list(model_results.keys()))) * 0.05
            x_jittered = x + jitter
            
            # Create label combining RL model and interval
            label = f'{rl_model}'  # Fixed the missing (Int {interval}) part
            
            # Use different line styles for different intervals of the same model
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            axs[1].errorbar(x_jittered, y, yerr=yerr, marker='s', capsize=4, 
                          label=label, color=base_color, linestyle=linestyle, linewidth=2)
    
    axs[1].set_title('Termination Step')
    axs[1].set_xlabel('n_planning')
    axs[1].set_ylabel('Step where |angle| > 1.5 (± SEM)')
    axs[1].set_ylim(0, config['time_steps'] + 5)
    axs[1].set_xlim(0, 100)
    axs[1].grid(True)
    axs[1].axhline(y=config['time_steps'], color='r', linestyle='--', label='Max Time Steps')
    
    # Add legend with title for plot 2
    legend2 = axs[1].legend(title='RL Training', title_fontsize=10)
    legend2.get_title().set_fontweight('bold')
    
    # Update title to show multiple RL models if applicable
    rl_models_str = ', '.join(rl_models) if len(rl_models) <= 3 else f"{len(rl_models)} RL models"
    
    title = (
        f'Reward and Termination Step across Planning & Recompute Intervals\n'
        f'Obs Noise_mu={config["obs_noise_mu"]}, Obs Noise_sigma={config["obs_noise_sigma"]}\n'
        f'Act Noise_mu={config["act_noise_mu"]}, Act Noise_sigma={config["act_noise_sigma"]}\n'
        f'Act cost={config["action_cost"]}, Control model={config["control_model"]}\n'
        f'Plan Width={config["planning_width"]}, Episodes={config["n_episodes"]}, Timesteps={config["time_steps"]}\n'
        f'Value={config["value"]}, RL Models={rl_models_str}'
    )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def run_simulations(config):
    """
    Run simulations based on the given configuration.
    
    Now supports multiple RL models in the configuration.
    
    Args:
        config: Configuration dictionary with experiment parameters
                Can now include 'rl_models' as a list for multiple models
        
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
    
    # Get intervals, planning values, and RL models from config
    intervals_to_run = config['recompute_intervals']
    planning_values_to_run = config['n_planning_values']
    
    # Support both single rl_model and multiple rl_models
    if 'rl_models' in config:
        rl_models_to_run = config['rl_models']
    else:
        rl_models_to_run = [config['rl_model']]
    
    # Print appropriate message
    total_sims = len(intervals_to_run) * len(planning_values_to_run) * len(rl_models_to_run)
    print(f"Starting {total_sims} simulations with {len(intervals_to_run)} intervals, "
          f"{len(planning_values_to_run)} planning values, and {len(rl_models_to_run)} RL models")
    
    # Track start time
    start_time = time.time()
    
    # Run simulations
    for rl_model in rl_models_to_run:
        # Update config with current rl_model
        current_config = config.copy()
        current_config['rl_model'] = rl_model
        
        for interval in intervals_to_run:
            for n_planning in planning_values_to_run:
                if n_planning >= interval:  # Skip invalid configuration
                    # Define model file path
                    model_file = os.path.join(models_dir, f'model_interval{interval}_planning{n_planning}_obsnoise{current_config["obs_noise_sigma"][0]}_obsmu{current_config["obs_noise_mu"][0]}_actnoise{current_config["act_noise_sigma"][0]}_actmu{current_config["act_noise_mu"][0]}_time{timestamp}_model{current_config["control_model"]}_value{current_config["value"]}_rl_model{rl_model}.pkl')
                    
                    # Run the simulation for this configuration
                    result = run_simulation(interval, n_planning, current_config)
                    
                    # Save individual model result
                    with open(model_file, 'wb') as f:
                        pickle.dump(result, f)
                    print(f"Model result saved to {model_file}")
    
    print(f"Simulations completed in {time.time() - start_time:.2f} seconds")
    return timestamp

def analyze_and_plot_results(config=None, timestamp=None):
    """
    Load ALL simulation results from the models directory, compute statistics, and generate plots.
    
    Now supports filtering and plotting multiple RL models.
    
    Args:
        config: Optional dictionary that can include:
            - filtering criteria (e.g., 'obs_noise_mu' to filter results)
            - 'rl_models': list of RL models to include (for filtering)
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
    rl_models_filter = None
    
    if config is not None:
        for key, value in config.items():
            if key == 'rl_models':
                rl_models_filter = value  # List of RL models to include
            elif key in ['obs_noise_mu', 'obs_noise_sigma', 'act_noise_mu', 'act_noise_sigma', 'control_model', 'rl_model', 'value', 'n_episodes', 'time_steps']:
                filter_criteria[key] = value
            else:
                custom_plotting_config[key] = value

    # Filter results based on filter_criteria and rl_models_filter
    filtered_results = []
    for result in results_list:
        match = True
        
        # Check standard filter criteria
        for key, value in filter_criteria.items():
            if not np.array_equal(result['config'][key], value):
                match = False
                break
        
        # Check RL model filter
        if match and rl_models_filter is not None:
            if result['config']['rl_model'] not in rl_models_filter:
                match = False
        
        if match:
            filtered_results.append(result)

    if not filtered_results:
        print("No model results found with the specified configuration.")
        return None, None, None

    # Extract configuration from the first filtered model result
    config_for_plotting = filtered_results[0]['config'].copy()
    
    # Update plotting configuration parameters if not overridden
    config_for_plotting.update(custom_plotting_config)
    
    # Compute statistics
    results_with_stats, termination_with_stats = compute_stats(filtered_results)
    
    # Plot results
    fig = plot_results(results_with_stats, termination_with_stats, config_for_plotting)
    
    return fig, results_with_stats, termination_with_stats