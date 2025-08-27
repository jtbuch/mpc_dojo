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
    fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    
    # Get RL models in the order they appear in config
    if 'rl_models' in config:
        rl_models = config['rl_models']
    else:
        rl_models = list(results_with_stats.keys())
    
    # Convert to numeric values for proper ordering
    def extract_numeric_value(model_str):
        """Extract numeric value from strings like '50k', '500k', etc."""
        # Remove 'k' from the end and convert to numeric
        if model_str.endswith('k'):
            return int(model_str[:-1])  # Remove 'k' and convert to int
        return int(model_str)  # In case there's no 'k'
    
    # Sort rl_models by numeric value
    rl_models_sorted = sorted(rl_models, key=extract_numeric_value)
    
    # Use viridis colormap for nicer colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(rl_models_sorted)))
    color_map = dict(zip(rl_models_sorted, colors))
    
    # Add jitter amount
    jitter_amount = 1.0  # Adjust this value as needed
    
    # Plot 1: Average Reward
    for rl_model in rl_models_sorted:  # Use sorted list
        if rl_model not in results_with_stats:
            continue
        model_results = results_with_stats[rl_model]
        base_color = color_map[rl_model]
        
        for i, (interval, stats_dict) in enumerate(model_results.items()):
            x = np.array(stats_dict["n_planning"])
            y = np.array(stats_dict["means"])
            yerr = np.array(stats_dict["std_errors"])
            
            # Add jitter to x coordinates
            x_jittered = x + np.random.normal(0, jitter_amount, len(x))
            
            label = f'{rl_model}'
            
            # Use different line styles for different intervals of the same model
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            axs[0].errorbar(x_jittered, y, yerr=yerr, marker='o', capsize=4, 
                          label=label, color=base_color, linestyle=linestyle, linewidth=2)
    
    axs[0].set_title('Average Reward')
    axs[0].set_xlabel('n_planning')
    axs[0].set_ylabel('Avg Reward (± SEM)')
    axs[0].set_ylim(-2, 0.1)
    axs[0].set_xlim(-5, 105)
    axs[0].grid(True)
    
    # Plot 2: Termination Step
    for rl_model in rl_models_sorted:  # Use sorted list
        if rl_model not in termination_with_stats:
            continue
        model_results = termination_with_stats[rl_model]
        base_color = color_map[rl_model]
        
        for i, (interval, stats_dict) in enumerate(model_results.items()):
            x = np.array(stats_dict["n_planning"])
            y = np.array(stats_dict["means"])
            yerr = np.array(stats_dict["std_errors"])
            
            # Add jitter to x coordinates
            x_jittered = x + np.random.normal(0, jitter_amount, len(x))
            
            label = f'{rl_model}'
            
            # Use different line styles for different intervals of the same model
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            axs[1].errorbar(x_jittered, y, yerr=yerr, marker='s', capsize=4, 
                          label=label, color=base_color, linestyle=linestyle, linewidth=2)
    
    axs[1].set_title('Termination Step')
    axs[1].set_xlabel('n_planning')
    axs[1].set_ylabel('Step where |angle| < 1.5 (± SEM)')
    axs[1].set_ylim(-50, 1050)
    axs[1].set_xlim(-5, 105)
    axs[1].grid(True)
    
    # Add max time steps line (without label to exclude from legend)
    axs[1].axhline(y=config['time_steps'], color='r', linestyle='--')
    
    # Remove individual legends from subplots
    axs[0].legend().remove() if axs[0].get_legend() else None
    axs[1].legend().remove() if axs[1].get_legend() else None
    
    # Create legend in the numerically sorted order
    legend_handles = []
    legend_labels = []
    
    # For each RL model in sorted order, create a representative handle
    for rl_model in rl_models_sorted:
        if rl_model in results_with_stats or rl_model in termination_with_stats:
            # Create a simple line with the model's color for the legend
            import matplotlib.lines as mlines
            legend_handle = mlines.Line2D([], [], color=color_map[rl_model], 
                                        marker='o', linestyle='-', linewidth=2,
                                        markersize=6)
            legend_handles.append(legend_handle)
            legend_labels.append(rl_model)
    
    # Add the legend at the bottom
    fig.legend(legend_handles, legend_labels, 
              title='RL Training', title_fontsize=10,
              loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(legend_labels), 50))
    
    # Update title to show multiple RL models if applicable
    rl_models_str = ', '.join(rl_models_sorted) if len(rl_models_sorted) <= 3 else f"{len(rl_models_sorted)} RL models"
    
    title = (
        f'Reward and Termination Step across Planning & Recompute Intervals\n'
        f'Obs Noise_mu={config["obs_noise_mu"]}, Obs Noise_sigma={config["obs_noise_sigma"]}\n'
        f'Act Noise_mu={config["act_noise_mu"]}, Act Noise_sigma={config["act_noise_sigma"]}\n'
        f'Act cost={config["action_cost"]}, Control model={config["control_model"]}\n'
        f'Plan Width={config["planning_width"]}, Episodes={config["n_episodes"]}, Timesteps={config["time_steps"]}\n'
        f'Value={config["value"]}, RL Models={rl_models_str}'
    )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjusted to make room for bottom legend
    return fig

def plot_rl_models(results_with_stats, termination_with_stats, config):
    """Plot bar plots comparing RL models (n_planning=1 only)."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Get RL models in the order they appear in config
    if 'rl_models' in config:
        rl_models = config['rl_models']
    else:
        rl_models = list(results_with_stats.keys())
    
    # Convert to numeric values for proper ordering
    def extract_numeric_value(model_str):
        """Extract numeric value from strings like '50k', '500k', etc."""
        if model_str.endswith('k'):
            return int(model_str[:-1])
        return int(model_str)
    
    # Sort rl_models by numeric value
    rl_models_sorted = sorted(rl_models, key=extract_numeric_value)
    
    # Use viridis colormap for nicer colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(rl_models_sorted)))
    color_map = dict(zip(rl_models_sorted, colors))
    
    # Extract data for n_planning=1 only
    reward_data = []
    reward_errors = []
    termination_data = []
    termination_errors = []
    model_labels = []
    
    for rl_model in rl_models_sorted:
        # Find data for n_planning=1 in reward results
        if rl_model in results_with_stats:
            model_results = results_with_stats[rl_model]
            for interval, stats_dict in model_results.items():
                n_planning_values = np.array(stats_dict["n_planning"])
                # Find index where n_planning=1
                idx = np.where(n_planning_values == 1)[0]
                if len(idx) > 0:
                    reward_data.append(stats_dict["means"][idx[0]])
                    reward_errors.append(stats_dict["std_errors"][idx[0]])
                    break
            else:
                # If n_planning=1 not found, skip this model
                continue
        else:
            continue
            
        # Find data for n_planning=1 in termination results
        if rl_model in termination_with_stats:
            model_results = termination_with_stats[rl_model]
            for interval, stats_dict in model_results.items():
                n_planning_values = np.array(stats_dict["n_planning"])
                # Find index where n_planning=1
                idx = np.where(n_planning_values == 1)[0]
                if len(idx) > 0:
                    termination_data.append(stats_dict["means"][idx[0]])
                    termination_errors.append(stats_dict["std_errors"][idx[0]])
                    break
            else:
                # If we added reward data but can't find termination data, remove the reward data
                reward_data.pop()
                reward_errors.pop()
                continue
        else:
            # If we added reward data but no termination data exists, remove the reward data
            reward_data.pop()
            reward_errors.pop()
            continue
            
        model_labels.append(rl_model)
    
    # Create bar positions
    x_pos = np.arange(len(model_labels))
    
    # Plot 1: Average Reward Bar Plot
    bars1 = axs[0].bar(x_pos, reward_data, yerr=reward_errors, 
                       color=[color_map[model] for model in model_labels],
                       capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    
    axs[0].set_title('Average Reward (RL Models, n_planning=1)', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Avg Reward (± SEM)', fontsize=12)
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(model_labels, rotation=45, ha='right')
    axs[0].grid(True, alpha=0.3)
    # Fixed y-axis limits to match original plot
    axs[0].set_ylim(-2, 0.1)
    
    # Plot 2: Termination Step Bar Plot
    bars2 = axs[1].bar(x_pos, termination_data, yerr=termination_errors,
                       color=[color_map[model] for model in model_labels],
                       capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    
    axs[1].set_title('Termination Step (RL Models, n_planning=1)', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('RL Models', fontsize=12)
    axs[1].set_ylabel('Step where |angle| < 1.5 (± SEM)', fontsize=12)
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(model_labels, rotation=45, ha='right')
    axs[1].grid(True, alpha=0.3)
    # Fixed y-axis limits to match original plot
    axs[1].set_ylim(-50, 1050)
    
    # Add max time steps line
    axs[1].axhline(y=config['time_steps'], color='r', linestyle='--', 
                   linewidth=2, label=f'Max timesteps ({config["time_steps"]})')
    axs[1].legend()
    
    # Add value labels on top of bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Reward values on top of bars
        height1 = bar1.get_height()
        axs[0].text(bar1.get_x() + bar1.get_width()/2., height1 + reward_errors[i],
                   f'{reward_data[i]:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Termination values on top of bars
        height2 = bar2.get_height()
        axs[1].text(bar2.get_x() + bar2.get_width()/2., height2 + termination_errors[i],
                   f'{termination_data[i]:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Create overall title
    title = (
        f'RL Models Comparison (Pure RL: n_planning=1)\n'
        f'Obs Noise: μ={config["obs_noise_mu"]}, σ={config["obs_noise_sigma"]} | '
        f'Act Noise: μ={config["act_noise_mu"]}, σ={config["act_noise_sigma"]}\n'
        f'Action Cost={config["action_cost"]} | Control Model={config["control_model"]} | '
        f'Episodes={config["n_episodes"]} | Timesteps={config["time_steps"]}'
    )
    
    plt.suptitle(title, fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
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
        fig_main: The main matplotlib figure object (original plot)
        fig_rl: The RL models comparison figure object (new barplot)
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
        return None, None, None, None

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
        return None, None, None, None

    # Extract configuration from the first filtered model result
    config_for_plotting = filtered_results[0]['config'].copy()
    
    # Update plotting configuration parameters if not overridden
    config_for_plotting.update(custom_plotting_config)
    
    # Add rl_models list to config_for_plotting if it was provided
    if rl_models_filter is not None:
        config_for_plotting['rl_models'] = rl_models_filter
    
    # Compute statistics
    results_with_stats, termination_with_stats = compute_stats(filtered_results)
    
    # Create both plots
    fig_main = plot_results(results_with_stats, termination_with_stats, config_for_plotting)
    fig_rl = plot_rl_models(results_with_stats, termination_with_stats, config_for_plotting)
    
    return fig_main, fig_rl, results_with_stats, termination_with_stats

def analyze_and_plot_noise_levels(config=None, timestamp=None):
    """
    Load simulation results and create plots comparing different noise levels.
    
    Args:
        config: Dictionary that should include:
            - 'obs_noise_sigma_levels': np.array of different noise sigma values to compare
            - 'obs_noise_mu_levels': np.array of corresponding noise mu values 
            - 'rl_models': list of RL models to include
            - other filtering criteria
        timestamp: Optional timestamp for naming output files
        
    Returns:
        fig: The matplotlib figure object
        all_results_with_stats: Dictionary containing statistics for all noise levels
        all_termination_with_stats: Dictionary containing termination statistics for all noise levels
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

    # Extract noise levels from config
    obs_noise_sigma_levels = config.get('obs_noise_sigma_levels', np.array([0.0]))
    obs_noise_mu_levels = config.get('obs_noise_mu_levels', np.array([0.0]))
    
    # Ensure mu and sigma arrays have same length
    if len(obs_noise_mu_levels) != len(obs_noise_sigma_levels):
        obs_noise_mu_levels = np.full_like(obs_noise_sigma_levels, obs_noise_mu_levels[0])
    
    # Store results for each noise level
    all_results_with_stats = {}
    all_termination_with_stats = {}
    
    # Process each noise level
    for i, (mu_level, sigma_level) in enumerate(zip(obs_noise_mu_levels, obs_noise_sigma_levels)):
        # Initialize filter criteria for this noise level
        filter_criteria = {}
        rl_models_filter = None
        custom_plotting_config = {}
        
        # Set noise parameters for this level
        filter_criteria['obs_noise_mu'] = np.array([mu_level, mu_level, mu_level*0.1, mu_level*0.1])
        filter_criteria['obs_noise_sigma'] = np.array([sigma_level, sigma_level, sigma_level*0.1, sigma_level*0.1])
        
        # Add other filter criteria from config
        if config is not None:
            for key, value in config.items():
                if key == 'rl_models':
                    rl_models_filter = value
                elif key in ['act_noise_mu', 'act_noise_sigma', 'control_model', 'rl_model', 'value', 'n_episodes', 'time_steps']:
                    filter_criteria[key] = value
                elif key not in ['obs_noise_mu_levels', 'obs_noise_sigma_levels']:
                    custom_plotting_config[key] = value

        # Filter results for this noise level
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
        
        if filtered_results:
            # Compute statistics for this noise level using the same function
            results_with_stats, termination_with_stats = compute_stats(filtered_results)
            noise_label = f"σ={sigma_level:.1f}"
            all_results_with_stats[noise_label] = results_with_stats
            all_termination_with_stats[noise_label] = termination_with_stats
            print(f"Found {len(filtered_results)} results for noise level σ={sigma_level:.1f}")
        else:
            print(f"No results found for noise level σ={sigma_level:.1f}")
    
    if not all_results_with_stats:
        print("No results found for any noise level.")
        return None, None, None
    
    # Get config for plotting from the first result
    if results_list:
        config_for_plotting = results_list[0]['config'].copy()
        config_for_plotting.update(custom_plotting_config)
        if rl_models_filter is not None:
            config_for_plotting['rl_models'] = rl_models_filter
    else:
        config_for_plotting = config
    
    # Create the plot
    fig1 = plot_noise_comparison(all_results_with_stats, all_termination_with_stats, config_for_plotting)
    fig2 = plot_noise_planning_comparison(all_results_with_stats, all_termination_with_stats, config_for_plotting)
    
    return fig1, fig2, all_results_with_stats, all_termination_with_stats


def plot_noise_comparison(all_results_with_stats, all_termination_with_stats, config):
    """
    Create plots comparing different noise levels, following the same style as plot_results.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    
    # Get noise levels and sort them numerically
    noise_levels = list(all_results_with_stats.keys())
    
    # Sort noise levels by their sigma values
    def extract_sigma_value(noise_str):
        # Extract sigma value from strings like 'σ=0.4'
        return float(noise_str.split('=')[1])
    
    noise_levels_sorted = sorted(noise_levels, key=extract_sigma_value)
    
    # Use viridis colormap for different noise levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels_sorted)))
    color_map = dict(zip(noise_levels_sorted, colors))
    
    # Get RL models to plot (assuming all noise levels have same RL models)
    first_noise_level = list(all_results_with_stats.values())[0]
    if 'rl_models' in config:
        rl_models = config['rl_models']
    else:
        rl_models = list(first_noise_level.keys())
    
    # Convert to numeric values for proper ordering
    def extract_numeric_value(model_str):
        """Extract numeric value from strings like '50k', '500k', etc."""
        if model_str.endswith('k'):
            return int(model_str[:-1])
        return int(model_str)
    
    rl_models_sorted = sorted(rl_models, key=extract_numeric_value)
    
    # Add jitter amount
    jitter_amount = 1.0
    
    # Plot 1: Average Reward
    for noise_level in noise_levels_sorted:
        if noise_level not in all_results_with_stats:
            continue
            
        results_with_stats = all_results_with_stats[noise_level]
        base_color = color_map[noise_level]
        
        # Aggregate data across all RL models for this noise level
        all_x = []
        all_y = []
        all_yerr = []
        
        for rl_model in rl_models_sorted:
            if rl_model not in results_with_stats:
                continue
            model_results = results_with_stats[rl_model]
            
            for i, (interval, stats_dict) in enumerate(model_results.items()):
                x = np.array(stats_dict["n_planning"])
                y = np.array(stats_dict["means"])
                yerr = np.array(stats_dict["std_errors"])
                
                all_x.extend(x)
                all_y.extend(y)
                all_yerr.extend(yerr)
        
        if all_x:  # Only plot if we have data
            x_array = np.array(all_x)
            y_array = np.array(all_y)
            yerr_array = np.array(all_yerr)
            
            # Sort by x values
            sort_idx = np.argsort(x_array)
            x_sorted = x_array[sort_idx]
            y_sorted = y_array[sort_idx]
            yerr_sorted = yerr_array[sort_idx]
            
            # Add jitter
            x_jittered = x_sorted + np.random.normal(0, jitter_amount, len(x_sorted))
            
            axs[0].errorbar(x_jittered, y_sorted, yerr=yerr_sorted, marker='o', capsize=4, 
                          label=noise_level, color=base_color, linestyle='-', linewidth=2)
    
    axs[0].set_title('Average Reward')
    axs[0].set_xlabel('n_planning')
    axs[0].set_ylabel('Avg Reward (± SEM)')
    axs[0].set_ylim(-2, 0.1)
    axs[0].set_xlim(-5, 105)
    axs[0].grid(True)
    
    # Plot 2: Termination Step
    for noise_level in noise_levels_sorted:
        if noise_level not in all_termination_with_stats:
            continue
            
        termination_with_stats = all_termination_with_stats[noise_level]
        base_color = color_map[noise_level]
        
        # Aggregate data across all RL models for this noise level
        all_x = []
        all_y = []
        all_yerr = []
        
        for rl_model in rl_models_sorted:
            if rl_model not in termination_with_stats:
                continue
            model_results = termination_with_stats[rl_model]
            
            for i, (interval, stats_dict) in enumerate(model_results.items()):
                x = np.array(stats_dict["n_planning"])
                y = np.array(stats_dict["means"])
                yerr = np.array(stats_dict["std_errors"])
                
                all_x.extend(x)
                all_y.extend(y)
                all_yerr.extend(yerr)
        
        if all_x:  # Only plot if we have data
            x_array = np.array(all_x)
            y_array = np.array(all_y)
            yerr_array = np.array(all_yerr)
            
            # Sort by x values
            sort_idx = np.argsort(x_array)
            x_sorted = x_array[sort_idx]
            y_sorted = y_array[sort_idx]
            yerr_sorted = yerr_array[sort_idx]
            
            # Add jitter
            x_jittered = x_sorted + np.random.normal(0, jitter_amount, len(x_sorted))
            
            axs[1].errorbar(x_jittered, y_sorted, yerr=yerr_sorted, marker='s', capsize=4, 
                          label=noise_level, color=base_color, linestyle='-', linewidth=2)
    
    axs[1].set_title('Termination Step')
    axs[1].set_xlabel('n_planning')
    axs[1].set_ylabel('Step where |angle| < 1.5 (± SEM)')
    axs[1].set_ylim(-50, 1050)
    axs[1].set_xlim(-5, 105)
    axs[1].grid(True)
    
    # Add max time steps line
    if 'time_steps' in config:
        axs[1].axhline(y=config['time_steps'], color='r', linestyle='--')
    
    # Remove individual legends from subplots and create a shared legend
    axs[0].legend().remove() if axs[0].get_legend() else None
    axs[1].legend().remove() if axs[1].get_legend() else None
    
    # Create legend for noise levels
    legend_handles = []
    legend_labels = []
    
    for noise_level in noise_levels_sorted:
        import matplotlib.lines as mlines
        legend_handle = mlines.Line2D([], [], color=color_map[noise_level], 
                                    marker='o', linestyle='-', linewidth=2,
                                    markersize=6)
        legend_handles.append(legend_handle)
        legend_labels.append(noise_level)
    
    # Add the legend at the bottom
    fig.legend(legend_handles, legend_labels, 
              title='Noise Levels', title_fontsize=10,
              loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(legend_labels), 4))
    
    # Create title
    rl_models_str = ', '.join(rl_models_sorted) if len(rl_models_sorted) <= 3 else f"{len(rl_models_sorted)} RL models"
    
    title = (
        f'Reward and Termination Step across Planning & Noise Levels\n'
        f'Act Noise_mu={config.get("act_noise_mu", "N/A")}, Act Noise_sigma={config.get("act_noise_sigma", "N/A")}\n'
        f'Act cost={config.get("action_cost", "N/A")}, Control model={config.get("control_model", "N/A")}\n'
        f'Plan Width={config.get("planning_width", "N/A")}, Episodes={config.get("n_episodes", "N/A")}, Timesteps={config.get("time_steps", "N/A")}\n'
        f'Value={config.get("value", "N/A")}, RL Models={rl_models_str}'
    )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_noise_planning_comparison(all_results_with_stats, all_termination_with_stats, config):
    """
    Create plots with noise on x-axis and n_planning as color, following the same style as other plot functions.
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    
    # Get noise levels and sort them numerically
    noise_levels = list(all_results_with_stats.keys())
    
    # Sort noise levels by their sigma values
    def extract_sigma_value(noise_str):
        # Extract sigma value from strings like 'σ=0.4'
        return float(noise_str.split('=')[1])
    
    noise_levels_sorted = sorted(noise_levels, key=extract_sigma_value)
    noise_values = [extract_sigma_value(noise) for noise in noise_levels_sorted]
    
    # Get all unique n_planning values across all noise levels and RL models
    all_n_planning = set()
    for noise_level in noise_levels_sorted:
        if noise_level in all_results_with_stats:
            results_with_stats = all_results_with_stats[noise_level]
            for rl_model, model_results in results_with_stats.items():
                for interval, stats_dict in model_results.items():
                    all_n_planning.update(stats_dict["n_planning"])
    
    # Filter n_planning values if specified in config
    if 'n_planning_filter' in config:
        all_n_planning = set(config['n_planning_filter']).intersection(all_n_planning)
    
    n_planning_sorted = sorted(list(all_n_planning))
    
    # Use viridis colormap for different n_planning values
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_planning_sorted)))
    color_map = dict(zip(n_planning_sorted, colors))
    
    # Add jitter amount for x-axis
    jitter_amount = 0.02
    
    # Plot 1: Average Reward
    for n_planning in n_planning_sorted:
        reward_means = []
        reward_errors = []
        noise_x_values = []
        
        for noise_level in noise_levels_sorted:
            if noise_level not in all_results_with_stats:
                continue
                
            results_with_stats = all_results_with_stats[noise_level]
            noise_value = extract_sigma_value(noise_level)
            
            # Aggregate data across all RL models for this noise level and n_planning
            planning_rewards = []
            planning_errors = []
            
            for rl_model, model_results in results_with_stats.items():
                for interval, stats_dict in model_results.items():
                    n_planning_values = np.array(stats_dict["n_planning"])
                    # Find indices where n_planning matches
                    indices = np.where(n_planning_values == n_planning)[0]
                    
                    for idx in indices:
                        planning_rewards.append(stats_dict["means"][idx])
                        planning_errors.append(stats_dict["std_errors"][idx])
            
            if planning_rewards:  # Only add if we have data
                # Calculate mean and standard error across all instances
                mean_reward = np.mean(planning_rewards)
                # Combine standard errors (approximate)
                combined_error = np.sqrt(np.sum(np.array(planning_errors)**2)) / len(planning_errors)
                
                reward_means.append(mean_reward)
                reward_errors.append(combined_error)
                noise_x_values.append(noise_value)
        
        if reward_means:  # Only plot if we have data
            x_array = np.array(noise_x_values)
            y_array = np.array(reward_means)
            yerr_array = np.array(reward_errors)
            
            # Add jitter
            x_jittered = x_array + np.random.normal(0, jitter_amount, len(x_array))
            
            axs[0].errorbar(x_jittered, y_array, yerr=yerr_array, marker='o', capsize=4, 
                          label=f'n_planning={n_planning}', color=color_map[n_planning], 
                          linestyle='-', linewidth=2)
    
    axs[0].set_title('Average Reward')
    axs[0].set_xlabel('Observation Noise (σ)')
    axs[0].set_ylabel('Avg Reward (± SEM)')
    axs[0].set_ylim(-2, 0.1)
    axs[0].grid(True)
    
    # Plot 2: Termination Step
    for n_planning in n_planning_sorted:
        termination_means = []
        termination_errors = []
        noise_x_values = []
        
        for noise_level in noise_levels_sorted:
            if noise_level not in all_termination_with_stats:
                continue
                
            termination_with_stats = all_termination_with_stats[noise_level]
            noise_value = extract_sigma_value(noise_level)
            
            # Aggregate data across all RL models for this noise level and n_planning
            planning_terminations = []
            planning_errors = []
            
            for rl_model, model_results in termination_with_stats.items():
                for interval, stats_dict in model_results.items():
                    n_planning_values = np.array(stats_dict["n_planning"])
                    # Find indices where n_planning matches
                    indices = np.where(n_planning_values == n_planning)[0]
                    
                    for idx in indices:
                        planning_terminations.append(stats_dict["means"][idx])
                        planning_errors.append(stats_dict["std_errors"][idx])
            
            if planning_terminations:  # Only add if we have data
                # Calculate mean and standard error across all instances
                mean_termination = np.mean(planning_terminations)
                # Combine standard errors (approximate)
                combined_error = np.sqrt(np.sum(np.array(planning_errors)**2)) / len(planning_errors)
                
                termination_means.append(mean_termination)
                termination_errors.append(combined_error)
                noise_x_values.append(noise_value)
        
        if termination_means:  # Only plot if we have data
            x_array = np.array(noise_x_values)
            y_array = np.array(termination_means)
            yerr_array = np.array(termination_errors)
            
            # Add jitter
            x_jittered = x_array + np.random.normal(0, jitter_amount, len(x_array))
            
            axs[1].errorbar(x_jittered, y_array, yerr=yerr_array, marker='s', capsize=4, 
                          label=f'n_planning={n_planning}', color=color_map[n_planning], 
                          linestyle='-', linewidth=2)
    
    axs[1].set_title('Termination Step')
    axs[1].set_xlabel('Observation Noise (σ)')
    axs[1].set_ylabel('Step where |angle| < 1.5 (± SEM)')
    axs[1].set_ylim(-50, 1050)
    axs[1].grid(True)
    
    # Add max time steps line
    if 'time_steps' in config:
        axs[1].axhline(y=config['time_steps'], color='r', linestyle='--')
    
    # Remove individual legends from subplots and create a shared legend
    axs[0].legend().remove() if axs[0].get_legend() else None
    axs[1].legend().remove() if axs[1].get_legend() else None
    
    # Create legend for n_planning values
    legend_handles = []
    legend_labels = []
    
    for n_planning in n_planning_sorted:
        import matplotlib.lines as mlines
        legend_handle = mlines.Line2D([], [], color=color_map[n_planning], 
                                    marker='o', linestyle='-', linewidth=2,
                                    markersize=6)
        legend_handles.append(legend_handle)
        legend_labels.append(f'n_planning={n_planning}')
    
    # Add the legend at the bottom
    fig.legend(legend_handles, legend_labels, 
              title='Planning Steps', title_fontsize=10,
              loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=min(len(legend_labels), 4))
    
    # Get RL models info for title
    if 'rl_models' in config:
        rl_models = config['rl_models']
        rl_models_str = ', '.join(rl_models) if len(rl_models) <= 3 else f"{len(rl_models)} RL models"
    else:
        rl_models_str = "Multiple RL models"
    
    # Create title
    title = (
        f'Reward and Termination Step across Noise Levels & Planning Steps\n'
        f'Act Noise_mu={config.get("act_noise_mu", "N/A")}, Act Noise_sigma={config.get("act_noise_sigma", "N/A")}\n'
        f'Act cost={config.get("action_cost", "N/A")}, Control model={config.get("control_model", "N/A")}\n'
        f'Plan Width={config.get("planning_width", "N/A")}, Episodes={config.get("n_episodes", "N/A")}, Timesteps={config.get("time_steps", "N/A")}\n'
        f'Value={config.get("value", "N/A")}, RL Models={rl_models_str}'
    )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig