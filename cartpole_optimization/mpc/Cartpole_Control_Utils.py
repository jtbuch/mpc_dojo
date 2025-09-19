import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os 
import do_mpc
import time

class MPCController:
    def __init__(self, horizon=10, dt=0.02, linear=False, recompute_every=1, model_length=0.5, wind_mu=0.0, wind_sigma=0.0):
        self.horizon = horizon
        self.dt = dt
        self.linear = linear
        self.recompute_every = recompute_every
        self.force_mag = 10.0
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        
        # Model parameters (what MPC thinks)
        self.length = model_length
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma

        # Define model
        model_type = "continuous"
        self.model = do_mpc.model.Model(model_type)
        
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        x_dot = self.model.set_variable(var_type='_x', var_name='x_dot', shape=(1,1))
        theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
        theta_dot = self.model.set_variable(var_type='_x', var_name='theta_dot', shape=(1,1))
        u = self.model.set_variable(var_type='_u', var_name='u', shape=(1,1))
        
        if self.linear:
            # Linearized dynamics (θ ≈ 0)
            temp = u / self.total_mass
            thetaacc = (self.gravity * theta - temp) / (
                self.length * (4.0/3.0 - self.masspole/self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc / self.total_mass
        else:
            # Original nonlinear dynamics
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (u + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        self.model.set_rhs('x', x_dot)
        self.model.set_rhs('x_dot', xacc)
        self.model.set_rhs('theta', theta_dot)
        self.model.set_rhs('theta_dot', thetaacc)
        
        self.model.setup()
        
        # Configure MPC
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': self.dt,
            'n_robust': 0,
            'state_discretization': 'discrete' if self.linear else 'collocation',
            'discretization': 'euler' if self.linear else None,
            'collocation_type': 'radau' if not self.linear else None,
            'collocation_deg': 2 if not self.linear else None,
            'collocation_ni': 1 if not self.linear else None,
            'store_full_solution': True,
        }
        self.mpc.set_param(**{k: v for k, v in setup_mpc.items() if v is not None})
        
        self.mpc.set_objective(mterm=theta**2 + x**2, lterm=theta**2 + x**2 + 0.01*u**2)
        self.mpc.bounds['lower','_u','u'] = -self.force_mag
        self.mpc.bounds['upper','_u','u'] = self.force_mag
        
        self.mpc.setup()

    def get_action(self, obs):
        # add wind disturbance only to the pole angle (index 2)
        obs_with_noise = obs.copy()  # Create a copy to avoid modifying original
        wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma)
        obs_with_noise[2] += wind_disturbance  # Only affect pole angle
        
        self.mpc.x0 = np.array(obs_with_noise).reshape(-1, 1)
        self.mpc.set_initial_guess()
        self.mpc.make_step(self.mpc.x0)
        trajectory = self.mpc.data.prediction(('_u',))
        return trajectory

def evaluate_mpc_controllers(horizons, recompute_intervals, results_folder="../results/PerformanceResults/", 
                             episode_length=500, num_episodes=20, seed=42, linear=True, 
                             length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6], wind_mus=[0.0], wind_sigmas=[0.0]):
    """Evaluate MPC with various horizons, recompute intervals, pole length misspecifications, and wind conditions."""
    
    timing = [] 
    true_length = 0.5  # True system pole length
    
    # Check that the smallest recompute interval is less than or equal to the smallest horizon
    if min(recompute_intervals) > min(horizons):
        raise ValueError("The smallest recompute interval must be less than or equal to the smallest horizon.")
    
    os.makedirs(results_folder, exist_ok=True)
    # video_dir = os.path.join("Results", "Videos", "mpc")
    # os.makedirs(video_dir, exist_ok=True)
    
    results_length = {}
    results_states = {}
    
    # Add loops over wind parameters
    for wind_mu in wind_mus:
        for wind_sigma in wind_sigmas:
            for h in horizons:
                for e in recompute_intervals:
                    for ratio in length_ratios:
                        model_length = true_length * ratio
                        start_time = time.time()
                        
                        # Updated key to include wind parameters
                        key = f"h_{h}_e_{e}_ratio_{ratio:.1f}_wmu_{wind_mu:.2f}_wsig_{wind_sigma:.2f}"
                        results_length[key] = []
                        results_states[key] = []
                        
                        # Pass wind parameters to MPC controller
                        mpc = MPCController(horizon=h, recompute_every=e, linear=linear, 
                                          model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma)
                        
                        for ep in range(num_episodes):
                            if ep == 0:
                                env = gym.make("CartPole-v1", render_mode="rgb_array")
                                # ep_video_dir = os.path.join(video_dir, key)
                                # os.makedirs(ep_video_dir, exist_ok=True)
                                # env = gym.wrappers.RecordVideo(env, f"{ep_video_dir}/episode_1.mp4", episode_trigger=lambda _: True)
                            else:
                                env = gym.make("CartPole-v1", render_mode=None)
                            
                            obs, _ = env.reset(seed=seed + ep)
                            length, step = 0, 0
                            done = False
                            states = []

                            while not done and length < episode_length:
                                if step % e == 0:
                                    within_step = 0
                                    trajectory = mpc.get_action(obs).flatten()
                                else:
                                    within_step += 1
                                
                                action = trajectory[min(within_step, len(trajectory)-1)]
                                if action > 0:
                                    action = 1
                                else:
                                    action = 0
                                
                                obs, _, done, _, _ = env.step(action)
                                # Wind effects handled in MPC controller if applicable
                                states.append(obs)
                                step += 1
                                length += 1
                            
                            results_length[key].append(length)
                            states_squared = np.square(states)
                            results_states[key].append(np.sum(states_squared, axis=0))
                            env.close()

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        timing.append({
                            "model": "mpc",
                            "horizon": h,
                            "recompute_interval": e,
                            "length_ratio": ratio,
                            "wind_mu": wind_mu,
                            "wind_sigma": wind_sigma,
                            "evaluation_time_seconds": elapsed_time
                        })

                        # Save results with wind parameters in filename
                        wind_suffix = f"_wmu_{wind_mu:.2f}_wsig_{wind_sigma:.2f}"
                        np.savetxt(f"{results_folder}/mpc_episode_lengths_{key}.csv", results_length[key], delimiter=",")
                        np.savetxt(f"{results_folder}/mpc_integrated_errors_{key}.csv", results_states[key], delimiter=",")

    timing_results_df = pd.DataFrame(timing)
    # Update CSV filename to reflect wind conditions
    wind_desc = "with_wind" if any(mu != 0.0 or sigma != 0.0 for mu in wind_mus for sigma in wind_sigmas) else "no_wind"
    results_csv_path = os.path.join(results_folder, f"Training_times_MPC_{wind_desc}.csv")
    timing_results_df.to_csv(results_csv_path, index=False)
    
    return 


def analyze_performance_results_episode_length(results_folder="../results/PerformanceResults/", 
                                episode_length=500,
                                num_episodes=20,
                                length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6],
                                horizon=20,
                                wind_mu=0.0,
                                wind_sigma=0.0,
                                dt=0.02):
    """
    Reads CSV files from the given results folder and plots results including pole length analysis.
    Now requires specifying wind_mu and wind_sigma levels.
    """
    
    # Get all CSV files - now looking for wind parameters
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    files = [file for file in files if "episode_lengths" in file and "mpc" in file]
    
    results = []
    for file in files:
        try:
            data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
            mean = np.mean(data)
            std = np.std(data)
            
            if "wmu" in file and "wsig" in file:
                # Parse filename with wind: mpc_episode_lengths_h_X_e_Y_ratio_Z_wmu_A_wsig_B.csv
                parts = file.replace("mpc_episode_lengths_", "").replace(".csv", "").split("_")
                file_horizon = int(parts[1])
                recompute = int(parts[3])
                ratio = float(parts[5])
                file_wind_mu = float(parts[7])
                file_wind_sigma = float(parts[9])
            else:
                # Parse old filename format (assume no wind)
                parts = file.replace("mpc_episode_lengths_", "").replace("_no_noise.csv", "").split("_")
                file_horizon = int(parts[1])
                recompute = int(parts[3])
                ratio = float(parts[5])
                file_wind_mu = 0.0
                file_wind_sigma = 0.0
            
            results.append({
                'horizon': file_horizon,  
                'recompute': recompute, 
                'length_ratio': ratio,
                'wind_mu': file_wind_mu,
                'wind_sigma': file_wind_sigma,
                'mean_length': mean,
                'std_length': std
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    
    # Filter data for the specified horizon AND wind parameters
    specified_data = df[
        (df['horizon'] == horizon) & 
        (df['wind_mu'] == wind_mu) & 
        (df['wind_sigma'] == wind_sigma)
    ]
    
    if specified_data.empty:
        print(f"No data found for h={horizon}, wind_mu={wind_mu}, wind_sigma={wind_sigma}")
        return
    
    # Convert to seconds for display
    horizon_sec = horizon * dt
    
    # Create single plot: Length ratio impact for different recompute frequencies
    plt.figure(figsize=(10, 6))
    
    # Get unique recompute values and create color map
    recompute_values = sorted(specified_data['recompute'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(recompute_values)))
    
    for i, recompute in enumerate(recompute_values):
        subset = specified_data[specified_data['recompute'] == recompute].sort_values('length_ratio')
        if len(subset) > 0:
            # Convert recompute frequency to seconds
            recompute_sec = recompute * dt
            label = f'Recompute every {recompute_sec:.2f}s'
            
            # Convert episode lengths to seconds
            mean_length_sec = subset['mean_length'] * dt
            std_length_sec = subset['std_length'] * dt
            
            plt.errorbar(subset['length_ratio'], mean_length_sec, yerr=std_length_sec,
                        marker='o', label=label, capsize=3, linewidth=2, markersize=8, color=colors[i])
    
    # Add horizontal line at max episode length (converted to seconds)
    max_episode_length_sec = episode_length * dt
    plt.axhline(y=max_episode_length_sec, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Max Episode Length')
    
    # Updated title with all parameters
    plt.title(f'MPC Performance: Pole Length Misspecification vs Recompute Frequency\n(h={horizon_sec:.1f}s, wind_μ={wind_mu}, wind_σ={wind_sigma}, N={num_episodes} episodes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Model Length / True Length', fontsize=12)
    plt.ylabel('Mean Episode Length (s) (± SD)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max_episode_length_sec + 4)  # +4 seconds buffer

    # save the plot
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_performance_length_ratio_h{horizon}_wmu{wind_mu}_wsig{wind_sigma}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return

def plot_4d_performance_heatmaps(results_folder="../results/PerformanceResults/", 
                                length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6],
                                wind_mu=0.0,
                                wind_sigma=0.0,
                                num_episodes=20,
                                dt=0.02):
    """
    Create multiple heatmaps showing horizon × recompute performance for each pole error ratio.
    Now requires specifying wind_mu and wind_sigma levels.
    """
    
    # Get all CSV files
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    files = [file for file in files if "episode_lengths" in file and "mpc" in file]
    
    if not files:
        print("No result files found! Make sure you've run the evaluation first.")
        return None
    
    # Read and parse all results
    results = []
    for file in files:
        try:
            data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
            mean = np.mean(data)
            std = np.std(data)
            
            if "wmu" in file and "wsig" in file:
                # Parse filename with wind
                parts = file.replace("mpc_episode_lengths_", "").replace(".csv", "").split("_")
                horizon = int(parts[1])
                recompute = int(parts[3])
                ratio = float(parts[5])
                file_wind_mu = float(parts[7])
                file_wind_sigma = float(parts[9])
            else:
                # Parse old filename format (assume no wind)
                parts = file.replace("mpc_episode_lengths_", "").replace("_no_noise.csv", "").split("_")
                horizon = int(parts[1])
                recompute = int(parts[3])
                ratio = float(parts[5])
                file_wind_mu = 0.0
                file_wind_sigma = 0.0
            
            results.append({
                'horizon': horizon,
                'recompute': recompute, 
                'length_ratio': ratio,
                'wind_mu': file_wind_mu,
                'wind_sigma': file_wind_sigma,
                'mean_length': mean,
                'std_length': std
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    
    # Filter for specified wind parameters
    filtered_data = df[
        (df['wind_mu'] == wind_mu) & 
        (df['wind_sigma'] == wind_sigma)
    ]
    
    if filtered_data.empty:
        print(f"No data found for wind_mu={wind_mu}, wind_sigma={wind_sigma}")
        return
    
    # Get unique values for axes
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute'].unique())
    available_ratios = sorted(filtered_data['length_ratio'].unique())
    
    # Convert to seconds for display
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    # Filter to requested ratios that exist
    plot_ratios = [r for r in length_ratios if r in available_ratios]
    
    print(f"Found data for horizons: {horizons} steps = {[f'{h:.1f}s' for h in horizons_sec]}")
    print(f"Found data for recompute intervals: {recomputes} steps = {[f'{r:.2f}s' for r in recomputes_sec]}")
    print(f"Plotting ratios: {plot_ratios}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(plot_ratios), figsize=(4*len(plot_ratios), 5))
    
    # Handle case with single subplot
    if len(plot_ratios) == 1:
        axes = [axes]
    
    # Global min/max for consistent color scale (convert to seconds for colorbar)
    all_performance_sec = filtered_data['mean_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, ratio in enumerate(plot_ratios):
        ratio_data = filtered_data[filtered_data['length_ratio'] == ratio]
        
        # Create performance matrix: recompute (rows) × horizon (cols)
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = ratio_data[(ratio_data['recompute'] == recompute) & 
                                  (ratio_data['horizon'] == horizon)]
                if len(subset) > 0:
                    # Convert to seconds for display
                    perf_matrix[i, j] = subset['mean_length'].iloc[0] * dt
        
        # Create heatmap
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        # Set ticks and labels with seconds
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])  # Show horizons in seconds
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])  # Show recomputes in seconds
        
        # Labels and title with updated units
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        # Title with status
        title = f'Ratio = {ratio:.1f}'
        if ratio == 1.0:
            title += '\n(Perfect Model)'
        else:
            error_pct = abs(ratio - 1.0) * 100
            direction = "Underestimate" if ratio < 1.0 else "Overestimate"
            title += f'\n({direction} {error_pct:.0f}%)'
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        # Add performance values as text (in seconds)
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    # Use white text for dark colors, black for light colors
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{perf_matrix[i, j]:.1f}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')
    
    # Overall title with wind parameters
    fig.suptitle(f'MPC Performance: Horizon × Recompute × Pole Length Error\n(wind_μ={wind_mu}, wind_σ={wind_sigma}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_4d_performance_heatmaps_wmu{wind_mu}_wsig{wind_sigma}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

def analyze_wind_mu_performance(results_folder="../results/PerformanceResults/", 
                               episode_length=500,
                               num_episodes=20,
                               wind_mus=[0.0, 0.5, 1.0, 1.5, 2.0],
                               horizon=20,
                               length_ratio=1.0,
                               wind_sigma=0.0,
                               dt=0.02):
    """
    Analyze MPC performance vs wind mean (mu) for different recompute frequencies.
    Now requires specifying length_ratio and wind_sigma levels.
    """
    
    # Get CSV files with wind parameters
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    files = [file for file in files if "episode_lengths" in file and "mpc" in file and "wmu" in file]
    
    results = []
    for file in files:
        try:
            data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
            mean = np.mean(data)
            std = np.std(data)
            
            # Parse filename: mpc_episode_lengths_h_X_e_Y_ratio_Z_wmu_A_wsig_B.csv
            parts = file.replace("mpc_episode_lengths_", "").replace(".csv", "").split("_")
            file_horizon = int(parts[1])
            file_recompute = int(parts[3])
            file_ratio = float(parts[5])
            file_wind_mu = float(parts[7])
            file_wind_sigma = float(parts[9])
            
            results.append({
                'horizon': file_horizon,
                'recompute': file_recompute,
                'length_ratio': file_ratio,
                'wind_mu': file_wind_mu,
                'wind_sigma': file_wind_sigma,
                'mean_length': mean,
                'std_length': std
            })
        except:
            continue
    
    if not results:
        print("No wind data files found!")
        return
    
    df = pd.DataFrame(results)
    
    # Filter for specified parameters (fixed horizon, length_ratio, wind_sigma)
    filtered_data = df[
        (df['horizon'] == horizon) & 
        (df['length_ratio'] == length_ratio) &
        (df['wind_sigma'] == wind_sigma)
    ]
    
    if filtered_data.empty:
        print(f"No data found for h={horizon}, ratio={length_ratio}, sigma={wind_sigma}")
        return
    
    # Convert to seconds
    horizon_sec = horizon * dt
    
    plt.figure(figsize=(10, 6))
    
    # Get unique recompute values and create color map (same as original)
    recompute_values = sorted(filtered_data['recompute'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(recompute_values)))
    
    for i, recompute in enumerate(recompute_values):
        subset = filtered_data[filtered_data['recompute'] == recompute].sort_values('wind_mu')
        if len(subset) > 0:
            # Convert recompute frequency to seconds
            recompute_sec = recompute * dt
            label = f'Recompute every {recompute_sec:.2f}s'
            
            # Convert episode lengths to seconds
            mean_length_sec = subset['mean_length'] * dt
            std_length_sec = subset['std_length'] * dt
            
            plt.errorbar(subset['wind_mu'], mean_length_sec, yerr=std_length_sec,
                        marker='o', label=label, capsize=3, linewidth=2, markersize=8, color=colors[i])
    
    # Add horizontal line at max episode length
    max_episode_length_sec = episode_length * dt
    plt.axhline(y=max_episode_length_sec, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Max Episode Length')
    
    plt.title(f'MPC Performance: Wind Mean vs Recompute Frequency\n(h={horizon_sec:.1f}s, length_ratio={length_ratio}, wind_σ={wind_sigma}, N={num_episodes} episodes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Wind Mean (μ)', fontsize=12)
    plt.ylabel('Mean Episode Length (s) (± SD)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max_episode_length_sec + 4)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_mu_analysis_h{horizon}_r{length_ratio:.1f}_s{wind_sigma:.1f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    return 


def analyze_wind_sigma_performance(results_folder="../results/PerformanceResults/", 
                                  episode_length=500,
                                  num_episodes=20,
                                  wind_sigmas=[0.0, 0.5, 1.0, 1.5, 2.0],
                                  horizon=20,
                                  length_ratio=1.0,
                                  wind_mu=0.0,
                                  dt=0.02):
    """
    Analyze MPC performance vs wind std dev (sigma) for different recompute frequencies.
    Now requires specifying length_ratio and wind_mu levels.
    """
    
    # Get CSV files with wind parameters
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    files = [file for file in files if "episode_lengths" in file and "mpc" in file and "wsig" in file]
    
    results = []
    for file in files:
        try:
            data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
            mean = np.mean(data)
            std = np.std(data)
            
            # Parse filename: mpc_episode_lengths_h_X_e_Y_ratio_Z_wmu_A_wsig_B.csv
            parts = file.replace("mpc_episode_lengths_", "").replace(".csv", "").split("_")
            file_horizon = int(parts[1])
            file_recompute = int(parts[3])
            file_ratio = float(parts[5])
            file_wind_mu = float(parts[7])
            file_wind_sigma = float(parts[9])
            
            results.append({
                'horizon': file_horizon,
                'recompute': file_recompute,
                'length_ratio': file_ratio,
                'wind_mu': file_wind_mu,
                'wind_sigma': file_wind_sigma,
                'mean_length': mean,
                'std_length': std
            })
        except:
            continue
    
    if not results:
        print("No wind data files found!")
        return
    
    df = pd.DataFrame(results)
    
    # Filter for specified parameters (fixed horizon, length_ratio, wind_mu)
    filtered_data = df[
        (df['horizon'] == horizon) & 
        (df['length_ratio'] == length_ratio) &
        (df['wind_mu'] == wind_mu)
    ]
    
    if filtered_data.empty:
        print(f"No data found for h={horizon}, ratio={length_ratio}, mu={wind_mu}")
        return
    
    # Convert to seconds
    horizon_sec = horizon * dt
    
    plt.figure(figsize=(10, 6))
    
    # Get unique recompute values and create color map (same as original)
    recompute_values = sorted(filtered_data['recompute'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(recompute_values)))
    
    for i, recompute in enumerate(recompute_values):
        subset = filtered_data[filtered_data['recompute'] == recompute].sort_values('wind_sigma')
        if len(subset) > 0:
            # Convert recompute frequency to seconds
            recompute_sec = recompute * dt
            label = f'Recompute every {recompute_sec:.2f}s'
            
            # Convert episode lengths to seconds
            mean_length_sec = subset['mean_length'] * dt
            std_length_sec = subset['std_length'] * dt
            
            plt.errorbar(subset['wind_sigma'], mean_length_sec, yerr=std_length_sec,
                        marker='o', label=label, capsize=3, linewidth=2, markersize=8, color=colors[i])
    
    # Add horizontal line at max episode length
    max_episode_length_sec = episode_length * dt
    plt.axhline(y=max_episode_length_sec, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Max Episode Length')
    
    plt.title(f'MPC Performance: Wind Std Dev vs Recompute Frequency\n(h={horizon_sec:.1f}s, length_ratio={length_ratio}, wind_μ={wind_mu}, N={num_episodes} episodes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Wind Standard Deviation (σ)', fontsize=12)
    plt.ylabel('Mean Episode Length (s) (± SD)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max_episode_length_sec + 4)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_sigma_analysis_h{horizon}_r{length_ratio:.1f}_m{wind_mu:.1f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    return 


def plot_wind_mu_heatmaps(results_folder="../results/PerformanceResults/", 
                         wind_mus=[0.0, 0.5, 1.0, 1.5, 2.0],
                         length_ratio=1.0,
                         wind_sigma=0.0,
                         num_episodes=20,
                         dt=0.02):
    """
    Create multiple heatmaps showing horizon × recompute performance for each wind_mu level.
    Now requires specifying length_ratio and wind_sigma levels.
    """
    
    # Get CSV files with wind parameters
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    files = [file for file in files if "episode_lengths" in file and "mpc" in file and "wmu" in file]
    
    if not files:
        print("No wind data files found!")
        return None
    
    results = []
    for file in files:
        try:
            data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
            mean = np.mean(data)
            std = np.std(data)
            
            # Parse filename
            parts = file.replace("mpc_episode_lengths_", "").replace(".csv", "").split("_")
            file_horizon = int(parts[1])
            file_recompute = int(parts[3])
            file_ratio = float(parts[5])
            file_wind_mu = float(parts[7])
            file_wind_sigma = float(parts[9])
            
            results.append({
                'horizon': file_horizon,
                'recompute': file_recompute,
                'length_ratio': file_ratio,
                'wind_mu': file_wind_mu,
                'wind_sigma': file_wind_sigma,
                'mean_length': mean,
                'std_length': std
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    
    # Filter for specified parameters (fixed length_ratio and wind_sigma)
    filtered_data = df[
        (df['length_ratio'] == length_ratio) & 
        (df['wind_sigma'] == wind_sigma)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, sigma={wind_sigma}")
        return
    
    # Get unique values for axes
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute'].unique())
    available_wind_mus = sorted(filtered_data['wind_mu'].unique())
    
    # Convert to seconds for display
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    # Filter to requested wind_mus that exist
    plot_wind_mus = [mu for mu in wind_mus if mu in available_wind_mus]
    
    print(f"Found data for horizons: {horizons} steps = {[f'{h:.1f}s' for h in horizons_sec]}")
    print(f"Found data for recompute intervals: {recomputes} steps = {[f'{r:.2f}s' for r in recomputes_sec]}")
    print(f"Plotting wind means: {plot_wind_mus}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(plot_wind_mus), figsize=(4*len(plot_wind_mus), 5))
    
    # Handle case with single subplot
    if len(plot_wind_mus) == 1:
        axes = [axes]
    
    # Global min/max for consistent color scale (convert to seconds)
    all_performance_sec = filtered_data['mean_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, wind_mu in enumerate(plot_wind_mus):
        mu_data = filtered_data[filtered_data['wind_mu'] == wind_mu]
        
        # Create performance matrix: recompute (rows) × horizon (cols)
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = mu_data[(mu_data['recompute'] == recompute) & 
                               (mu_data['horizon'] == horizon)]
                if len(subset) > 0:
                    # Convert to seconds for display
                    perf_matrix[i, j] = subset['mean_length'].iloc[0] * dt
        
        # Create heatmap
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        # Set ticks and labels with seconds
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])
        
        # Labels and title
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        # Title with wind mu value
        title = f'Wind μ = {wind_mu:.1f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        # Add performance values as text (in seconds)
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{perf_matrix[i, j]:.1f}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    
    # Overall title with fixed parameters
    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Mean\n(length_ratio={length_ratio}, wind_σ={wind_sigma}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_mu_heatmaps_r{length_ratio:.1f}_s{wind_sigma:.1f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return 


def plot_wind_sigma_heatmaps(results_folder="../results/PerformanceResults/", 
                            wind_sigmas=[0.0, 0.5, 1.0, 1.5, 2.0],
                            length_ratio=1.0,
                            wind_mu=0.0,
                            num_episodes=20,
                            dt=0.02):
    """
    Create multiple heatmaps showing horizon × recompute performance for each wind_sigma level.
    Now requires specifying length_ratio and wind_mu levels.
    """
    
    # Get CSV files with wind parameters
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    files = [file for file in files if "episode_lengths" in file and "mpc" in file and "wsig" in file]
    
    if not files:
        print("No wind data files found!")
        return None
    
    results = []
    for file in files:
        try:
            data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
            mean = np.mean(data)
            std = np.std(data)
            
            # Parse filename
            parts = file.replace("mpc_episode_lengths_", "").replace(".csv", "").split("_")
            file_horizon = int(parts[1])
            file_recompute = int(parts[3])
            file_ratio = float(parts[5])
            file_wind_mu = float(parts[7])
            file_wind_sigma = float(parts[9])
            
            results.append({
                'horizon': file_horizon,
                'recompute': file_recompute,
                'length_ratio': file_ratio,
                'wind_mu': file_wind_mu,
                'wind_sigma': file_wind_sigma,
                'mean_length': mean,
                'std_length': std
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    
    # Filter for specified parameters (fixed length_ratio and wind_mu)
    filtered_data = df[
        (df['length_ratio'] == length_ratio) & 
        (df['wind_mu'] == wind_mu)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, mu={wind_mu}")
        return
    
    # Get unique values for axes
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute'].unique())
    available_wind_sigmas = sorted(filtered_data['wind_sigma'].unique())
    
    # Convert to seconds for display
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    # Filter to requested wind_sigmas that exist
    plot_wind_sigmas = [sigma for sigma in wind_sigmas if sigma in available_wind_sigmas]
    
    print(f"Found data for horizons: {horizons} steps = {[f'{h:.1f}s' for h in horizons_sec]}")
    print(f"Found data for recompute intervals: {recomputes} steps = {[f'{r:.2f}s' for r in recomputes_sec]}")
    print(f"Plotting wind std devs: {plot_wind_sigmas}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(plot_wind_sigmas), figsize=(4*len(plot_wind_sigmas), 5))
    
    # Handle case with single subplot
    if len(plot_wind_sigmas) == 1:
        axes = [axes]
    
    # Global min/max for consistent color scale (convert to seconds)
    all_performance_sec = filtered_data['mean_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, wind_sigma in enumerate(plot_wind_sigmas):
        sigma_data = filtered_data[filtered_data['wind_sigma'] == wind_sigma]
        
        # Create performance matrix: recompute (rows) × horizon (cols)
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = sigma_data[(sigma_data['recompute'] == recompute) & 
                                  (sigma_data['horizon'] == horizon)]
                if len(subset) > 0:
                    # Convert to seconds for display
                    perf_matrix[i, j] = subset['mean_length'].iloc[0] * dt
        
        # Create heatmap
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        # Set ticks and labels with seconds
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])
        
        # Labels and title
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        # Title with wind sigma value
        title = f'Wind σ = {wind_sigma:.1f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        # Add performance values as text (in seconds)
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{perf_matrix[i, j]:.1f}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')
    
    
    # Overall title with fixed parameters
    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Std Dev\n(length_ratio={length_ratio}, wind_μ={wind_mu}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_sigma_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return