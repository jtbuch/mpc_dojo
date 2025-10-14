import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os 
import do_mpc
import time
import pickle
import mujoco

class MPCController:
    def __init__(self, horizon=10, dt=0.02, linear=False, recompute_every=1, model_length=0.5, wind_mu=0.0, wind_sigma=0.0, action_space='continuous'):
        self.action_space = action_space
        self.horizon = horizon
        self.dt = dt
        self.linear = linear
        self.recompute_every = recompute_every
        self.force_mag = 3.0
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        
        self.length = model_length
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma

        model_type = "continuous"
        self.model = do_mpc.model.Model(model_type)
        
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        x_dot = self.model.set_variable(var_type='_x', var_name='x_dot', shape=(1,1))
        theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
        theta_dot = self.model.set_variable(var_type='_x', var_name='theta_dot', shape=(1,1))
        u = self.model.set_variable(var_type='_u', var_name='u', shape=(1,1))
        
        if self.linear:
            temp = u / self.total_mass
            thetaacc = (self.gravity * theta - temp) / (
                self.length * (4.0/3.0 - self.masspole/self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc / self.total_mass
        else:
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
        self.mpc.settings.supress_ipopt_output()
        self.mpc.set_param(**{k: v for k, v in setup_mpc.items() if v is not None})
 
        self.mpc.set_objective(
            mterm=10*theta**2 + x**2 + theta_dot**2 + x_dot**2,
            lterm=10*theta**2 + x**2 + theta_dot**2 + x_dot**2 + 0.1*u**2
        )

        self.mpc.set_rterm(u=0.1)
    
        self.mpc.bounds['lower','_u','u'] = -self.force_mag
        self.mpc.bounds['upper','_u','u'] = self.force_mag

        # self.mpc.bounds['lower','_x','x'] = -3.0  # Cart position limits
        # self.mpc.bounds['upper','_x','x'] = 3.0
        
        self.mpc.setup()

    def get_action(self, obs, env):
        obs_with_noise = obs.copy()
        wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma)
        obs_with_noise[2] += wind_disturbance
        self.mpc.x0 = np.array(obs_with_noise).reshape(-1, 1)
        self.mpc.set_initial_guess()
        self.mpc.make_step(self.mpc.x0)
        trajectory = self.mpc.data.prediction(('_u',))
        return trajectory

class SamplingController:
    def __init__(self, controller='predictive',horizon=10, dt=0.02, linear=False, recompute_every=1, model_length=0.5, wind_mu=0.0, wind_sigma=0.0, action_space="continuous"):
        self.action_space = action_space
        self.horizon = horizon
        self.dt = dt
        self.linear = linear
        self.recompute_every = recompute_every
        
        self.length = model_length
        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma
        self.control_model = controller

    def generate_action_trajectories(self, env, obs):
        """Generate action trajectories based on control model."""
        self.n_candidate_trajectories = 200
        actions = np.zeros([self.n_candidate_trajectories, self.horizon, 1])
        
        if self.control_model=='predictive':
            # Shift nominal trajectory
            if self.horizon > 1:
                self.nominal_trajectory = np.vstack([
                    self.nominal_trajectory[1:],
                    np.zeros((1, 1))
                ])           
            # First candidate is nominal
            actions[0] = self.nominal_trajectory
            
            # Add noise for other candidates
            if self.n_candidate_trajectories > 1:
                noise_std = 0.1 * (6) # env.action_space.high - env.action_space.low
                for i in range(1, self.n_candidate_trajectories):
                    noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                    actions[i] = np.clip(
                        self.nominal_trajectory + noise,
                        -3.0, 3.0 #  env.action_space.low,env.action_space.high
                    )
        
        elif self.control_model == 'random':
            # Random shooting
            if self.action_space == 'discrete':
                actions = np.random.choice(
                    [0, 1], 
                    size=(self.n_candidate_trajectories, self.horizon, 1)
                )
            elif self.action_space == 'continuous':
                actions = np.random.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=(self.n_candidate_trajectories, self.horizon, 1)
                )

        return actions
    
    def evaluate_trajectories(self, env, initial_obs, actions):
        """Evaluate all action trajectories."""
        observations = np.zeros([self.n_candidate_trajectories, self.horizon, *env.observation_space.shape])
        discrete_rewards = np.zeros([self.n_candidate_trajectories, self.horizon])
        done_flags = np.zeros([self.n_candidate_trajectories, self.horizon], dtype=bool)

        # Save the current state
        if self.action_space == 'continuous':
            saved_qpos = env.unwrapped.data.qpos.copy()
            saved_qvel = env.unwrapped.data.qvel.copy()
        elif self.action_space == 'discrete':
            saved_state = env.unwrapped.state.copy()
            saved_qpos = saved_state[[0, 2]] 
            saved_qvel = saved_state[[1, 3]]  
        

        for i in range(self.n_candidate_trajectories):
            self.restore_state(env, saved_qpos, saved_qvel)

            if self.action_space == 'discrete':
                env.unwrapped.length = self.length * env.unwrapped.length
            elif self.action_space == 'continuous':
                env.unwrapped.model.geom('cpole').size[1] = self.length * env.unwrapped.model.geom('cpole').size[1]

            for step in range(self.horizon):
                action = actions[i, step]
                    
                next_obs, reward, done, truncated, info = env.step(action)
                
                actions[i, step] = action  # Store actual action used
                observations[i, step] = next_obs
                discrete_rewards[i, step] = reward
                done_flags[i, step] = done or truncated
                        
        return observations, discrete_rewards, done_flags
    
    def get_action(self, obs, env):
        """Handle trajectory-based methods (random, predictive)."""
        
        # Initialize nominal trajectory for predictive methods
        if self.control_model=='predictive':
            if not hasattr(self, 'nominal_trajectory'):
                    self.nominal_trajectory = np.zeros((self.horizon, 1))       
        
        # Save the current state
        if self.action_space == 'continuous':
            saved_qpos = env.unwrapped.data.qpos.copy()
            saved_qvel = env.unwrapped.data.qvel.copy()
        elif self.action_space == 'discrete':
            saved_state = env.unwrapped.state.copy()
            saved_qpos = saved_state[[0, 2]] 
            saved_qvel = saved_state[[1, 3]]  

        # Generate action trajectories
        actions = self.generate_action_trajectories(env, obs)
        
        # Evaluate trajectories
        observations, discrete_rewards, done_flags = self.evaluate_trajectories(env, obs, actions)
        
        # Calculate rewards and select best
        cumulative_rewards = self.calculate_trajectory_rewards(env, actions, observations, discrete_rewards, done_flags)
        
        # Restore state and return best trajectory
        self.restore_state(env, saved_qpos, saved_qvel)
        
        best_idx = np.argmax(cumulative_rewards)
        best_trajectory = actions[best_idx]
        
        # Update nominal trajectory for predictive methods
        self.nominal_trajectory = best_trajectory
        
        return best_trajectory
    
    def restore_state(self, env, qpos, qvel):
        """Restore environment state."""
        if self.action_space == 'continuous':
            env.unwrapped.data.qpos[:] = qpos
            env.unwrapped.data.qvel[:] = qvel
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        elif self.action_space == 'discrete':
            # For CartPole-v1: reconstruct state from qpos and qvel
            state = np.concatenate([qpos, qvel])  # [cart_pos, pole_angle, cart_vel, pole_angular_vel]
            env.unwrapped.state = state

    def calculate_trajectory_rewards(self, env, actions, observations, discrete_rewards, done_flags):
        """Calculate rewards for trajectories."""

        # Your simple quadratic reward function
        x = observations[:, :, 0]        # cart position
        theta = observations[:, :, 2]    # pole angle  
        x_dot = observations[:, :, 1]    # cart velocity
        theta_dot = observations[:, :, 3] # pole angular velocity
                
        # Quadratic costs (matching MPC cost matrices)
        position_cost = x**2
        angle_cost = 10.0 * theta**2
        velocity_cost = 0.1 * (x_dot**2 + theta_dot**2)
                
        total_cost = position_cost + angle_cost + velocity_cost
        reward = -total_cost
                
        return reward.sum(axis=1)

def evaluate_mpc_controllers(controller, horizons, recompute_intervals, results_folder="../results/PerformanceResults/", 
                             episode_length=500, num_episodes=20, seed=42, linear=True, 
                             length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6], wind_mus=[0.0], wind_sigmas=[0.0], init_angles=[0.0],action_space='discrete'):
    """Evaluate MPC with various parameters and save each configuration separately."""
    
    true_length = 0.5
    
    if min(recompute_intervals) > min(horizons):
        raise ValueError("The smallest recompute interval must be less than or equal to the smallest horizon.")
    
    os.makedirs(results_folder, exist_ok=True)
    
    overall_timing = []
    
    for init_angle in init_angles:
        for wind_mu in wind_mus:
            for wind_sigma in wind_sigmas:
                for h in horizons:
                    for e in recompute_intervals:
                        for ratio in length_ratios:
                            model_length = true_length * ratio
                            start_time = time.time()
                            
                            if controller == 'mpc':
                                mpc = MPCController(horizon=h, recompute_every=e, linear=linear, 
                                                model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, action_space=action_space)
                            elif controller == 'predictive':
                                mpc = SamplingController(controller=controller,horizon=h, recompute_every=e, linear=linear, 
                                                model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, action_space=action_space)
                            elif controller == 'random':
                                mpc = SamplingController(controller=controller,horizon=h, recompute_every=e, linear=linear, 
                                                model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, action_space=action_space)


                            episode_lengths = []
                            episode_times = []
                            integrated_errors = []
                            
                            for ep in range(num_episodes):
                                episode_start_time = time.time()
                                                              
                                if action_space == 'discrete':
                                    env = gym.make("CartPole-v1", render_mode=None)
                                elif action_space == 'continuous':
                                    env = gym.make("InvertedPendulum-v5", render_mode=None, reset_noise_scale=0.00)
                                        
                                obs, _ = env.reset(seed=seed + ep, options={"low": init_angle-0.05, "high": init_angle+0.05})
                                
                                if action_space == 'continuous':
                                    obs = np.array([obs[0], obs[2], obs[1], obs[3]])

                                length, step = 0, 0
                                done = False
                                states = []

                                while not done and length < episode_length:
                                    if step % e == 0:
                                        within_step = 0
                                        trajectory = mpc.get_action(obs,env).flatten()
                                    else:
                                        within_step += 1
                                    
                                    action = trajectory[within_step]

                                    if action_space == 'discrete':
                                        if action > 0:
                                            action = 1
                                        else:
                                            action = 0
                                        

                                    else:
                                        action = np.array([np.clip(action, -3.0, 3.0)]) 
                                    
                                    obs, _, done, _, _ = env.step(action)
                                    
                                    if action_space == 'continuous':
                                        obs = np.array([obs[0], obs[2], obs[1], obs[3]])
                                        
                                    states.append(obs)
                                    step += 1
                                    length += 1
                                
                                episode_end_time = time.time()
                                episode_time = episode_end_time - episode_start_time
                                
                                episode_lengths.append(length)
                                episode_times.append(episode_time)
                                states_squared = np.square(states)
                                integrated_errors.append(np.sum(states_squared, axis=0))
                                env.close()

                            end_time = time.time()
                            total_evaluation_time = end_time - start_time
                            avg_episode_time = np.mean(episode_times)
                            
                            config_data = {
                                'parameters': {
                                    'controller': controller,
                                    'horizon': h,
                                    'recompute_interval': e,
                                    'length_ratio': ratio,
                                    'wind_mu': wind_mu,
                                    'wind_sigma': wind_sigma,
                                    'init_angle': init_angle,
                                    'linear': linear,
                                    'episode_length': episode_length,
                                    'num_episodes': num_episodes,
                                    'seed': seed,
                                    'action_space': action_space
                                },
                                'results': {
                                    'episode_lengths': episode_lengths,
                                    'episode_times': episode_times,
                                    'integrated_errors': integrated_errors,
                                    'avg_episode_time': avg_episode_time,
                                    'total_evaluation_time': total_evaluation_time
                                },
                                'statistics': {
                                    'mean_episode_length': np.mean(episode_lengths),
                                    'std_episode_length': np.std(episode_lengths),
                                    'mean_episode_time': avg_episode_time,
                                    'std_episode_time': np.std(episode_times)
                                }
                            }

                            filename = f"mpc__cont{controller}_h{h}_e{e}_r{ratio:.1f}_wmu{wind_mu:.2f}_wsig{wind_sigma:.2f}_iang{init_angle:.2f}_as{action_space}.pkl"
                            filepath = os.path.join(results_folder, filename)
                            
                            with open(filepath, 'wb') as f:
                                pickle.dump(config_data, f)
                            
                            overall_timing.append({
                                "controller": controller,
                                "horizon": h,
                                "recompute_interval": e,
                                "length_ratio": ratio,
                                "wind_mu": wind_mu,
                                "wind_sigma": wind_sigma,
                                "init_angle": init_angle,
                                "total_evaluation_time": total_evaluation_time,
                                "avg_episode_time": avg_episode_time
                            })

    timing_df = pd.DataFrame(overall_timing)
    timing_path = os.path.join(results_folder, "evaluation_timing_summary.csv")
    timing_df.to_csv(timing_path, index=False)
    
    return

def load_results_data(results_folder="../results/PerformanceResults/"):
    """Load all pickle files and return combined DataFrame."""
    files = [f for f in os.listdir(results_folder) if f.endswith('.pkl') and f.startswith('mpc_')]
    
    if not files:
        print("No pickle files found!")
        return pd.DataFrame()
    
    all_results = []
    for file in files:
        try:
            with open(os.path.join(results_folder, file), 'rb') as f:
                data = pickle.load(f)
            
            result_row = data['parameters'].copy()
            result_row.update(data['statistics'])
            all_results.append(result_row)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    return pd.DataFrame(all_results)

def plot_pole_angle_heatmaps(results_folder="../results/PerformanceResults/", 
                                controller='mpc',
                                length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6],
                                wind_mu=0.0,
                                wind_sigma=0.0,
                                init_angle=0.0,
                                num_episodes=20,
                                dt=0.02,
                                action_space='discrete'):
    """Create multiple heatmaps showing horizon × recompute performance for each pole error ratio."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['wind_mu'] == wind_mu) & 
        (df['wind_sigma'] == wind_sigma) &
        (df['init_angle'] == init_angle) &
        (df['action_space'] == action_space)
    ]
    
    if filtered_data.empty:
        print(f"No data found for controller {controller}, wind_mu={wind_mu}, wind_sigma={wind_sigma}, init_angle={init_angle}, action_space={action_space}")
        return
    
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute_interval'].unique())
    available_ratios = sorted(filtered_data['length_ratio'].unique())
    
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    plot_ratios = [r for r in length_ratios if r in available_ratios]
    
    print(f"Found data for horizons: {horizons} steps = {[f'{h:.1f}s' for h in horizons_sec]}")
    print(f"Found data for recompute intervals: {recomputes} steps = {[f'{r:.2f}s' for r in recomputes_sec]}")
    print(f"Plotting ratios: {plot_ratios}")
    
    fig, axes = plt.subplots(1, len(plot_ratios), figsize=(4*len(plot_ratios), 5))
    
    if len(plot_ratios) == 1:
        axes = [axes]
    
    all_performance_sec = filtered_data['mean_episode_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, ratio in enumerate(plot_ratios):
        ratio_data = filtered_data[filtered_data['length_ratio'] == ratio]
        
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = ratio_data[(ratio_data['recompute_interval'] == recompute) & 
                                  (ratio_data['horizon'] == horizon)]
                if len(subset) > 0:
                    perf_matrix[i, j] = subset['mean_episode_length'].iloc[0] * dt
        
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])
        
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        title = f'Ratio = {ratio:.1f}'
        if ratio == 1.0:
            title += '\n(Perfect Model)'
        else:
            error_pct = abs(ratio - 1.0) * 100
            direction = "Underestimate" if ratio < 1.0 else "Overestimate"
            title += f'\n({direction} {error_pct:.0f}%)'
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{round(perf_matrix[i, j])}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Pole Length Error\n(controller={controller}, wind_μ={wind_mu}, wind_σ={wind_sigma}, init_angle={init_angle:.2f}, action_space={action_space}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_4d_performance_heatmaps_wmu{wind_mu}_wsig{wind_sigma}_iang{init_angle:.2f}_as{action_space}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def plot_wind_mu_heatmaps(results_folder="../results/PerformanceResults/", 
                         controller='mpc',
                         wind_mus=[0.0, 0.5, 1.0, 1.5, 2.0],
                         length_ratio=1.0,
                         wind_sigma=0.0,
                         init_angle=0.0,
                         num_episodes=20,
                         dt=0.02,
                         action_space='discrete'):
    """Create multiple heatmaps showing horizon × recompute performance for each wind_mu level."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['length_ratio'] == length_ratio) & 
        (df['wind_sigma'] == wind_sigma) &
        (df['init_angle'] == init_angle) &
        (df['action_space'] == action_space)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, sigma={wind_sigma}, init_angle={init_angle}, action_space={action_space}")
        return
    
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute_interval'].unique())
    available_wind_mus = sorted(filtered_data['wind_mu'].unique())
    
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    plot_wind_mus = [mu for mu in wind_mus if mu in available_wind_mus]
    
    print(f"Found data for horizons: {horizons} steps = {[f'{h:.1f}s' for h in horizons_sec]}")
    print(f"Found data for recompute intervals: {recomputes} steps = {[f'{r:.2f}s' for r in recomputes_sec]}")
    print(f"Plotting wind means: {plot_wind_mus}")
    
    fig, axes = plt.subplots(1, len(plot_wind_mus), figsize=(4*len(plot_wind_mus), 5))
    
    if len(plot_wind_mus) == 1:
        axes = [axes]
    
    all_performance_sec = filtered_data['mean_episode_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, wind_mu in enumerate(plot_wind_mus):
        mu_data = filtered_data[filtered_data['wind_mu'] == wind_mu]
        
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = mu_data[(mu_data['recompute_interval'] == recompute) & 
                               (mu_data['horizon'] == horizon)]
                if len(subset) > 0:
                    perf_matrix[i, j] = subset['mean_episode_length'].iloc[0] * dt
        
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])
        
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        title = f'Wind μ = {wind_mu:.1f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{round(perf_matrix[i, j])}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Mean\n(controller={controller}, length_ratio={length_ratio}, wind_σ={wind_sigma}, init_angle={init_angle:.2f}, action_space={action_space}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_mu_heatmaps_r{length_ratio:.1f}_s{wind_sigma:.1f}_iang{init_angle:.2f}_as{action_space}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return 

def plot_wind_sigma_heatmaps(results_folder="../results/PerformanceResults/", 
                            controller='mpc',
                            wind_sigmas=[0.0, 0.5, 1.0, 1.5, 2.0],
                            length_ratio=1.0,
                            wind_mu=0.0,
                            init_angle=0.0,
                            num_episodes=20,
                            dt=0.02,
                            action_space='discrete'):
    """Create multiple heatmaps showing horizon × recompute performance for each wind_sigma level."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['length_ratio'] == length_ratio) & 
        (df['wind_mu'] == wind_mu) &
        (df['init_angle'] == init_angle) &
        (df['action_space'] == action_space)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, mu={wind_mu}, init_angle={init_angle}, action_space={action_space}")
        return
    
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute_interval'].unique())
    available_wind_sigmas = sorted(filtered_data['wind_sigma'].unique())
    
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    plot_wind_sigmas = [sigma for sigma in wind_sigmas if sigma in available_wind_sigmas]
    
    print(f"Found data for horizons: {horizons} steps = {[f'{h:.1f}s' for h in horizons_sec]}")
    print(f"Found data for recompute intervals: {recomputes} steps = {[f'{r:.2f}s' for r in recomputes_sec]}")
    print(f"Plotting wind std devs: {plot_wind_sigmas}")
    
    fig, axes = plt.subplots(1, len(plot_wind_sigmas), figsize=(4*len(plot_wind_sigmas), 5))
    
    if len(plot_wind_sigmas) == 1:
        axes = [axes]
    
    all_performance_sec = filtered_data['mean_episode_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, wind_sigma in enumerate(plot_wind_sigmas):
        sigma_data = filtered_data[filtered_data['wind_sigma'] == wind_sigma]
        
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = sigma_data[(sigma_data['recompute_interval'] == recompute) & 
                                  (sigma_data['horizon'] == horizon)]
                if len(subset) > 0:
                    perf_matrix[i, j] = subset['mean_episode_length'].iloc[0] * dt
        
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])
        
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        title = f'Wind σ = {wind_sigma:.1f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{round(perf_matrix[i, j])}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Std Dev\n(controller={controller}, length_ratio={length_ratio}, wind_μ={wind_mu}, init_angle={init_angle:.2f}, action_space={action_space}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_sigma_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}_iang{init_angle:.2f}_as{action_space}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return

def plot_init_angle_heatmaps(results_folder="../results/PerformanceResults/", 
                                      controller='mpc',
                                      init_angles=[0.0, 0.1, 0.2, 0.3, 0.4],
                                      length_ratio=1.0,
                                      wind_mu=0.0,
                                      wind_sigma=0.0,
                                      num_episodes=20,
                                      dt=0.02,
                                      action_space='discrete'):
    """Create multiple heatmaps showing horizon × recompute performance for each initial angle."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['length_ratio'] == length_ratio) & 
        (df['wind_mu'] == wind_mu) &
        (df['wind_sigma'] == wind_sigma) &
        (df['action_space'] == action_space)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, mu={wind_mu}, sigma={wind_sigma}")
        return
    
    horizons = sorted(filtered_data['horizon'].unique())
    recomputes = sorted(filtered_data['recompute_interval'].unique())
    available_init_angles = sorted(filtered_data['init_angle'].unique())
    
    horizons_sec = [h * dt for h in horizons]
    recomputes_sec = [r * dt for r in recomputes]
    
    plot_init_angles = [angle for angle in init_angles if angle in available_init_angles]
    
    if not plot_init_angles:
        print(f"None of the requested init_angles {init_angles} found in data")
        print(f"Available init_angles: {available_init_angles}")
        return
    
    print(f"Found data for horizons: {horizons}")
    print(f"Found data for recompute intervals: {recomputes}")
    print(f"Plotting initial angles: {plot_init_angles}")
    
    fig, axes = plt.subplots(1, len(plot_init_angles), figsize=(4*len(plot_init_angles), 5))
    
    if len(plot_init_angles) == 1:
        axes = [axes]
    
    all_performance_sec = filtered_data['mean_episode_length'].values * dt
    vmin, vmax = np.nanmin(all_performance_sec), np.nanmax(all_performance_sec)
    
    for idx, init_angle in enumerate(plot_init_angles):
        angle_data = filtered_data[filtered_data['init_angle'] == init_angle]
        
        perf_matrix = np.full((len(recomputes), len(horizons)), np.nan)
        
        for i, recompute in enumerate(recomputes):
            for j, horizon in enumerate(horizons):
                subset = angle_data[(angle_data['recompute_interval'] == recompute) & 
                                  (angle_data['horizon'] == horizon)]
                if len(subset) > 0:
                    perf_matrix[i, j] = subset['mean_episode_length'].iloc[0] * dt
        
        im = axes[idx].imshow(perf_matrix, cmap='viridis', aspect='auto', 
                             vmin=vmin, vmax=vmax, origin='lower')
        
        axes[idx].set_xticks(range(len(horizons)))
        axes[idx].set_xticklabels([f'{h:.1f}' for h in horizons_sec])
        axes[idx].set_yticks(range(len(recomputes)))
        axes[idx].set_yticklabels([f'{r:.2f}' for r in recomputes_sec])
        
        axes[idx].set_xlabel('Planning Horizon (s)', fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel('Recompute Every (s)', fontsize=11)
        
        title = f'Init Angle = {init_angle:.2f} rad'
        if init_angle == 0.0:
            title += '\n(Upright)'
        else:
            degrees = init_angle * 180 / np.pi
            title += f'\n({degrees:.1f}°)'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{round(perf_matrix[i, j])}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Initial Angle\n(controller={controller}, length_ratio={length_ratio}, wind_μ={wind_mu}, wind_σ={wind_sigma}, action_space={action_space}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_init_angle_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}_s{wind_sigma:.1f}_as{action_space}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return