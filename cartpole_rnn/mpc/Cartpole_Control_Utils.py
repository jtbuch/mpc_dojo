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
from scipy.integrate import solve_ivp
from casadi import cos, sin  
import torch

def cartpole_dynamics(self, state, action, model_length=1.0):
    """Works with both symbolic and numeric inputs"""
    self.gravity = 9.81
    self.masscart = 1.0
    self.masspole = 0.1
    self.length = model_length
    self.total_mass = self.masscart + self.masspole
    self.polemass_length = self.masspole * self.length

    x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
    u = action
    
    # Auto-detect if symbolic or numeric
    if hasattr(theta, 'is_symbolic') or str(type(theta)).find('SX') != -1 or str(type(theta)).find('MX') != -1:
        # Symbolic (CasADi)
        costheta = cos(theta)
        sintheta = sin(theta)
        is_symbolic = True
    else:
        # Numeric (NumPy)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        is_symbolic = False
    
    temp = (u + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
    thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
            )
    xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
    
    if is_symbolic:
        return [x_dot, xacc, theta_dot, thetaacc]
    else:
        return np.array([x_dot, xacc, theta_dot, thetaacc])



class MPCController:
    def __init__(self, horizon=10, dt=0.02, recompute_every=1, model_length=1.0, wind_mu=0.0, wind_sigma=0.0):
        self.horizon = horizon
        self.dt = dt
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

        # Get the derivatives
        state = [x, x_dot, theta, theta_dot]
        derivatives = cartpole_dynamics(self, state, u, model_length=self.length)
        
        # Unpack the returned list
        x_dot_rhs = derivatives[0]
        xacc = derivatives[1]
        theta_dot_rhs = derivatives[2]
        thetaacc = derivatives[3]
        
        self.model.set_rhs('x', x_dot_rhs)
        self.model.set_rhs('x_dot', xacc)
        self.model.set_rhs('theta', theta_dot_rhs)
        self.model.set_rhs('theta_dot', thetaacc)
        
        self.model.setup()
        
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': self.dt,
            'n_robust': 0,
            'store_full_solution': True,
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 1,
        }
            
        self.mpc.settings.supress_ipopt_output()
        self.mpc.set_param(**setup_mpc)

        # Increased x penalty to 10 for better centering
        self.mpc.set_objective(
            mterm=10*theta**2 + 10*x**2 + theta_dot**2 + x_dot**2,
            lterm=10*theta**2 + 10*x**2 + theta_dot**2 + x_dot**2
        )
        self.mpc.set_rterm(u=0.1)
        self.mpc.bounds['lower','_u','u'] = -self.force_mag
        self.mpc.bounds['upper','_u','u'] = self.force_mag
        
        self.mpc.setup()

    def get_action(self, obs, env=None):
        obs_with_noise = obs.copy()
        wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma)
        obs_with_noise[2] += wind_disturbance
        self.mpc.x0 = np.array(obs_with_noise, dtype=float).reshape(-1, 1)
        self.mpc.set_initial_guess()
        self.mpc.make_step(self.mpc.x0)
        trajectory = self.mpc.data.prediction(('_u',))
        
        x_pred = self.mpc.data.prediction(('_x', 'x')).squeeze()[1:]
        x_dot_pred = self.mpc.data.prediction(('_x', 'x_dot')).squeeze()[1:]
        theta_pred = self.mpc.data.prediction(('_x','theta')).squeeze()[1:]
        theta_dot_pred = self.mpc.data.prediction(('_x', 'theta_dot')).squeeze()[1:]

        predictions = np.vstack([x_pred, x_dot_pred, theta_pred, theta_dot_pred]).T

        return trajectory, predictions
    
class SamplingController:
    def __init__(self, controller='predictive',horizon=10, dt=0.02, recompute_every=1, model_length=1.0, wind_mu=0.0, wind_sigma=0.0):
        self.horizon = horizon
        self.dt = dt
        self.recompute_every = recompute_every
        
        self.length = model_length
        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma
        self.control_model = controller

        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma

        self.force_mag = 3.0
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1

    def generate_action_trajectories(self, env, obs):
        """Generate action trajectories based on control model."""
        self.n_candidate_trajectories = 200
        actions = np.zeros([self.n_candidate_trajectories, self.horizon, 1])
        
        if self.control_model == 'predictive':
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
                noise_std = 0.6
                for i in range(1, self.n_candidate_trajectories):
                    noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                    actions[i] = np.clip(
                        self.nominal_trajectory + noise,
                        -3.0, 3.0
                    )
        
        elif self.control_model == 'random':
            # Random shooting
            actions = np.random.uniform(
                low=-3.0,
                high=3.0,
                size=(self.n_candidate_trajectories, self.horizon, 1))

        return actions
    
    def calculate_trajectories(self, env, initial_obs, actions):
        """
        Ttrajectory evaluation using Euler integration.
        
        Args:
            env: Environment (for compatibility, not used in vectorized version)
            initial_obs: Initial state (4,)
            actions: Action trajectories (n_traj, horizon, 1)
            
        Returns:
            observations: (n_traj, horizon, 4)
            discrete_rewards: (n_traj, horizon) - zeros for compatibility
            done_flags: (n_traj, horizon) - zeros for compatibility
        """

        # Initialize output arrays
        observations = np.zeros([self.n_candidate_trajectories, self.horizon, env.observation_space.shape[0]])

        # Loop over trajectories
        for i in range(self.n_candidate_trajectories):

            current_state = initial_obs

            # Simulate forward in time
            for step in range(self.horizon):
                # Get actions for this time step 
                action = actions[i, step, 0]
                
                # Compute derivatives for all trajectories simultaneously
                derivatives = cartpole_dynamics(self, current_state, action, self.length)
                
                # Euler integration: x_{t+1} = x_t + dt * f(x_t, u_t)
                current_state = current_state + derivatives * self.dt

                current_state = self.clip_to_env_bounds(current_state, env)

                next_obs_with_noise = current_state.copy()
                wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma)
                next_obs_with_noise[2] += wind_disturbance
                
                observations[i, step] = next_obs_with_noise
            
        return observations
    
    def clip_to_env_bounds(self, obs, env):
        """Clip observations to environment's observation space"""
        clipped_obs = obs.copy()
        
        # Get environment bounds
        low = env.observation_space.low
        high = env.observation_space.high
        
        # Only clip finite bounds (not -inf/+inf)
        for i in range(len(obs)):
            if np.isfinite(low[i]) and np.isfinite(high[i]):
                clipped_obs[i] = np.clip(obs[i], low[i], high[i])
            elif np.isfinite(low[i]):
                clipped_obs[i] = np.maximum(obs[i], low[i])
            elif np.isfinite(high[i]):
                clipped_obs[i] = np.minimum(obs[i], high[i])
        
        return clipped_obs
    
    def get_action(self, obs, env):
        """Handle trajectory-based methods (random, predictive)."""
        
        # Initialize nominal trajectory for predictive methods
        if self.control_model=='predictive':
            if not hasattr(self, 'nominal_trajectory'):
                    self.nominal_trajectory = np.zeros((self.horizon, 1))       

        # Generate action trajectories
        actions = self.generate_action_trajectories(env, obs)
        
        # Evaluate trajectories
        observations = self.calculate_trajectories(env, obs, actions)
        
        # Calculate rewards and select best
        cumulative_rewards = self.calculate_trajectory_rewards(env, actions, observations)
        
        best_idx = np.argmax(cumulative_rewards)
        best_trajectory = actions[best_idx]

        predicted_observations = observations[best_idx]
        
        # Update nominal trajectory for predictive methods
        self.nominal_trajectory = best_trajectory

        return best_trajectory, predicted_observations
 
    def calculate_trajectory_rewards(self, env, actions, obs):
        reward = np.zeros((self.n_candidate_trajectories, self.horizon))
        
        # Match MPC objective exactly
        stage_cost = (10 * obs[:, :, 2]**2 +     # 10*theta^2
                    obs[:, :, 0]**2 +           # x^2  
                    obs[:, :, 3]**2 +           # theta_dot^2
                    obs[:, :, 1]**2)            # x_dot^2
        
        action_cost = 0.1 * actions[:, :, 0]**2   # Match MPC's rterm
        
        reward = -(stage_cost + action_cost)  # Negative because we maximize reward
        
        return reward.sum(axis=1)

def evaluate_mpc_controllers(controller, horizons, recompute_intervals, dt, results_folder="../results/PerformanceResults/", 
                             episode_length=500, num_episodes=20, seed=42, 
                             length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6], wind_mus=[0.0], wind_sigmas=[0.0], init_angles=[0.0]):
    """Evaluate MPC with various parameters and save each configuration separately."""
    
    if min(recompute_intervals) > min(horizons):
        raise ValueError("The smallest recompute interval must be less than or equal to the smallest horizon.")
    
    os.makedirs(results_folder, exist_ok=True)

    # Get the pole lenght from the environment
    env = gym.make("CartPole-v1", render_mode=None)
    env.unwrapped.tau = dt
    # mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    env.reset() 
    true_length = env.unwrapped.length
    
    overall_timing = []
    
    for init_angle in init_angles:
        for wind_mu in wind_mus:
            for wind_sigma in wind_sigmas:
                for h in horizons:
                    for e in recompute_intervals:
                        for ratio in length_ratios:
                            # change the pole length in simulation
                            model_length = ratio * true_length
                            start_time = time.time()
                            
                            if controller == 'mpc':
                                mpc = MPCController(horizon=h, recompute_every=e, 
                                                model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma)
                            elif controller == 'predictive':
                                mpc = SamplingController(controller=controller,horizon=h, recompute_every=e, 
                                                model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma)
                            elif controller == 'random':
                                mpc = SamplingController(controller=controller,horizon=h, recompute_every=e, 
                                                model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma)


                            episode_lengths = []
                            episode_times = []
                            integrated_errors = []
                            
                            for ep in range(num_episodes):
                                episode_start_time = time.time()

                                env = gym.make("CartPole-v1", render_mode=None)
                                env.unwrapped.tau = dt
                                        
                                obs, _ = env.reset(seed=seed + ep)
                                
                                length, step = 0, 0
                                done = False
                                states = []

                                while not done and length < episode_length:
                                    if step % e == 0:
                                        within_step = 0
                                        trajectory, predictions = mpc.get_action(obs, env)
                                        trajectory = trajectory.flatten()
                                    else:
                                        within_step += 1
                                    
                                    action = trajectory[within_step]

                                    action = np.array([np.clip(action, -3.0, 3.0)]) 

                                    # Make action discrete
                                    if action >= 0:
                                        action = 1
                                    else:
                                        action = 0
                                    
                                    obs, _, done, _, _ = env.step(action)
                                        
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
                                    'episode_length': episode_length,
                                    'num_episodes': num_episodes,
                                    'seed': seed
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

                            filename = f"mpc__cont{controller}_h{h}_e{e}_r{ratio:.1f}_wmu{wind_mu:.2f}_wsig{wind_sigma:.2f}_iang{init_angle:.2f}.pkl"
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

def calculate_cost_error(controller_type, true_state, predicted_state, action):
    """Calculate cost prediction error."""
    true_cost = 0.0
    predicted_cost = 0.0

    true_cost -= 10 * (true_state[1]**2) + true_state[0]**2
    true_cost -= (true_state[2]**2 + true_state[3]**2)
    true_cost -= 0.1 * (action**2)

    predicted_cost -= 10 * (predicted_state[1]**2) + predicted_state[0]**2
    predicted_cost -= (predicted_state[2]**2 + predicted_state[3]**2)
    predicted_cost -= 0.1 * (action**2)

    cost_pe = abs(true_cost - predicted_cost)

    return cost_pe

def run_single_episode_with_plots(controller_type='mpc', horizon=10, dt=0.02, 
                                 recompute_every=1, model_length=0.5, wind_mu=0.0, wind_sigma=0.0,
                                 episode_length=500, seed=42, init_angle=0.1):
    """
    Run a single episode with the specified controller and plot angle and actions over time.
    """

    # Get the pole length from the environment
    env = gym.make("CartPole-v1", render_mode=None)
    env.unwrapped.tau = dt
    env.reset() 
    true_length = env.unwrapped.length
    model_length = model_length * true_length
    
    print(f"True length: {true_length}, Model length: {model_length}")
    
    # Initialize the controller
    if controller_type == 'mpc':
        controller = MPCController(horizon=horizon, dt=dt, 
                                 recompute_every=recompute_every, model_length=model_length, 
                                 wind_mu=wind_mu, wind_sigma=wind_sigma)
        
    elif controller_type in ['predictive', 'random']:
        controller = SamplingController(controller=controller_type, horizon=horizon, dt=dt, 
                                      recompute_every=recompute_every, 
                                      model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma)
    elif controller_type == 'rnn_mpc':
        # Load model
        model, info = load_rnn_model(checkpoint=5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        controller = RNNMPCController(horizon=horizon, dt=dt, 
                                       model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma)
    else:
        raise ValueError("controller_type must be 'mpc', 'predictive', or 'random'")
    
    # Initialize environment (create fresh one with tau set)
    env = gym.make("CartPole-v1", render_mode=None)
    env.unwrapped.tau = dt  
    obs, _ = env.reset(seed=seed)
    
    # Storage for plotting
    angles = []
    actions_taken = []
    cart_positions = []
    time_steps = []
    prediction_errors = []
    cost_prediction_errors = []
    
    length, step = 0, 0
    done = False
    
    trajectory = None
    all_predictions = None
    
    while length < episode_length:# and not done:  
        # Get action from controller
        # if step % recompute_every == 0:
        if step % recompute_every == 0:
            within_step = 0
            trajectory, all_predictions = controller.get_action(obs, env)
            trajectory = trajectory.flatten()
        else:
            within_step = (step % recompute_every)

        action = trajectory[within_step]
        predictions = all_predictions[within_step]

        continuous_action = float(action)

        # Make action discrete
        if float(action) >= 0:
            discrete_action = 1
        else:
            discrete_action = 0
        
        # Store data for plotting
        time_steps.append(length * dt)
        cart_positions.append(obs[0])
        angles.append(obs[2])
        actions_taken.append(continuous_action)
        
        # Step environment
        obs, _, done, _, _ = env.step(discrete_action)
        
        prediction_errors.append(obs - predictions)
        step += 1
        length += 1

        cost_prediction_error = calculate_cost_error(controller_type, obs, predictions, continuous_action)
        cost_prediction_errors.append(cost_prediction_error)

    env.close()
    
    # Convert to numpy arrays
    time_steps = np.array(time_steps)
    angles = np.array(angles)
    actions_taken = np.array(actions_taken)
    cart_positions = np.array(cart_positions)
    prediction_errors = np.array(prediction_errors).T  
    
    # Create plots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 9))
    
    # Plot 1: Angle over time
    ax1.plot(time_steps, np.rad2deg(angles), 'b-', linewidth=2, label='Pole Angle')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target (0°)')
    ax1.axhline(y=24, color='r', linestyle=':', alpha=0.3, label='Failure Limits')
    ax1.axhline(y=-24, color='r', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title(f'Pole Angle Over Time - {controller_type.upper()} Controller\n'
                  f'H={horizon}, Recompute={recompute_every}, Length={model_length:.3f}m, '
                  f'Wind σ={wind_sigma:.2f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-30, 30])
    
    # Plot 2: Actions over time
    ax2.plot(time_steps, actions_taken, 'g-', linewidth=2, label='Control Action')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No Force')
    ax2.axhline(y=3.0, color='r', linestyle=':', alpha=0.5, label='Force Limits')
    ax2.axhline(y=-3.0, color='r', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Force (N)')
    ax2.set_title('Control Actions Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-4, 4])

    # Plot 3: prediction errors
    colors = ['b', 'g', 'orange', 'purple']
    labels = ['x (cart position)', 'ẋ (cart velocity)', 'θ (pole angle)', 'θ̇ (angular velocity)']
    
    for i in range(4):
        ax3.plot(time_steps, prediction_errors[i], color=colors[i], label=labels[i], linewidth=2)
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Prediction Error')
    ax3.set_title('Prediction Errors Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: cost prediction error
    ax4.plot(time_steps, cost_prediction_errors, 'm-', linewidth=2, label='Cost Prediction Error')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Cost Prediction Error')
    ax4.set_title('Cost Prediction Error Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('mpc_cartpole_results.png', dpi=150)
    plt.show()
       
    return prediction_errors

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
                                dt=0.02):
    """Create multiple heatmaps showing horizon × recompute performance for each pole error ratio."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['wind_mu'] == wind_mu) & 
        (df['wind_sigma'] == wind_sigma) &
        (df['init_angle'] == init_angle)
    ]
    
    if filtered_data.empty:
        print(f"No data found for controller {controller}, wind_mu={wind_mu}, wind_sigma={wind_sigma}, init_angle={init_angle}")
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

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Pole Length Error\n(controller={controller}, wind_μ={wind_mu}, wind_σ={wind_sigma}, init_angle={init_angle:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_4d_performance_heatmaps_wmu{wind_mu}_wsig{wind_sigma}_iang{init_angle:.2f}.png")
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
                         dt=0.02):
    """Create multiple heatmaps showing horizon × recompute performance for each wind_mu level."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['length_ratio'] == length_ratio) & 
        (df['wind_sigma'] == wind_sigma) &
        (df['init_angle'] == init_angle)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, sigma={wind_sigma}, init_angle={init_angle}")
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

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Mean\n(controller={controller}, length_ratio={length_ratio}, wind_σ={wind_sigma}, init_angle={init_angle:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_mu_heatmaps_r{length_ratio:.1f}_s{wind_sigma:.1f}_iang{init_angle:.2f}.png")
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
                            dt=0.02):
    """Create multiple heatmaps showing horizon × recompute performance for each wind_sigma level."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['length_ratio'] == length_ratio) & 
        (df['wind_mu'] == wind_mu) &
        (df['init_angle'] == init_angle)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, mu={wind_mu}, init_angle={init_angle}")
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

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Std Dev\n(controller={controller}, length_ratio={length_ratio}, wind_μ={wind_mu}, init_angle={init_angle:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_sigma_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}_iang{init_angle:.2f}.png")
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
                                      dt=0.02):
    """Create multiple heatmaps showing horizon × recompute performance for each initial angle."""
    
    df = load_results_data(results_folder)
    
    if df.empty:
        print("No data found!")
        return
    
    filtered_data = df[
        (df['controller'] == controller) &
        (df['length_ratio'] == length_ratio) & 
        (df['wind_mu'] == wind_mu) &
        (df['wind_sigma'] == wind_sigma)
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

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Initial Angle\n(controller={controller}, length_ratio={length_ratio}, wind_μ={wind_mu}, wind_σ={wind_sigma}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_init_angle_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}_s{wind_sigma:.1f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return