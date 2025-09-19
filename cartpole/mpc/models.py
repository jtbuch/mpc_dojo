import numpy as np
import mujoco
import copy
from scipy.linalg import expm, svd
import torch
from stable_baselines3 import TD3
import scipy.linalg

class MPCShittyBird:
    """ Doesn't aspire to much. """
    def __init__(self, n_actions, recompute=10, planning_width=5, n_planning=10, 
                 reward_type='discrete', action_cost=0.0, control_model='random', 
                 value='env', rl_model='50k', dt_physics=0.02):
        
        # Model parameters
        self.n_actions = n_actions
        self.recompute = recompute
        self.control_trajectory = None
        self.control_step = 0

        # Planning parameters
        self.n_planning = n_planning
        self.planning_width = planning_width
        self.epsilon = 0.2

        # Reward and control settings
        self.reward_type = reward_type
        self.action_cost = action_cost
        self.control_model = control_model
        self.value = value
        self.rl_model = rl_model
        self.dt_physics = dt_physics
        
        # MPC parameters (system matrices cached, gains computed each time)
        self.A = None  # System matrix
        self.B = None  # Input matrix
        self.system_linearized = False
        
        # Load RL model if needed
        self._load_rl_model()
  
    def _load_rl_model(self):
        """Load RL model if using RL-based methods."""
        model_files = {
            '10k': "../rl_models/td3_invertedpendulum_continuous_10k_steps",
            '50k': "../rl_models/td3_invertedpendulum_continuous_50k_steps", 
            '100k': "../rl_models/td3_invertedpendulum_continuous_100k_steps",
            '300k': "../rl_models/td3_invertedpendulum_continuous_300k_steps", 
            '500k': "../rl_models/td3_invertedpendulum_continuous_500k_steps",
            '1000k': "../rl_models/td3_invertedpendulum_continuous_1000k_steps",
            '2000k': "../rl_models/td3_invertedpendulum_continuous_2000k_steps",
            '3000k': "../rl_models/td3_invertedpendulum_continuous_3000k_steps",
            '4000k': "../rl_models/td3_invertedpendulum_continuous_4000k_steps",
            '5000k': "../rl_models/td3_invertedpendulum_continuous_5000k_steps",
        }
            
        if self.rl_model in model_files:
            try:
                self.model = TD3.load(model_files[self.rl_model])
                print(f"✓ Loaded {self.rl_model} model")
            except Exception as e:
                print(f"✗ Failed to load {self.rl_model} model: {e}")
        else:
            print(f"✗ Unknown model name: {self.rl_model}")

    def _initialize_system_matrices(self):
        """Initialize with much more conservative tuning."""
        if self.system_linearized:
            return
            
        print("Computing system linearization for MPC...")
        
        self.A, self.B = self._linearize_system_analytical()
        self.x_eq = np.array([0.0, 0.0, 0.0, 0.0])
        self.u_eq = np.array([0.0])
        
        # MUCH more conservative cost matrices
        self.Q = np.diag([
            0.1,   # position - very low penalty (let it move)
            10.0,  # angle - moderate penalty (not too aggressive)  
            0.01,  # cart velocity - very small
            0.1    # angular velocity - small damping
        ])
        
        # HIGH control cost to prevent aggressive actions
        self.R = np.array([[1.0]])  # Much higher than before
        
        # Conservative terminal cost
        self.Q_terminal = 2.0 * self.Q  # Not too high
        
        self.system_linearized = True
        print(f"Conservative tuning: Q = {np.diag(self.Q)}, R = {self.R[0,0]}")
        
    def _linearize_system_analytical(self):
        """Analytical linearization of inverted pendulum around upright equilibrium."""
        # Physical parameters (typical values for InvertedPendulum-v5)
        g = 9.8  # gravity
        m_cart = 1.0  # cart mass
        m_pole = 0.1  # pole mass
        l = 0.5  # pole half-length
        dt = self.dt_physics
        
        # Linearized continuous-time dynamics around upright position
        M = m_cart + m_pole
        denom = l * (4/3 - m_pole / M)
        
        A_continuous = np.array([
            [0, 0, 1, 0],                    # ẋ = ẋ
            [0, 0, 0, 1],                    # θ̇ = θ̇  
            [0, -m_pole*g/M, 0, 0],          # ẍ = ... - mg*θ/M
            [0, g/denom, 0, 0]               # θ̈ = g*θ/denom
        ])
        
        B_continuous = np.array([
            [0],
            [0], 
            [1/M],                           # ẍ component
            [-1/(M*denom)]                   # θ̈ component  
        ])
        
        # Convert to discrete time using matrix exponential
        AB = np.block([[A_continuous, B_continuous], 
                       [np.zeros((1, 5))]])
        
        exp_AB = scipy.linalg.expm(AB * dt)
        
        A = exp_AB[:4, :4]
        B = exp_AB[:4, 4:5]
        
        return A, B

    def _solve_finite_horizon_mpc(self, current_state):
        """
        Solve finite horizon MPC problem using dynamic programming.
        Returns optimal action sequence for the current planning horizon.
        """
        N = self.n_planning  # Planning horizon
        n_states = self.A.shape[0]
        n_controls = self.B.shape[1]
        
        # Backward pass: compute time-varying gains via Riccati recursion
        P = [None] * (N + 1)
        K = [None] * N  # Feedback gains
        k = [None] * N  # Feedforward terms
        
        # Terminal condition
        P[N] = self.Q_terminal
        
        # Backward recursion
        for t in reversed(range(N)):  # t = N-1, N-2, ..., 0
            # Compute optimal gain for time step t
            temp_inv = np.linalg.solve(self.R + self.B.T @ P[t+1] @ self.B, 
                                     np.eye(n_controls))
            K[t] = temp_inv @ self.B.T @ P[t+1] @ self.A
            k[t] = np.zeros(n_controls)  # No feedforward term for regulation problem
            
            # Update cost-to-go (Riccati equation)
            P[t] = (self.Q + self.A.T @ P[t+1] @ self.A - 
                   self.A.T @ P[t+1] @ self.B @ K[t])
        
        # Forward pass: compute optimal trajectory
        action_sequence = np.zeros((N, n_controls))
        state = current_state - self.x_eq  # Error coordinates
        
        for t in range(N):
            # Optimal control law: u_t = -K_t * x_t + k_t + u_eq
            action = -K[t] @ state + k[t] + self.u_eq
            action_sequence[t] = action
            
            # Propagate state forward
            state = self.A @ state + self.B @ (action - self.u_eq)
        
        return action_sequence

    def restore_state(self, env, qpos, qvel):
        """Restore environment state."""
        env.unwrapped.data.qpos[:] = qpos
        env.unwrapped.data.qvel[:] = qvel
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    def calculate_trajectory_rewards(self, env, actions, observations, discrete_rewards, done_flags):
        """Calculate rewards for trajectories."""
        if self.reward_type == 'discrete':
            return discrete_rewards.sum(axis=1)
        
        elif self.reward_type == 'continuous':
            if self.value == 'env':
                # Your simple quadratic reward function
                x = observations[:, :, 0]        # cart position
                theta = observations[:, :, 1]    # pole angle  
                x_dot = observations[:, :, 2]    # cart velocity
                theta_dot = observations[:, :, 3] # pole angular velocity
                
                # Quadratic costs (matching MPC cost matrices)
                position_cost = x**2
                angle_cost = 10.0 * theta**2
                velocity_cost = 0.1 * (x_dot**2 + theta_dot**2)
                
                total_cost = position_cost + angle_cost + velocity_cost
                reward = -total_cost
                
                return reward.sum(axis=1)
            
            elif self.value == 'rl':
                # RL-based rewards
                batch_size = self.planning_width * self.n_planning
                obs_batch = observations.reshape(batch_size, -1)
                
                with torch.no_grad():
                    actions_batch, _ = self.model.predict(obs_batch, deterministic=True)
                    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32)
                    actions_tensor = torch.as_tensor(actions_batch, dtype=torch.float32)
                    critic_output = self.model.critic(obs_tensor, actions_tensor)
                    rewards_batch = critic_output[0].detach().numpy().flatten()
                
                rewards_matrix = rewards_batch.reshape(self.planning_width, self.n_planning)
                return rewards_matrix.sum(axis=1)

    def mujoco_policy(self, env, obs):
        """Main policy function - routes to appropriate controller."""
        
        # MPC Controller - finite horizon optimal control
        if self.control_model == 'mpc':
            # Initialize system matrices if needed
            if not self.system_linearized:
                self._initialize_system_matrices()
            
            # Solve finite horizon MPC problem
            action_sequence = self._solve_finite_horizon_mpc(obs)
            
            # Clip actions to bounds
            action_sequence = np.clip(action_sequence, 
                                    env.action_space.low, 
                                    env.action_space.high)
            
            return action_sequence
        
        # All other methods (random, predictive, rl) 
        return self._trajectory_based_policy(env, obs)
    
    def _trajectory_based_policy(self, env, obs):
        """Handle trajectory-based methods (random, predictive, rl)."""
        
        # Initialize nominal trajectory for predictive methods
        if self.control_model in ['predictive', 'rl']:
            if not hasattr(self, 'nominal_trajectory'):
                if self.control_model == 'rl':
                    self.nominal_trajectory = np.array([
                        self.model.predict(obs, deterministic=True)[0] 
                        for _ in range(self.n_planning)
                    ])
                else:  # predictive
                    self.nominal_trajectory = np.zeros((self.n_planning, *env.action_space.shape))
        
        # Save environment state 
        saved_qpos = env.unwrapped.data.qpos.copy()
        saved_qvel = env.unwrapped.data.qvel.copy()

        # Generate action trajectories
        actions = self._generate_action_trajectories(env, obs)
        
        # Evaluate trajectories
        observations, discrete_rewards, done_flags = self._evaluate_trajectories(env, obs, actions)
        
        # Calculate rewards and select best
        cumulative_rewards = self.calculate_trajectory_rewards(env, actions, observations, discrete_rewards, done_flags)
        
        # Restore state and return best trajectory
        self.restore_state(env, saved_qpos, saved_qvel)
        
        best_idx = np.argmax(cumulative_rewards)
        best_trajectory = actions[best_idx]
        
        # Update nominal trajectory for predictive methods
        if self.control_model in ['predictive', 'rl']:
            self.nominal_trajectory = best_trajectory
        
        return best_trajectory

    def _generate_action_trajectories(self, env, obs):
        """Generate action trajectories based on control model."""
        actions = np.zeros([self.planning_width, self.n_planning, *env.action_space.shape])
        
        if self.control_model in ['predictive', 'rl']:
            # Shift nominal trajectory
            if self.n_planning > 1:
                self.nominal_trajectory = np.vstack([
                    self.nominal_trajectory[1:],
                    np.zeros((1, *env.action_space.shape))
                ])
            
            # First candidate is nominal
            actions[0] = self.nominal_trajectory
            
            # Add noise for other candidates
            if self.planning_width > 1:
                noise_std = 0.1 * (env.action_space.high - env.action_space.low)
                for i in range(1, self.planning_width):
                    noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                    actions[i] = np.clip(
                        self.nominal_trajectory + noise,
                        env.action_space.low, 
                        env.action_space.high
                    )
        
        elif self.control_model == 'random':
            # Random shooting
            actions = np.random.uniform(
                low=env.action_space.low,
                high=env.action_space.high,
                size=(self.planning_width, self.n_planning, *env.action_space.shape)
            )
        
        return actions

    def _evaluate_trajectories(self, env, initial_obs, actions):
        """Evaluate all action trajectories."""
        observations = np.zeros([self.planning_width, self.n_planning, *env.observation_space.shape])
        discrete_rewards = np.zeros([self.planning_width, self.n_planning])
        done_flags = np.zeros([self.planning_width, self.n_planning], dtype=bool)
        
        saved_qpos = env.unwrapped.data.qpos.copy()
        saved_qvel = env.unwrapped.data.qvel.copy()
        
        for i in range(self.planning_width):
            self.restore_state(env, saved_qpos, saved_qvel)
            current_obs = initial_obs.copy()

            for step in range(self.n_planning):
                if self.n_planning < 2 and self.control_model == 'rl':
                    # Single-step RL
                    action = self.model.predict(current_obs, deterministic=True)[0]
                else:
                    action = actions[i, step]
                    
                next_obs, reward, done, truncated, info = env.step(action)
                
                actions[i, step] = action  # Store actual action used
                observations[i, step] = next_obs
                discrete_rewards[i, step] = reward
                done_flags[i, step] = done or truncated
                
                current_obs = next_obs
        
        return observations, discrete_rewards, done_flags