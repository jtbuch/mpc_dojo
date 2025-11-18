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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MDRNN(nn.Module):
    """Mixture Density RNN for CartPole world model.
    
    Predicts next state distribution as a Gaussian Mixture Model.
    State: [x, x_dot, theta, theta_dot] (4D)
    Action: one-hot encoded discrete action (2D for CartPole: [1,0] or [0,1])
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=256, 
                 n_gaussian=5, n_layers=1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussian = n_gaussian
        self.n_layers = n_layers
        
        # LSTM that processes state+action sequences
        self.lstm = nn.LSTM(
            state_dim + action_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True
        )
        
        # Output heads for mixture density network
        self.pi_head = nn.Linear(hidden_dim, n_gaussian)
        self.mu_head = nn.Linear(hidden_dim, n_gaussian * state_dim)
        self.sigma_head = nn.Linear(hidden_dim, n_gaussian * state_dim)
        
    def forward(self, states, actions, hidden=None):
        """
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            hidden: tuple of (h, c) for LSTM hidden states
        
        Returns:
            dict with:
                pi: mixture weights (batch, seq_len, n_gaussian)
                mu: means (batch, seq_len, n_gaussian, state_dim)
                sigma: std devs (batch, seq_len, n_gaussian, state_dim)
                hidden: updated LSTM hidden state
        """
        batch_size, seq_len = states.shape[:2]
        
        # Concatenate state and action
        x = torch.cat([states, actions], dim=-1)
        
        # Process through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Compute mixture parameters
        pi = self.pi_head(lstm_out).view(batch_size, seq_len, self.n_gaussian)
        mu = self.mu_head(lstm_out).view(batch_size, seq_len, self.n_gaussian, self.state_dim)
        sigma = self.sigma_head(lstm_out).view(batch_size, seq_len, self.n_gaussian, self.state_dim)
        
        # Apply activations
        pi = F.softmax(pi, dim=-1)  # Normalize mixture weights
        sigma = F.softplus(sigma) + 0.01  # Ensure positive std dev with reasonable minimum
        
        return {
            'pi': pi,
            'mu': mu, 
            'sigma': sigma,
            'hidden': hidden
        }
    
    def sample_prediction(self, pi, mu, sigma, temperature=1.0):
        """Sample next state from the mixture distribution.
        
        Args:
            pi: (n_gaussian,) - mixture weights
            mu: (n_gaussian, state_dim) - means
            sigma: (n_gaussian, state_dim) - std devs
            temperature: sampling temperature (higher = more random)
        
        Returns:
            next_state: (state_dim,) - sampled next state
        """
        # Apply temperature to mixture weights
        if temperature != 1.0:
            pi = F.softmax(torch.log(pi + 1e-8) / temperature, dim=-1)
        
        # Sample mixture component
        k = torch.multinomial(pi, 1).item()
        
        # Sample from the selected gaussian
        mean = mu[k]
        std = sigma[k] * temperature
        next_state = torch.normal(mean, std)
        
        return next_state
    
    def predict_next_state(self, state, action, hidden=None, temperature=1.0, deterministic=False):
        """Predict next state given current state and action.
        
        Args:
            state: (state_dim,) numpy array or tensor
            action: int (discrete action) or (action_dim,) one-hot array
            hidden: LSTM hidden state, or None to create new
            temperature: sampling temperature
            deterministic: if True, return mean of highest-weight mixture component
        
        Returns:
            next_state: (state_dim,) predicted next state
            hidden: updated hidden state
        """
        device = next(self.parameters()).device
        
        # Convert inputs to tensors
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, int):
            # Convert discrete action to one-hot
            action_onehot = torch.zeros(self.action_dim)
            action_onehot[action] = 1.0
            action = action_onehot
        elif isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        # Add batch and sequence dimensions: (1, 1, dim)
        state = state.unsqueeze(0).unsqueeze(0).to(device)
        action = action.unsqueeze(0).unsqueeze(0).to(device)
        
        # Initialize hidden if needed
        if hidden is None:
            hidden = self.init_hidden(1, device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self(state, action, hidden)
        
        # Get predictions for single timestep
        pi = outputs['pi'][0, 0]      # (n_gaussian,)
        mu = outputs['mu'][0, 0]      # (n_gaussian, state_dim)
        sigma = outputs['sigma'][0, 0]  # (n_gaussian, state_dim)
        
        if deterministic:
            # Return mean of highest-weight component
            k = pi.argmax()
            next_state = mu[k].cpu().numpy()
        else:
            # Sample from mixture
            next_state = self.sample_prediction(pi, mu, sigma, temperature).cpu().numpy()
        
        return next_state, outputs['hidden']
    
    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

def load_model(save_dir='../trained_models', checkpoint=None, model_path=None):
    """Load a saved MDRNN model.
    
    Args:
        save_dir: Directory where models are saved (default: '../trained_models')
        checkpoint: Checkpoint number (1-5) to load, or None if using model_path
        model_path: Direct path to model file (overrides save_dir and checkpoint)
        
    Returns:
        loaded_model: MDRNN model with loaded weights
        checkpoint_data: Dictionary containing all checkpoint information
        
    Examples:
        # Load by checkpoint number
        model, info = load_model(checkpoint=3)
        
        # Load by checkpoint number from specific directory
        model, info = load_model(save_dir='my_models', checkpoint=2)
        
        # Load by direct path
        model, info = load_model(model_path='trained_models/mdrnn_60000_steps.pt')
    """
    import os
    import glob
    
    # Determine the file path
    if model_path is not None:
        # Direct path provided
        file_path = model_path
    elif checkpoint is not None:
        # Find file by checkpoint number
        if not 1 <= checkpoint <= 5:
            raise ValueError(f"Checkpoint must be between 1 and 5, got {checkpoint}")
        
        # Find all model files in the directory (try steps first, then episodes for backwards compatibility)
        pattern_steps = os.path.join(save_dir, 'mdrnn_*_steps.pt')
        pattern_episodes = os.path.join(save_dir, 'mdrnn_*_episodes.pt')
        
        model_files = sorted(glob.glob(pattern_steps))
        if not model_files:
            # Fallback to old episode-based naming
            model_files = sorted(glob.glob(pattern_episodes))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {save_dir}")
        
        if len(model_files) < checkpoint:
            raise FileNotFoundError(
                f"Checkpoint {checkpoint} not found. Only {len(model_files)} checkpoints available: {model_files}"
            )
        
        # Get the checkpoint file (1-indexed)
        file_path = model_files[checkpoint - 1]
        print(f"Loading checkpoint {checkpoint}/{len(model_files)}: {os.path.basename(file_path)}")
    else:
        raise ValueError("Must provide either 'checkpoint' (1-5) or 'model_path'")
    
    # Load the checkpoint
    checkpoint_data = torch.load(file_path)
    
    # Create model with saved parameters
    loaded_model = MDRNN(
        state_dim=checkpoint_data['state_dim'], 
        action_dim=checkpoint_data['action_dim'], 
        hidden_dim=checkpoint_data['hidden_dim'], 
        n_gaussian=checkpoint_data['n_gaussian'],
        n_layers=checkpoint_data['n_layers']
    )
    
    # Load the state dict
    loaded_model.load_state_dict(checkpoint_data['model_state_dict'])
    
    # Print info about loaded model
    print(f"✓ Successfully loaded model from {file_path}")
    
    # Handle both old (episodes) and new (steps) checkpoint formats
    if 'total_steps_experienced' in checkpoint_data:
        print(f"  Steps experienced: {checkpoint_data['total_steps_experienced']}")
    if 'total_episodes_experienced' in checkpoint_data:
        print(f"  Episodes experienced: {checkpoint_data['total_episodes_experienced']}")
    
    print(f"  Training iteration: {checkpoint_data['training_iteration']}")
    print(f"  Final state loss: {checkpoint_data['state_loss']:.4f}")
    print(f"  Action dimension: {checkpoint_data['action_dim']} (one-hot: {checkpoint_data['action_dim'] > 1})")
    
    return loaded_model, checkpoint_data

def cartpole_dynamics(self, state, action, model_length):
    """Works with both symbolic and numeric inputs"""
    gravity = 9.81
    masscart = 1.0
    masspole = 0.1
    length = model_length  # Use local variable
    total_mass = masscart + masspole
    polemass_length = masspole * length

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
    
    temp = (u + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0/3.0 - masspole * costheta**2 / total_mass)
            )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    
    if is_symbolic:
        return [x_dot, xacc, theta_dot, thetaacc]
    else:
        return np.array([x_dot, xacc, theta_dot, thetaacc])
    
def cartpole_dynamics_rnn(self, state, action, hidden):
    """
    Predict the next state using trained RNN model.
    
    Args:
        state: Current state [x, x_dot, theta, theta_dot] (numpy array or list)
        action: Action to take (0 or 1, or float that gets converted to discrete)
        hidden: LSTM hidden state tuple (h, c)
    
    Returns:
        next_state: [x, x_dot, theta, theta_dot] as numpy array
        hidden: Updated LSTM hidden state
    """
    # Convert inputs to numpy if needed
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.float32)
    else:
        state = state.astype(np.float32)

    # Convert action to discrete (0 or 1)
    if action >= 0:
        discrete_action = 1
    else:
        discrete_action = 0
    
    # Convert discrete action to one-hot encoding (2D for CartPole)
    action_onehot = np.zeros(2, dtype=np.float32)
    action_onehot[discrete_action] = 1.0
    
    # Convert to tensors with proper shape (batch=1, seq_len=1, dim)
    state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 4)
    action_tensor = torch.from_numpy(action_onehot).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 2)
     
    # Forward pass
    with torch.no_grad():
        outputs = self.rnn_model(state_tensor, action_tensor, hidden)
 
    # Update the hidden state
    hidden = outputs['hidden']

    # Extract mixture parameters
    pi = outputs['pi'].squeeze(0).squeeze(0)      # (n_gaussian,)
    mu = outputs['mu'].squeeze(0).squeeze(0)      # (n_gaussian, state_dim)
    sigma = outputs['sigma'].squeeze(0).squeeze(0)  # (n_gaussian, state_dim)

    # For controller, use MEAN of highest-weight component (more stable)
    k = pi.argmax()
    next_state_np = mu[k].cpu().numpy()
    
    # # Sample next state from the mixture distribution
    # next_state = self.rnn_model.sample_prediction(pi, mu, sigma, temperature=1.0)
    # next_state_np = next_state.cpu().numpy()
    
    return next_state_np, hidden


class MPCController:
    def __init__(self, horizon=10, dt=0.02, recompute_every=1, model_length=0.5, wind_mu=0.0, wind_sigma=0.0):
        self.horizon = horizon
        self.dt = dt
        self.recompute_every = recompute_every
        self.force_mag = 10.0
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

    def get_action(self, obs, env=None, hidden=None):
        # obs_with_noise = obs.copy()
        # wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma)
        # obs_with_noise[2] += wind_disturbance
        self.mpc.x0 = np.array(obs, dtype=float).reshape(-1, 1)
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
    def __init__(self, controller='predictive',world_model='dynamics',horizon=10, dt=0.02, recompute_every=1, model_length=1.0, wind_mu=0.0, wind_sigma=0.0, rnn_model=None):
        self.horizon = horizon
        self.dt = dt
        self.recompute_every = recompute_every

        self.world_model = world_model
        self.rnn_model = rnn_model
        
        self.length = model_length
        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma
        self.control_model = controller

        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma

        self.force_mag = 10.0
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1

        if world_model == 'rnn' and rnn_model is not None:
            self.device = torch.device('cpu')  # or 'cuda' if available
            self.rnn_model = rnn_model.to(self.device)
            self.rnn_model.eval()
            
            # Pre-allocate tensors for reuse (will reshape as needed)
            self._state_tensor = None
            self._action_tensor = None

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

                # Generate all random actions at once
                actions[1:] = np.random.choice(
                    [-self.force_mag, self.force_mag],
                    size=(self.n_candidate_trajectories - 1, self.horizon, 1)
                )
    
                # noise_std = 2.0
                # for i in range(1, self.n_candidate_trajectories):
                #     # noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                #     # actions[i] = np.clip(
                #     #     self.nominal_trajectory + noise,
                #     #     -self.force_mag, self.force_mag
                #     # )
  
        
        elif self.control_model == 'random':
            # Random shooting
            # actions = np.random.uniform(
            #     low=-self.force_mag,
            #     high=self.force_mag,
            #     size=(self.n_candidate_trajectories, self.horizon, 1))
            # randomly sample from two options -self.force_mag and +self.force_mag
            actions = np.random.choice(
                [-self.force_mag, self.force_mag],
                size=(self.n_candidate_trajectories, self.horizon, 1)
            )
        # Discretize actions to -10 or +10
        actions = np.where(actions >= 0, self.force_mag, -self.force_mag)

        return actions
    
    def calculate_trajectories(self, env, initial_obs, actions, hidden):
        """
        Trajectory evaluation using Euler integration.
        
        Args:
            env: Environment (for compatibility, not used in vectorized version)
            initial_obs: Initial state (4,)
            actions: Action trajectories (n_traj, horizon, 1)
            hidden: Initial hidden state (tuple of tensors for LSTM)
            
        Returns:
            observations: (n_traj, horizon, 4)
        """
        # Initialize output arrays
        observations = np.zeros([self.n_candidate_trajectories, self.horizon, env.observation_space.shape[0]])
        
        # Loop over trajectories
        for i in range(self.n_candidate_trajectories):
            current_state = initial_obs.copy()  # Copy the state
            
            # Initialize hidden state for this trajectory
            if self.world_model == 'rnn' and hidden is not None:
                # Properly clone the hidden state for each trajectory
                # Hidden is typically a tuple of (h, c) for LSTM
                if isinstance(hidden, tuple):
                    current_hidden = tuple(h.clone() for h in hidden)
                else:
                    current_hidden = hidden.clone()
            else:
                current_hidden = None

            # Simulate forward in time
            for step in range(self.horizon):
                # Get actions for this time step 
                action = actions[i, step, 0]

                # Compute next state
                if self.world_model == 'dynamics':
                    derivatives = cartpole_dynamics(self, current_state, action, self.length)
                    # Euler integration: x_{t+1} = x_t + dt * f(x_t, u_t)
                    current_state = current_state + derivatives * self.dt
                elif self.world_model == 'rnn':
                    current_state, current_hidden = cartpole_dynamics_rnn(
                        self, current_state, action, current_hidden
                    )
                
                observations[i, step] = current_state

        return observations
        
    def get_action(self, obs, env, hidden):
        """Handle trajectory-based methods (random, predictive)."""
        
        # Initialize nominal trajectory for predictive methods
        if self.control_model=='predictive':
            if not hasattr(self, 'nominal_trajectory'):
                    self.nominal_trajectory = np.zeros((self.horizon, 1))       

        # Generate action trajectories
        actions = self.generate_action_trajectories(env, obs)
        
        # Evaluate trajectories
        observations = self.calculate_trajectories(env, obs, actions, hidden)
        
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
                    10*obs[:, :, 0]**2 +           # x^2  
                    obs[:, :, 3]**2 +           # theta_dot^2
                    obs[:, :, 1]**2)            # x_dot^2
        
        action_cost = 0.1 * actions[:, :, 0]**2   # Match MPC's rterm
        
        reward = -(stage_cost + action_cost)  # Negative because we maximize reward
        
        return reward.sum(axis=1)

class StepSamplingController:
    def __init__(self, controller='predictive',world_model='dynamics',horizon=10, dt=0.02, recompute_every=1, model_length=1.0, wind_mu=0.0, wind_sigma=0.0, rnn_model=None):
        self.horizon = horizon
        self.dt = dt
        self.recompute_every = recompute_every

        self.world_model = world_model
        self.rnn_model = rnn_model
        
        self.length = model_length
        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma
        self.control_model = controller

        self.wind_mu = wind_mu
        self.wind_sigma = wind_sigma

        self.force_mag = 10.0
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

                # Generate all random actions at once
                actions[1:] = np.random.choice(
                    [-self.force_mag, self.force_mag],
                    size=(self.n_candidate_trajectories - 1, self.horizon, 1)
                )
    
                # noise_std = 2.0
                # for i in range(1, self.n_candidate_trajectories):
                #     # noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                #     # actions[i] = np.clip(
                #     #     self.nominal_trajectory + noise,
                #     #     -self.force_mag, self.force_mag
                #     # )
  
        
        elif self.control_model == 'random':
            # Random shooting
            # actions = np.random.uniform(
            #     low=-self.force_mag,
            #     high=self.force_mag,
            #     size=(self.n_candidate_trajectories, self.horizon, 1))
            # randomly sample from two options -self.force_mag and +self.force_mag
            actions = np.random.choice(
                [-self.force_mag, self.force_mag],
                size=(self.n_candidate_trajectories, self.horizon, 1)
            )
        # Discretize actions to -10 or +10
        actions = np.where(actions >= 0, self.force_mag, -self.force_mag)

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
                if self.world_model == 'dynamics':
                    derivatives = cartpole_dynamics(self, current_state, action, self.length)
                    current_state = current_state + derivatives * self.dt
                    # Euler integration: x_{t+1} = x_t + dt * f(x_t, u_t)
                    current_state = current_state + derivatives * self.dt
                elif self.world_model == 'rnn':
                    current_state = cartpole_dynamics_rnn(self, current_state, action)

                # Add noise to the state
                # current_state += np.random.normal(self.wind_mu, self.wind_sigma, size=current_state.shape)

                # wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma, size=current_state.shape)
                # wind_disturbance[2] += np.random.normal(self.wind_mu, self.wind_sigma)  # Extra noise on theta
                # current_state += wind_disturbance

                observations[i, step] = current_state

        return observations
    
    def get_action(self, obs, env):
        """Handle trajectory-based methods (random, predictive)."""
        actions = np.zeros([self.horizon, 1])
        observations = np.zeros([self.horizon, env.observation_space.shape[0]])

        current_state = obs

        # Simulate forward in time
        for step in range(self.horizon):
            # Get states for two actions
            action1 = -self.force_mag
            action2 = self.force_mag           
                
            # Compute derivatives for all trajectories simultaneously
            if self.world_model == 'dynamics':
                derivatives1 = cartpole_dynamics(self, current_state, action1, self.length)
                derivatives2 = cartpole_dynamics(self, current_state, action2, self.length)
                current_state1 = current_state + derivatives1 * self.dt
                current_state2 = current_state + derivatives2 * self.dt
                # Euler integration: x_{t+1} = x_t + dt * f(x_t, u_t)
                current_state1 = current_state + derivatives1 * self.dt
                current_state2 = current_state + derivatives2 * self.dt
            elif self.world_model == 'rnn':
                current_state1 = cartpole_dynamics_rnn(self, current_state, action1)
                current_state2 = cartpole_dynamics_rnn(self, current_state, action2)

            # Add noise to the state
            current_state1 += np.random.normal(self.wind_mu, self.wind_sigma, size=current_state.shape)
            current_state2 += np.random.normal(self.wind_mu, self.wind_sigma, size=current_state.shape)

            wind_disturbance = np.random.normal(self.wind_mu, self.wind_sigma, size=current_state.shape)
            wind_disturbance[2] += np.random.normal(self.wind_mu, self.wind_sigma)  # Extra noise on theta
            current_state1 += wind_disturbance
            current_state2 += wind_disturbance

            # Evaluate the two states with respect to the cost function
            action, current_state = self.get_optimal_action(current_state1, current_state2, action1, action2)
            actions[step,:] = action
            observations[step,:] = current_state

        return actions, observations
 
    def get_optimal_action(self, current_state1, current_state2, action1, action2):
        
        # Match MPC objective exactly
        stage_cost1 = (10 * current_state1[2]**2 +     # 10*theta^2
                    current_state1[0]**2 +           # x^2  
                    current_state1[3]**2 +           # theta_dot^2
                    current_state1[1]**2)            # x_dot^2
        
        stage_cost2 = (10 * current_state2[2]**2 +     # 10*theta^2
                    current_state2[0]**2 +           # x^2  
                    current_state2[3]**2 +           # theta_dot^2
                    current_state2[1]**2)            # x_dot^2
        
        action_cost1 = 0.1 * action1**2   # Match MPC's rterm
        action_cost2 = 0.1 * action2**2   # Match MPC's rterm
        
        reward1 = -(stage_cost1 + action_cost1)  # Negative because we maximize reward
        reward2 = -(stage_cost2 + action_cost2)  # Negative because we maximize reward

        if reward1>reward2:
            optimal_action = action1
            predicted_state = current_state1
        else:
            optimal_action = action2
            predicted_state = current_state2
        return optimal_action, predicted_state

def evaluate_mpc_controllers(controller, world_model, horizons, recompute_intervals, dt, results_folder="../results/PerformanceResults/", 
                             episode_length=500, num_episodes=20, seed=42, 
                             length_ratios=[0.6, 0.8, 1.0, 1.2, 1.6], wind_mus=[0.0], wind_sigmas=[0.0], init_angles=[0.0]):
    """Evaluate MPC with various parameters and save each configuration separately."""
    
    if min(recompute_intervals) > min(horizons):
        raise ValueError("The smallest recompute interval must be less than or equal to the smallest horizon.")
    
    os.makedirs(results_folder, exist_ok=True)

    # Get the pole lenght from the environment
    env = gym.make("CartPole-v1", render_mode=None)

    # Initialize the rnn model if needed
    if world_model == 'rnn':
        hidden_dim = 256

        # Load the model
        # rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_400000_steps.pt')
        # rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_1200000_steps.pt')
        rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_5000000_steps.pt')    

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rnn_model = rnn_model.to(device)
        
        # Set model to evaluation mode
        rnn_model.eval()
    else:
        rnn_model = None

    overall_timing = []
    
    for init_angle in init_angles:
        for wind_mu in wind_mus:
            for wind_sigma in wind_sigmas:
                for h in horizons:
                    for e in recompute_intervals:
                        for ratio in length_ratios:
                            # Get the start_time
                            start_time = time.time()

                            # Take the pole length from the environment
                            true_length = env.unwrapped.length
                                                        
                            if controller == 'mpc':
                                mpc = MPCController(horizon=h, recompute_every=e, 
                                                model_length=true_length, wind_mu=wind_mu, wind_sigma=wind_sigma)
                            elif controller == 'predictive':
                                mpc = SamplingController(controller=controller, world_model=world_model, horizon=h, recompute_every=e,
                                                model_length=true_length, wind_mu=wind_mu, wind_sigma=wind_sigma, rnn_model=rnn_model)
                            elif controller == 'random':
                                mpc = SamplingController(controller=controller, world_model=world_model, horizon=h, recompute_every=e,
                                                model_length=true_length, wind_mu=wind_mu, wind_sigma=wind_sigma, rnn_model=rnn_model)


                            episode_lengths = []
                            episode_times = []
                            integrated_errors = []
                            
                            for ep in range(num_episodes):
                                episode_start_time = time.time()

                                env = gym.make("CartPole-v1", render_mode=None)

                                # Set the environment time step
                                env.unwrapped.tau = dt

                                # Change the pole length in simulation
                                env_length = ratio * true_length
                                env.unwrapped.length = env_length

                                # Reset the env with seed
                                obs, _ = env.reset(seed=seed + ep)

                                # Change the initial angle
                                new_state = obs.copy()
                                new_state[2] += init_angle
                                env.unwrapped.state = new_state
                                obs = new_state
                               
                                length, step = 0, 0
                                done = False
                                states = []

                                # Initialize the rnn model hidden state if needed
                                if world_model == 'rnn':

                                    # Initialize the hidden state
                                    hidden = rnn_model.init_hidden(batch_size=1, device=device)   
                                else:
                                    hidden = None

                                while not done and length < episode_length:
                                    if step % e == 0:
                                        within_step = 0
                                        trajectory, predictions = mpc.get_action(obs, env, hidden)
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

                                    # Add disturbance to the observation
                                    wind_disturbance = np.zeros(obs.shape)
                                    wind_disturbance[2] += np.random.normal(wind_mu, wind_sigma)  # Extra noise on theta
                                    obs += wind_disturbance
                                        
                                    states.append(obs)
                                    step += 1
                                    length += 1

                                    # Get the hidden state for next step if RNN
                                    if world_model == 'rnn' and rnn_model is not None:
                                        # Convert discrete action to one-hot encoding (2D for CartPole)
                                        action_onehot = np.zeros(2, dtype=np.float32)
                                        action_onehot[action] = 1
                                        
                                        # Convert to tensors with proper shape (batch=1, seq_len=1, dim)
                                        state_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 4)
                                        action_tensor = torch.from_numpy(action_onehot).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 2)
                                        
                                        # Forward pass
                                        with torch.no_grad():
                                            outputs = rnn_model(state_tensor, action_tensor, hidden)
                                    
                                        # Update the hidden state
                                        hidden = outputs['hidden']
                                
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
                                    'world_model': world_model,
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

                            filename = f"mpc__cont{controller}_model{world_model}_h{h}_e{e}_r{ratio:.1f}_wmu{wind_mu:.2f}_wsig{wind_sigma:.2f}_iang{init_angle:.2f}.pkl"
                            filepath = os.path.join(results_folder, filename)
                            
                            with open(filepath, 'wb') as f:
                                pickle.dump(config_data, f)
                            
                            overall_timing.append({
                                "controller": controller,
                                "world_model": world_model,
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

    true_cost -= (true_state[1]**2) + 10*(true_state[0]**2)
    true_cost -= (10 * (true_state[2]**2) + true_state[3]**2)
    true_cost -= 0.1 * (action**2)

    predicted_cost -= (predicted_state[1]**2) + 10*(predicted_state[0]**2)
    predicted_cost -= (10 * (predicted_state[2]**2) + predicted_state[3]**2)
    predicted_cost -= 0.1 * (action**2)

    cost_pe = abs(true_cost - predicted_cost)

    return cost_pe

import time

def run_single_episode_adaptive_with_plots(controller_type='mpc', world_model='dynamics', horizon=10, dt=0.02, 
                                 recompute_every=1, ratio=0.5, wind_mu=0.0, wind_sigma=0.0,
                                 episode_length=500, seed=42, init_angle=0.0, adaptive_recompute=False, adaptive_window=20):
    """
    Run a single episode with the specified controller and plot angle and actions over time.
    
    Args:
        adaptive_recompute: If True, adaptively adjust recompute_every based on cost prediction errors
    """
    start_time = time.time()
    
    # Initialize the rnn model if needed
    if world_model == 'rnn':
        hidden_dim = 256

        # Load the model
        # rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_400000_steps.pt')
        # rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_1200000_steps.pt')
        rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_10000000_steps.pt')
        

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rnn_model = rnn_model.to(device)
        
        # Set model to evaluation mode
        rnn_model.eval()
    else:
        rnn_model = None

    # Get the pole length from the environment
    env = gym.make("CartPole-v1", render_mode=None)
    
    # Set the environment time step
    env.unwrapped.tau = dt
    
    # Change the pole length in simulation
    model_length = env.unwrapped.length
    env_length = ratio * model_length
    env.unwrapped.length = env_length
    env.reset()

    print(f"True length: {env_length}, Model length: {model_length}")
    
    # Initialize adaptive recompute parameters
    current_recompute = recompute_every  # Start with initial value
    cost_error_buffer = []
    adaptation_window = adaptive_window
    recompute_history = []
    running_average_history = []  # Track the running average at each step
    
    # Initialize the controller
    if controller_type == 'mpc':
        controller = MPCController(horizon=horizon, dt=dt, 
                                 recompute_every=current_recompute, model_length=model_length, 
                                 wind_mu=wind_mu, wind_sigma=wind_sigma)
        
    elif controller_type in ['predictive', 'random']:
        controller = SamplingController(controller=controller_type, world_model=world_model, horizon=horizon, dt=dt, 
                                      recompute_every=current_recompute, 
                                      model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, rnn_model=rnn_model)
    
    elif controller_type in ['stepsample']:
        controller = StepSamplingController(controller=controller_type, world_model=world_model, horizon=horizon, dt=dt, 
                                      recompute_every=recompute_every, 
                                      model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, rnn_model=rnn_model)

    else:
        raise ValueError("controller_type must be 'mpc', 'predictive', or 'random'")
    
    # Initialize environment (create fresh one with tau set)
    env = gym.make("CartPole-v1", render_mode=None)

    # Set the environment time step
    env.unwrapped.tau = dt
    
    # Change the pole length in simulation
    true_length = env.unwrapped.length
    env_length = ratio * true_length
    env.unwrapped.length = env_length
    env.reset()
    obs, _ = env.reset(seed=seed)

    # Change the initial angle
    new_state = obs.copy()
    new_state[2] += init_angle
    env.unwrapped.state = new_state
    obs = new_state
    
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

    # Initialize hidden state for RNN if needed
    hidden = None
    if world_model == 'rnn' and rnn_model is not None:
        device = torch.device('cpu')
        hidden = rnn_model.init_hidden(batch_size=1, device=device)
    
    while length < episode_length and not done:  
        # Calculate current running average (what we're actually using for decisions)
        if len(cost_error_buffer) > 0:
            current_running_avg = np.mean(cost_error_buffer)
        else:
            current_running_avg = np.nan  # No data yet
        
        running_average_history.append(current_running_avg)
        
        # Adaptive recompute adjustment
        if adaptive_recompute and len(cost_error_buffer) == adaptation_window:
            running_avg = np.mean(cost_error_buffer)
            
            # Use recent history for comparison (last 100 steps or all if fewer)
            recent_window = min(100, len(cost_prediction_errors))
            if len(cost_prediction_errors) >= recent_window:
                recent_median = np.median(cost_prediction_errors[-recent_window:])
            else:
                recent_median = np.median(cost_prediction_errors) if len(cost_prediction_errors) > 0 else 0
            
            # Add hysteresis: only change if difference is significant
            threshold = 0.05 * recent_median if recent_median > 0 else 0.01
            
            if running_avg < recent_median - threshold:
                # Model is doing well, can afford to recompute less often
                current_recompute = min(current_recompute + 1, 20)  # Cap at 20
                print(f"Step {step}: Increasing recompute to {current_recompute} (avg error: {running_avg:.4f}, recent median: {recent_median:.4f})")
            elif running_avg > recent_median + threshold:
                # Model is struggling, recompute more often
                current_recompute = max(current_recompute - 1, 1)  # Never below 1
                print(f"Step {step}: Decreasing recompute to {current_recompute} (avg error: {running_avg:.4f}, recent median: {recent_median:.4f})")
            
            # Update controller's recompute parameter
            controller.recompute_every = current_recompute
            
            # Clear buffer
            cost_error_buffer = []
        
        recompute_history.append(current_recompute)
        
        # Get action from controller
        if step % current_recompute == 0:
            within_step = 0
            trajectory, all_predictions = controller.get_action(obs, env, hidden)
            trajectory = trajectory.flatten()
        else:
            within_step = (step % current_recompute)

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

        # Add disturbance to the observation
        wind_disturbance = np.zeros(obs.shape)
        wind_disturbance[2] += np.random.normal(wind_mu, wind_sigma)  # Extra noise on theta
        obs += wind_disturbance
        
        prediction_errors.append(obs - predictions)
        step += 1
        length += 1

        cost_prediction_error = calculate_cost_error(controller_type, obs, predictions, discrete_action)
        cost_prediction_errors.append(cost_prediction_error)

        # Get the hidden state for next step if RNN
        if world_model == 'rnn' and rnn_model is not None:
            # Convert discrete action to one-hot encoding (2D for CartPole)
            action_onehot = np.zeros(2, dtype=np.float32)
            action_onehot[discrete_action] = 1.0
            
            # Convert to tensors with proper shape (batch=1, seq_len=1, dim)
            state_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 4)
            action_tensor = torch.from_numpy(action_onehot).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 2)
            
            # Forward pass
            with torch.no_grad():
                outputs = rnn_model(state_tensor, action_tensor, hidden)
        
            # Update the hidden state
            hidden = outputs['hidden']
        
        # Add to buffer for adaptive recompute
        if adaptive_recompute:
            cost_error_buffer.append(cost_prediction_error)

    env.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert to numpy arrays
    time_steps = np.array(time_steps)
    angles = np.array(angles)
    actions_taken = np.array(actions_taken)
    cart_positions = np.array(cart_positions)
    prediction_errors = np.array(prediction_errors).T  
    recompute_history = np.array(recompute_history)
    running_average_history = np.array(running_average_history)

# Create plots (6 subplots now - split last plot into 2)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 14))
    
    # Plot 1: Angle over time
    ax1.plot(time_steps, angles, 'b-', linewidth=2, label='Pole Angle (radians)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target (0)')
    ax1.axhline(y=-0.2095, color='r', linestyle=':', alpha=0.3, label='Failure Limits')
    ax1.axhline(y=0.2095, color='r', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (radians)')
    title = f'Pole Angle Over Time - {controller_type.upper()} Controller - WORLD Model: {world_model}\n'
    title += f'H={horizon}, Recompute={recompute_every}{f"(adaptive every {adaptation_window})" if adaptive_recompute else ""}, Length={model_length:.3f}m, Wind σ={wind_sigma:.2f}, Wind μ={wind_mu:.2f}'
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-0.45, 0.45])

    # Plot 2: Cart position over time
    ax2.plot(time_steps, cart_positions, 'm-', linewidth=2, label='Cart Position')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Center Position')
    ax2.axhline(y=2.4, color='r', linestyle=':', alpha=0.3, label='Failure Limits')
    ax2.axhline(y=-2.4, color='r', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Cart Position Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-4.8, 4.8])
    
    # Plot 3: Actions over time
    ax3.plot(time_steps, actions_taken, 'g-', linewidth=2, label='Control Action')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No Force')
    ax3.axhline(y=10.0, color='r', linestyle=':', alpha=0.5, label='Force Limits')
    ax3.axhline(y=-10.0, color='r', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Control Actions Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-12, 12])

    # Plot 4: Prediction errors (with transparency and reordering)
    colors = ['b', 'g', 'orange', 'purple']
    labels = ['x (cart position)', 'ẋ (cart velocity)', 'θ (pole angle)', 'θ̇ (angular velocity)']
    
    # Plot cart position and velocity first (behind)
    ax4.plot(time_steps, prediction_errors[0], color=colors[0], label=labels[0], linewidth=2, alpha=0.4)
    ax4.plot(time_steps, prediction_errors[1], color=colors[1], label=labels[1], linewidth=2, alpha=0.4)
    
    # Plot angle errors on top (more visible)
    ax4.plot(time_steps, prediction_errors[2], color=colors[2], label=labels[2], linewidth=2, alpha=0.6)
    ax4.plot(time_steps, prediction_errors[3], color=colors[3], label=labels[3], linewidth=2, alpha=0.6)

    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Prediction Error')
    ax4.set_title('Prediction Errors Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([-0.5, 0.5])

    # Plot 5: Cost prediction error with running average
    ax5.plot(time_steps, cost_prediction_errors, 'm-', linewidth=1, alpha=0.3, label='Cost Prediction Error')
    ax5.plot(time_steps, running_average_history, 'k-', linewidth=2, label=f'Running Average (window={adaptation_window})')
    
    # Add recent median if adaptive
    if adaptive_recompute and len(cost_prediction_errors) > 0:
        recent_window = min(100, len(cost_prediction_errors))
        if len(cost_prediction_errors) >= recent_window:
            recent_median = np.median(cost_prediction_errors[-recent_window:])
            ax5.axhline(y=recent_median, color='b', linestyle='--', alpha=0.5, 
                       label=f'Recent Median (window={recent_window}): {recent_median:.4f}')
    
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Cost Prediction Error')
    ax5.set_title('Cost Prediction Error Over Time')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim([0, 1])

    # Plot 6: Recompute level over time
    ax6.plot(time_steps, recompute_history, 'r-', linewidth=2, alpha=0.7, label='Recompute Level')
    
    # Add average recompute level
    if adaptive_recompute:
        avg_recompute = np.mean(recompute_history)
        ax6.axhline(y=avg_recompute, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Average Recompute ({avg_recompute:.1f})')
    
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Recompute Every N Steps')
    ax6.set_title('Adaptive Recompute Level Over Time')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_ylim([0, 8])

    plt.tight_layout()
    plt.savefig('mpc_cartpole_adaptive_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*50}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time*1000:.1f} ms)")
    print(f"Episode length: {length} steps ({length * dt:.2f} seconds simulated)")
    print(f"Steps per second: {length / total_time:.2f}")

def run_single_episode_with_plots(controller_type='mpc', world_model='dynamics', horizon=10, dt=0.02, 
                                 recompute_every=1, model_length=0.5, wind_mu=0.0, wind_sigma=0.0,
                                 episode_length=500, seed=42, init_angle=0.0):
    """
    Run a single episode with the specified controller and plot angle and actions over time.
    """
    # Initialize the rnn model if needed
    if world_model == 'rnn':
        hidden_dim = 256

        # Load the model
        rnn_model, checkpoint_data = load_model(model_path='../trained_models/mdrnn_500000_steps.pt')

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rnn_model = rnn_model.to(device)
        
        # Set model to evaluation mode
        rnn_model.eval()
    else:
        rnn_model = None

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
        controller = SamplingController(controller=controller_type, world_model=world_model, horizon=horizon, dt=dt, 
                                      recompute_every=recompute_every, 
                                      model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, rnn_model=rnn_model)
    elif controller_type in ['stepsample']:
        controller = StepSamplingController(controller=controller_type, world_model=world_model, horizon=horizon, dt=dt, 
                                      recompute_every=recompute_every, 
                                      model_length=model_length, wind_mu=wind_mu, wind_sigma=wind_sigma, rnn_model=rnn_model)

    else:
        raise ValueError("controller_type must be 'mpc', 'predictive', or 'random'")
    
    # Initialize environment (create fresh one with tau set)
    env = gym.make("CartPole-v1", render_mode=None)
    env.unwrapped.tau = dt  
    obs, _ = env.reset(seed=seed)

    # Change the initial angle
    new_state = obs.copy()
    new_state[2] += init_angle
    env.unwrapped.state = new_state
    obs = new_state
    
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
    
    while length < episode_length and not done:  
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
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 10))
    
    # Plot 1: Angle over time
    ax1.plot(time_steps, angles, 'b-', linewidth=2, label='Pole Angle (radians)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target (0)')
    ax1.axhline(y=-0.2095, color='r', linestyle=':', alpha=0.3, label='Failure Limits')
    ax1.axhline(y=0.2095, color='r', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title(f'Pole Angle Over Time - {controller_type.upper()} Controller - WORLD Model: {world_model}\n'
                  f'H={horizon}, Recompute={recompute_every}, Length={model_length:.3f}m, '
                  f'Wind σ={wind_sigma:.2f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-0.45, 0.45])

    # Plot cart position over time
    ax2.plot(time_steps, cart_positions, 'm-', linewidth=2, label='Cart Position')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Center Position')
    ax2.axhline(y=2.4, color='r', linestyle=':', alpha=0.3, label='Failiure Limits')
    ax2.axhline(y=-2.4, color='r', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Cart Position Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-4.8, 4.8])
    
    # Plot 2: Actions over time
    ax3.plot(time_steps, actions_taken, 'g-', linewidth=2, label='Control Action')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No Force')
    ax3.axhline(y=10.0, color='r', linestyle=':', alpha=0.5, label='Force Limits')
    ax3.axhline(y=-10.0, color='r', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Force (N)')
    ax3.set_title('Control Actions Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-12, 12])

    # Plot 3: prediction errors
    colors = ['b', 'g', 'orange', 'purple']
    labels = ['x (cart position)', 'ẋ (cart velocity)', 'θ (pole angle)', 'θ̇ (angular velocity)']
    
    for i in range(4):
        ax4.plot(time_steps, prediction_errors[i], color=colors[i], label=labels[i], linewidth=2)

    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Prediction Error')
    ax4.set_title('Prediction Errors Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 4: cost prediction error
    ax5.plot(time_steps, cost_prediction_errors, 'm-', linewidth=2, label='Cost Prediction Error')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Cost Prediction Error')
    ax5.set_title('Cost Prediction Error Over Time')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    plt.tight_layout()
    plt.savefig('mpc_cartpole_results.png', dpi=150)
    plt.show()

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
                                world_model='dynamics',
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
        (df['init_angle'] == init_angle) &
        (df['world_model'] == world_model)
    ]
    
    if filtered_data.empty:
        print(f"No data found for controller {controller}, wind_mu={wind_mu}, wind_sigma={wind_sigma}, init_angle={init_angle}, world_model={world_model}")
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
    
    fig, axes = plt.subplots(1, len(plot_ratios), figsize=(4*len(plot_ratios)+4, 6))
    
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

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Pole Length Error\n(controller={controller}, world_model={world_model}, wind_μ={wind_mu}, wind_σ={wind_sigma}, init_angle={init_angle:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_4d_performance_heatmaps_wmu{wind_mu}_wsig{wind_sigma}_iang{init_angle:.2f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def plot_wind_mu_heatmaps(results_folder="../results/PerformanceResults/", 
                         controller='mpc',
                         world_model='dynamics',
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
        (df['init_angle'] == init_angle) &
        (df['world_model'] == world_model)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, sigma={wind_sigma}, init_angle={init_angle}, world_model={world_model}")
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
    
    fig, axes = plt.subplots(1, len(plot_wind_mus), figsize=(4*len(plot_wind_mus)+4, 6))
    
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
        
        title = f'Wind μ = {wind_mu:.2f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{round(perf_matrix[i, j])}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Mean\n(controller={controller}, world_model={world_model}, length_ratio={length_ratio}, wind_σ={wind_sigma:.2f}, init_angle={init_angle:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_mu_heatmaps_r{length_ratio:.1f}_s{wind_sigma:.1f}_iang{init_angle:.2f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return 

def plot_wind_sigma_heatmaps(results_folder="../results/PerformanceResults/", 
                            controller='mpc',
                            world_model='dynamics',
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
        (df['init_angle'] == init_angle) &
        (df['world_model'] == world_model)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, mu={wind_mu}, init_angle={init_angle}, world_model={world_model}")
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
    
    fig, axes = plt.subplots(1, len(plot_wind_sigmas), figsize=(4*len(plot_wind_sigmas)+4, 6))
    
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
        
        title = f'Wind σ = {wind_sigma:.2f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        
        for i in range(len(recomputes)):
            for j in range(len(horizons)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < (vmin + vmax) / 2 else 'black'
                    axes[idx].text(j, i, f'{round(perf_matrix[i, j])}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9, fontweight='bold')

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Wind Std Dev\n(controller={controller}, world_model={world_model}, length_ratio={length_ratio}, wind_μ={wind_mu:.2f}, init_angle={init_angle:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_wind_sigma_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}_iang{init_angle:.2f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return

def plot_init_angle_heatmaps(results_folder="../results/PerformanceResults/", 
                                      controller='mpc',
                                      world_model='dynamics',
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
        (df['wind_sigma'] == wind_sigma) &
        (df['world_model'] == world_model)
    ]
    
    if filtered_data.empty:
        print(f"No data found for ratio={length_ratio}, mu={wind_mu}, sigma={wind_sigma}, world_model={world_model}")
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
    
    fig, axes = plt.subplots(1, len(plot_init_angles), figsize=(4*len(plot_init_angles)+4, 6))
    
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

    fig.suptitle(f'MPC Performance: Horizon × Recompute × Initial Angle\n(controller={controller}, world_model={world_model}, length_ratio={length_ratio}, wind_μ={wind_mu:.2f}, wind_σ={wind_sigma:.2f}, N={num_episodes} episodes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"mpc_init_angle_heatmaps_r{length_ratio:.1f}_m{wind_mu:.1f}_s{wind_sigma:.1f}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return