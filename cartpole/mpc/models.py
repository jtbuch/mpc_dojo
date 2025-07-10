import numpy as np
import mujoco
import copy
from scipy.linalg import expm, svd
import torch
from stable_baselines3 import TD3

class MPCShittyBird:
    """ Doesn't aspire to much. """
    def __init__(self, n_actions, recompute = 10, planning_width = 5, n_planning = 10, reward_type='discrete', model_type='mujoco', action_cost=0.0, control_model='random', value='env', rl_model='50k'):
        # Model parameters
        self.n_actions = n_actions      # Number of actions available
        self.recompute = recompute      # How often to recompute the control trajectory
        self.control_trajectory = None  # Stores cached control inputs
        self.control_step = 0           # Current step in the control trajectory

        # Trajectory search parameters, probably shouldn't be here...
        self.n_planning     = n_planning        # Number of planning steps
        self.planning_width = planning_width         # Number of trajectories to sample
        self.epsilon        = 0.2       # Epsilon greedy action selection

        self.reward_type = reward_type # rewards are discrete or continuous
        self.model_type  = model_type   # 'mujoco' or 'linear_full' or 'linear_reduced_<int>'
        self.action_cost = action_cost
        self.control_model = control_model  # 'random' or 'predictive' or 'rl'

        self.rl_model = rl_model  # Name of the RL model to use, if applicable
        self.value = value  # 'env' or 'rl' to use MuJoCo env or RL model
        
        if self.value == 'rl':
            # If using RL, we need to initialize the model
            

            # self.model = TD3.load("td3_invertedpendulum_continuous_50k_steps")

            model_files = {
                '10k': "../rl_models/td3_invertedpendulum_continuous_10k_steps",
                '50k': "../rl_models/td3_invertedpendulum_continuous_50k_steps", 
                '100k': "../rl_models/td3_invertedpendulum_continuous_100k_steps",
                '300k': "../rl_models/td3_invertedpendulum_continuous_300k_steps", 
                '500k': "../rl_models/td3_invertedpendulum_continuous_500k_steps"
            }
            
            # Remove the for loop and directly check the model name
            if self.rl_model in model_files:
                try:
                    self.model = TD3.load(model_files[self.rl_model])
                    print(f"✓ Loaded {self.rl_model} model")
                except Exception as e:
                    print(f"✗ Failed to load {self.rl_model} model: {e}")
            else:
                print(f"✗ Unknown model name: {self.rl_model}. Available: {list(model_files.keys())}")
  
    def init_transition_model(self, world):
        """  """
        self.world_transition = world.transition
        self.world_reward     = world.reward

    def transition_model(self, state, action):
        """ Bird's model of the world's transition structure. """
        next_state = self.world_transition(state, action)
        reward     = self.world_reward(next_state)
        return next_state, reward
    
    def policy(self, state):
        """ Could implement better policies here. """

        # Only occasionally recompute stored policy (really, control trajectory)
        if self.control_step % self.recompute == 0:
            self.control_trajectory = self.get_forward_policy(state)

        # Return the first action in the control trajectory
        return self.control_trajectory[self.control_step]

    def register_action(self):
        """ Register that an action has been taken. """
        self.control_step += 1
        self.control_step %= self.recompute

    def restore_state(self, env, qpos, qvel):
        # Restore the environment's state
        env.unwrapped.data.qpos[:] = qpos
        env.unwrapped.data.qvel[:] = qvel

        # We need to call mujoco.mj_forward to update the simulation state with the restored positions and velocities.
        # This function ensures that all the states are correctly computed after restoring the qpos and qvel.
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    def predict_next_state_inverted_pendulum(self, env, obs, action):
        """
        Predict next state for InvertedPendulum-v5 using linearized model,
        with rank determined by self.model_type:
        - 'linear_full' uses full rank A
        - 'linear_reduced_<int>' uses that integer as rank for reduced A

        Parameters:
        - env: the gymnasium environment object.
        - obs: current state (numpy array or scalar, shape=(4,))
        - action: control input (numpy array or scalar, shape=(1,))

        Returns:
        - next_obs: predicted next state (numpy array)
        """

        # # Continuous-time linearized system matrices
        # A_c = np.array([
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        #     [0, -2.90535, -0.08367, 0.19671],
        #     [0, 29.89257, 0.19671, -2.02391]
        # ])

        # B_c = np.array([
        #     [0],
        #     [0],
        #     [8.36743],
        #     [-19.67105]
        # ])

        # From mujoco using mujoco.mjd_transitionFD(model, data, epsilon, centered, A, B, None, None)
        A_d = np.array([
            [1.00000000, -0.00111508,  0.01996688,  0.00007550],
            [0.00000000,  1.01148765,  0.00007550,  0.01922222],
            [0.00000000, -0.05575393,  0.99834413,  0.00377490],
            [0.00000000,  0.57438229,  0.00377490,  0.96111076]
        ])

        B_d = np.array([
            [ 0.00331173],
            [-0.00754979],
            [ 0.16558646],
            [-0.37748958]
        ])

        # Discretize A and B
        dt = env.unwrapped.model.opt.timestep
        n = A_d.shape[0]
        m = B_d.shape[1]

        # Determine rank from self.model_type
        rank = None
        if hasattr(self, 'model_type'):
            if self.model_type == 'linear_full':
                rank = None
            elif self.model_type.startswith('linear_reduced_'):
                try:
                    rank = int(self.model_type.split('_')[-1])
                    if rank > n:
                        rank = n  # cap rank at full size
                except ValueError:
                    rank = None  # fallback to full rank if parsing fails

        # Apply rank reduction if requested
        if rank is not None and rank < A_d.shape[0]:
            U, S, Vh = svd(A_d)
            S_reduced = np.zeros_like(S)
            S_reduced[:rank] = S[:rank]
            A_d = U @ np.diag(S_reduced) @ Vh

        obs = np.array(obs)
        action = np.atleast_1d(action)

        next_obs = A_d @ obs + B_d @ action

        return next_obs
    
    def take_step(self, env, obs, action):
        """
        Compute reward based on either MuJoCo env step (model_type='mujoco') or linearized model prediction (reduced ('linear') or not ('reduced')).

        Parameters:
        - env: the gymnasium environment object.
        - obs: current state (numpy array or scalar).
        - action: control input/action (numpy array or scalar).

        Returns:
        - reward: reward computed based on selected method.
        """
        if self.model_type == 'mujoco':
            # Use MuJoCo step to get next obs and reward
            next_obs, reward, done, truncated, info = env.step(action)
            if self.reward_type == 'discrete':
                return reward, next_obs
            elif self.reward_type == 'continuous':
                if self.value == 'env':
                    # Use MuJoCo's reward directly
                    reward = -np.abs(next_obs[1])  # negative absolute angle
                elif self.value == 'rl':
                    # First, get the action from the learned policy
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Then get Q(s, π(s)) which approximates V(s)
                    critic_output = self.model.critic(
                        torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32),
                        torch.as_tensor(action.reshape(1, -1), dtype=torch.float32)
                    )
                    reward = critic_output[0].detach().numpy()[0][0]
                
                return reward, next_obs

        else:
            # Use linearized discrete model to predict next state and compute reward
            next_obs = self.predict_next_state_inverted_pendulum(env, obs, action)

            if self.reward_type == 'discrete':
                reward = 1.0 if (-0.2 <= next_obs[1] <= 0.2) else 0.0
            elif self.reward_type == 'continuous':
                reward = -np.abs(next_obs[1])
            
        return reward, next_obs

    def mujoco_policy(self, env, obs):
        """
        Model Predictive Control policy that can use either:
        - 'random': Random Shooting (uniform random sampling)
        - 'predictive': Predictive Sampling (persistent nominal trajectory with noise)
        - 'rl': RL-guided Predictive Sampling (use TD3 model as nominal trajectory)
        """
        # Initialize persistent nominal trajectory if using Predictive Sampling or RL
        if not hasattr(self, 'nominal_trajectory') and self.control_model in ['predictive', 'rl']:
            self.nominal_trajectory = np.zeros([self.n_planning, *env.action_space.shape])
        
        # Save environment state (Fix #8: use regular copy instead of deepcopy)
        if self.model_type == 'mujoco':
            saved_qpos = env.unwrapped.data.qpos.copy()
            saved_qvel = env.unwrapped.data.qvel.copy()
        else:
            initial_obs = obs.copy()

        # Initialize data structures
        cumulative_reward = np.zeros(self.planning_width)
        cummulative_action = np.zeros(self.planning_width)
        actions = np.zeros([self.planning_width, self.n_planning, *env.action_space.shape])

        # Generate candidate trajectories based on sampling method
        if self.control_model == 'rl':
            # --- RL-guided Predictive Sampling ---
            # Generate nominal trajectory using RL model
            if self.model_type == 'mujoco':
                self.restore_state(env, saved_qpos, saved_qvel)
                current_obs = obs.copy()
            else:
                current_obs = initial_obs.copy()
            
            nominal_actions = []
            for step in range(self.n_planning):
                # Get action from RL model (deterministic for nominal trajectory)
                action, _ = self.model.predict(current_obs, deterministic=True)
                nominal_actions.append(action.copy())
                
                # Move to next state for next prediction
                if step < self.n_planning - 1:
                    _, next_obs = self.take_step(env, current_obs, action)
                    current_obs = next_obs
            
            self.nominal_trajectory = np.array(nominal_actions)
            
            # Fix #1: Restore state after generating nominal trajectory
            if self.model_type == 'mujoco':
                self.restore_state(env, saved_qpos, saved_qvel)
            
            # First candidate is the pure RL policy (max Q-value trajectory)
            actions[0] = self.nominal_trajectory
            
            # If planning_width > 1, generate noisy variations
            if self.planning_width > 1:
                noise_std = 0.1 * (env.action_space.high - env.action_space.low)
                for i in range(1, self.planning_width):
                    noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                    actions[i] = self.nominal_trajectory + noise
                    actions[i] = np.clip(actions[i], env.action_space.low, env.action_space.high)

        elif self.control_model == 'predictive':
            # --- Standard Predictive Sampling ---
            # Shift previous best trajectory forward
            if self.n_planning > 1:
                self.nominal_trajectory = np.vstack([
                    self.nominal_trajectory[1:],
                    np.zeros((1, *env.action_space.shape))
                ])
            
            # First candidate is the nominal trajectory
            actions[0] = self.nominal_trajectory
            
            # Generate noisy variations for other candidates
            if self.planning_width > 1:
                noise_std = 0.1 * (env.action_space.high - env.action_space.low)
                for i in range(1, self.planning_width):
                    noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                    actions[i] = self.nominal_trajectory + noise
                    actions[i] = np.clip(actions[i], env.action_space.low, env.action_space.high)
        else:
            # --- Random Shooting ---
            low = env.action_space.low
            high = env.action_space.high
            actions = np.random.uniform(
                low=low, high=high,
                size=(self.planning_width, self.n_planning) + np.shape(low)
            )

        # Evaluate all trajectories
        for i_trajectory in range(self.planning_width):
            if self.model_type == 'mujoco':
                self.restore_state(env, saved_qpos, saved_qvel)
                current_obs = obs.copy()
            else:
                current_obs = initial_obs.copy()

            for step in range(self.n_planning):
                action = actions[i_trajectory, step]
                reward, next_obs = self.take_step(env, current_obs, action)

                # Subtract the squared action cost
                reward -= self.action_cost * np.sum(np.square(action))

                cumulative_reward[i_trajectory] += reward
                cummulative_action[i_trajectory] += np.abs(action)
                current_obs = next_obs

        # Restore environment state
        if self.model_type == 'mujoco':
            self.restore_state(env, saved_qpos, saved_qvel)

        # Select best trajectory
        best_trajectory = np.argmax(cumulative_reward)
        best_action_trajectory = actions[best_trajectory]
        
        # Update nominal trajectory if using any predictive method
        if self.control_model in ['predictive', 'rl']:
            self.nominal_trajectory = best_action_trajectory
        
        # Return the best policy from this state forward
        return best_action_trajectory


    def get_forward_policy(self, state):
        """
        Get the policy looking forward from a state.
        Currently implemented as epsilon-greedy-n-step-lookahead
        """

        # Generate a set of trajectories to compare
        trajectories = []
        for i in range(self.planning_width):
            trajectories.append(self.epsilon_greedy_n_step_lookahead(state))
        
        # Select the most rewarding trajectory
        cumulative_rewards = [np.sum(t[2,:]) for t in trajectories]
        best_trajectory = trajectories[np.argmax(cumulative_rewards)]

        # Return the best policy from this state forward
        return best_trajectory[0, :]

    def update(self, state, action, reward, next_state, next_action):
        """ Compatibility method for now. """
        pass

    def epsilon_greedy_n_step_lookahead(self, state):
        
        # Initialize memory for a trajectory
        trajectory = np.zeros([3, self.n_planning], dtype=int)
        
        # Loop over planning steps filling out trajectory
        for step in range(self.n_planning):
            
            # Initialize a memory for planning
            planning_cache = np.zeros([2, self.n_actions])

            for action in range(self.n_actions):
                # Determine the outcome of this action
                next_state, reward = self.transition_model(state, action)
                
                # Save what we see
                planning_cache[0, action] = next_state
                planning_cache[1, action] = reward

            # Get the epsilon-greedy action
            if np.random.rand() < self.epsilon:
                best_action = np.random.randint(self.n_actions)
            else:
                best_action = np.argmax(planning_cache[1, :])
    
            # Save the best action, state, and reward
            trajectory[0, step] = best_action
            trajectory[1, step] = planning_cache[0, best_action]    # Best state
            trajectory[2, step] = planning_cache[1, best_action]    # Best reward

        return trajectory