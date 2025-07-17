import numpy as np
import mujoco
import copy
from scipy.linalg import expm, svd
import torch
from stable_baselines3 import TD3

class MPCShittyBird:
    """ Doesn't aspire to much. """
    def __init__(self, n_actions, recompute = 10, planning_width = 5, n_planning = 10, reward_type='discrete', action_cost=0.0, control_model='random', value='env', rl_model='50k'):
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
        self.action_cost = action_cost
        self.control_model = control_model  # 'random' or 'predictive' 

        self.rl_model = rl_model  # Name of the RL model to use, if applicable
        self.value = value  # 'env' or 'rl' to use MuJoCo env or RL model
        
        # If using RL, we need to initialize the model
        model_files = {
                '10k': "../rl_models/td3_invertedpendulum_continuous_10k_steps",
                '50k': "../rl_models/td3_invertedpendulum_continuous_50k_steps", 
                '100k': "../rl_models/td3_invertedpendulum_continuous_100k_steps",
                '300k': "../rl_models/td3_invertedpendulum_continuous_300k_steps", 
                '500k': "../rl_models/td3_invertedpendulum_continuous_500k_steps",
                '1000k': "../rl_models/td3_invertedpendulum_continuous_1000k_steps",
                '2000k': "../rl_models/td3_invertedpendulum_continuous_2000k_steps",
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
  
    def restore_state(self, env, qpos, qvel):
        # Restore the environment's state
        env.unwrapped.data.qpos[:] = qpos
        env.unwrapped.data.qvel[:] = qvel

        # We need to call mujoco.mj_forward to update the simulation state with the restored positions and velocities.
        # This function ensures that all the states are correctly computed after restoring the qpos and qvel.
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    def calculate_trajectory_rewards(self, env, actions, observations, discrete_rewards, done_flags):
        """
        Compute reward based on MuJoCo env step.

        Parameters:
        - env: the gymnasium environment object.
        - actions: actions taken during the trajectory [planning_width, n_planning, action_dim]
        - observations: states observed during the trajectory [planning_width, n_planning, obs_dim]
        - discrete_rewards: rewards received at each step [planning_width, n_planning]
        - done_flags: flags indicating if the episode ended at each step [planning_width, n_planning]

        Returns:
        - cumulative_reward: array of cumulative rewards for each trajectory [planning_width]
        """
        
        if self.reward_type == 'discrete':
            return discrete_rewards.sum(axis=1)
        
        elif self.reward_type == 'continuous':
            if self.value == 'env':
                # Fully vectorized env rewards!
                # All operations work element-wise on the [planning_width, n_planning] arrays
                
                # Base reward for staying alive
                reward = np.ones_like(observations[:, :, 0])  # [planning_width, n_planning]
                
                # Angle reward (most important - stay upright)
                angle_reward = np.exp(-5 * observations[:, :, 1]**2)
                reward += 2.0 * angle_reward
                
                # Position reward (stay centered)
                position_reward = np.exp(-0.5 * observations[:, :, 0]**2)
                reward += 0.5 * position_reward
                
                # Stability reward (minimize velocities)
                velocity_penalty = 0.1 * (observations[:, :, 2]**2 + observations[:, :, 3]**2)
                reward -= velocity_penalty
                
                # Large penalty for falling
                reward -= 10.0 * done_flags  # done_flags is boolean, converts to 0/1
                
                # Bonus for being very stable
                stable_angle = np.abs(observations[:, :, 1]) < 0.1
                stable_velocity = np.abs(observations[:, :, 3]) < 0.1
                stability_bonus = stable_angle & stable_velocity
                reward += 0.5 * stability_bonus
                
                # Sum over planning steps
                cumulative_reward = reward.sum(axis=1)
                
                return cumulative_reward

            elif self.value == 'rl':
                # For RL rewards, we can also vectorize completely!
                # Reshape observations to [batch_size, obs_dim] where batch_size = planning_width * n_planning
                batch_size = self.planning_width * self.n_planning
                obs_batch = observations.reshape(batch_size, -1)
                
                # Get actions from the learned policy for all observations at once
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32)
                    actions_batch, _ = self.model.predict(obs_batch, deterministic=True)
                    actions_tensor = torch.as_tensor(actions_batch, dtype=torch.float32)
                    
                    # Get Q(s, π(s)) for all state-action pairs at once
                    critic_output = self.model.critic(obs_tensor, actions_tensor)
                    rewards_batch = critic_output[0].detach().numpy().flatten()
                
                # Reshape back to [planning_width, n_planning] and sum over planning steps
                rewards_matrix = rewards_batch.reshape(self.planning_width, self.n_planning)
                cumulative_reward = rewards_matrix.sum(axis=1)
                
                return cumulative_reward

    def mujoco_policy(self, env, obs):
        """
        Model Predictive Control policy that can use either:
        - 'random': Random Shooting (uniform random sampling)
        - 'predictive': Predictive Sampling (persistent nominal trajectory with noise)
        """
        # Initialize persistent nominal trajectory using the optimal actions from the RL model
        if not hasattr(self, 'nominal_trajectory'):
            self.nominal_trajectory = np.array([self.model.predict(obs, deterministic=True)[0] for _ in range(self.n_planning)])
        
        # Save environment state 
        saved_qpos = env.unwrapped.data.qpos.copy()
        saved_qvel = env.unwrapped.data.qvel.copy()

        # Initialize data structures
        actions = np.zeros([self.planning_width, self.n_planning, *env.action_space.shape])
        observations = np.zeros([self.planning_width, self.n_planning, *env.observation_space.shape])
        discrete_rewards = np.zeros([self.planning_width, self.n_planning])
        done_flags = np.zeros([self.planning_width, self.n_planning], dtype=bool)
        cumulative_reward = np.zeros([self.planning_width])

        # Generate candidate trajectories based on the control method

        # # ----------------------------------------------------------------------------------------------
        # # Predictive sampling starting from the RL deterministic policy
        # # ----------------------------------------------------------------------------------------------
        # if self.control_model == 'rl':
        #     # If using RL, we need to get the action from the learned policy
        #     actions[0] = np.array([self.model.predict(obs, deterministic=True)[0] for _ in range(self.n_planning)])
        #     self.nominal_trajectory = actions[0]
        #     # Generate noisy variations for other candidates
        #     if self.planning_width > 1:
        #         noise_std = 0.1 * (env.action_space.high - env.action_space.low)
        #         for i in range(1, self.planning_width):
        #             noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
        #             actions[i] = self.nominal_trajectory + noise
        #             actions[i] = np.clip(actions[i], env.action_space.low, env.action_space.high)

        # ----------------------------------------------------------------------------------------------
        # Predictive sampling starting from zeros (Algorithm 4 in: Howell, T., Gileadi, N., Tunyasuvunakool, S., Zakka, K., Erez, T., & Tassa, Y. (2022). Predictive sampling: Real-time behaviour synthesis with mujoco. arXiv preprint arXiv:2212.00541.)
        # ----------------------------------------------------------------------------------------------
        if self.control_model in ['predictive', 'rl']:
        
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

        # ----------------------------------------------------------------------------------------------
        # Random Shooting (uniform random sampling)
        # ----------------------------------------------------------------------------------------------
        elif self.control_model == 'random':
            low = env.action_space.low
            high = env.action_space.high
            actions = np.random.uniform(
                low=low, high=high,
                size=(self.planning_width, self.n_planning) + np.shape(low)
            )

        # ----------------------------------------------------------------------------------------------
        # Evaluate all trajectories
        # ----------------------------------------------------------------------------------------------
        for i_trajectory in range(self.planning_width):
            self.restore_state(env, saved_qpos, saved_qvel)
            current_obs = obs.copy()

            for step in range(self.n_planning):

                if self.planning_width <2:
                    # If using only RL and no planning, we need to get the action from the learned policy
                    action = self.model.predict(current_obs, deterministic=True)[0]
                    next_obs, reward, done, truncated, info = env.step(action)
                elif self.planning_width >1:
                    # Otherwise, use the sampled action
                    action = actions[i_trajectory, step]
                    next_obs, reward, done, truncated, info = env.step(action) 
                    
                # Save the action in the trajectory
                actions[i_trajectory, step] = action
                observations[i_trajectory, step] = next_obs
                discrete_rewards[i_trajectory, step] = reward
                done_flags[i_trajectory, step] = done

        # Calculate cumulative rewards for each trajectory
        cumulative_reward = self.calculate_trajectory_rewards(env, actions, observations, discrete_rewards, done_flags)

        # Restore environment state
        self.restore_state(env, saved_qpos, saved_qvel)

        # Select best trajectory
        best_trajectory = np.argmax(cumulative_reward)
        best_action_trajectory = actions[best_trajectory]
        
        # Update nominal trajectory if using any predictive method
        if self.control_model in ['predictive', 'rl']:
            self.nominal_trajectory = best_action_trajectory
        
        # Return the best policy from this state forward
        return best_action_trajectory

