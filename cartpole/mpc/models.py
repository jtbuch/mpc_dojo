import numpy as np
import mujoco
import copy
from scipy.linalg import expm, svd

class BirdWorldModel:
    """ What the bird thinks."""
    def __init__(self, n_states, n_actions):
        self.autonomous = np.randn(n_states, n_states)
        self.controlled = np.randn(n_states, n_actions)

    def transition(self, state, action):
        self.state = self.autonomous @ self.state + self.controlled @ self.action

class BirdWorld:
    """ The way life actually is. The bird falls down unless it flaps."""
    def __init__(self, x_size, y_size, z_size, wind=[0.0, 0.0, 0.0]):
        # Gridworld dimensions
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.wind   = wind

        self.n_states = x_size * y_size * z_size

        # Action definitions (6 possible actions)
        self.action_shifts = np.array([
            [ 0,  1, -1],  # north
            [ 0, -1, -1],  # south
            [ 1,  0, -1],  # east
            [-1,  0, -1],  # west
            [ 0,  0,  1],  # up
            [ 0,  0, -1],  # down
        ])
        self.n_actions = len(self.action_shifts)

    def random_state_coords(self):
        """ Randomly select a state in the environment. """
        x = np.random.randint(self.x_size)
        y = np.random.randint(self.y_size)
        z = np.random.randint(self.z_size)
        return x, y, z

    def in_bounds(self, x, y, z):
        """Check if (x,y,z) is within environment limits."""
        return (0 <= x < self.x_size) and \
               (0 <= y < self.y_size) and \
               (0 <= z < self.z_size)

    def state_to_index(self, x, y, z):
        """ Convert (x,y,z) in [0, x_size-1] x [0, y_size-1] x [0, z_size-1] to a single integer index. """
        return int(x + y*self.x_size + z*self.x_size*self.y_size)

    def index_to_state(self, index):
        """ Inverse of the above: flatten -> (x,y,z). """
        z = index // (self.x_size * self.y_size)
        remainder = index % (self.x_size * self.y_size)
        y = remainder // self.x_size
        x = remainder % self.x_size
        return x, y, z

    def transition(self, state, action, wind_on=True):
        """ Given a discrete state index and action index, return next_state_index. """
        # Get coordinates from state
        x, y, z = self.index_to_state(state)

        # Based on action, change coordinates
        dx, dy, dz = self.action_shifts[action]
        new_x = x + dx
        new_y = y + dy
        new_z = z + dz

        # Roll some dice to see if the wind pushes the bird
        if wind_on:
            wind = []
            for i in range(3):
                get_blown = np.random.rand() < np.abs(self.wind[i])
                wind.append(get_blown * np.sign(self.wind[i]))

            new_x += wind[0]
            new_y += wind[1]
            new_z += wind[2]

        # If next state is out of bounds, ignore action but reduce z
        if not self.in_bounds(new_x, new_y, new_z):
            new_x, new_y = x, y
            new_z = max(0, z - 1)
            
        # Return state
        return self.state_to_index(new_x, new_y, new_z)

    def reward(self, state):
        """ Don't be on the ground, ceiling, or walls, but higher is better"""
        x, y, z = self.index_to_state(state)
        
        # Reward based on height
        reward = z

        # Check if the bird is against bounds of box
        ground  = z == 0
        ceiling = z == self.z_size - 1
        x_wall  = x == 0 or (x == self.x_size - 1) 
        y_wall  = y == 0 or (y == self.y_size - 1)

        # Make ground and ceiling bad, walls
        if ground:  reward += -5
        if ceiling: reward += -5
        if x_wall:  reward += -5
        if y_wall:  reward += -5

        return reward

class MPCShittyBird:
    """ Doesn't aspire to much. """
    def __init__(self, n_actions, recompute = 10, planning_width = 5, n_planning = 10, reward_type='discrete', model_type='mujoco', action_cost=0.0, sampling_method='random'):
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
        self.sampling_method = sampling_method  # 'random' or 'predictive'

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
                reward = -np.abs(next_obs[1])  # negative absolute angle
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
        """
        # Initialize persistent nominal trajectory if using Predictive Sampling
        if not hasattr(self, 'nominal_trajectory') and hasattr(self, 'sampling_method') and self.sampling_method == 'predictive':
            self.nominal_trajectory = np.zeros([self.n_planning, *env.action_space.shape])
        
        # Save environment state
        if self.model_type == 'mujoco':
            saved_qpos = copy.deepcopy(env.unwrapped.data.qpos)
            saved_qvel = copy.deepcopy(env.unwrapped.data.qvel)
        else:
            initial_obs = obs.copy()

        # Initialize data structures
        cumulative_reward = np.zeros(self.planning_width)
        cummulative_action = np.zeros(self.planning_width)  # Kept for compatibility
        actions = np.zeros([self.planning_width, self.n_planning, *env.action_space.shape])

        # Generate candidate trajectories based on sampling method
        if hasattr(self, 'sampling_method') and self.sampling_method == 'predictive':
            # --- Predictive Sampling ---
            # Shift previous best trajectory forward
            if self.n_planning > 1:
                self.nominal_trajectory = np.vstack([
                    self.nominal_trajectory[1:],
                    np.zeros((1, *env.action_space.shape))
                ])
            
            # First candidate is the nominal trajectory
            actions[0] = self.nominal_trajectory
            
            # Generate noisy variations for other candidates
            noise_std = 0.1 * (env.action_space.high - env.action_space.low)
            for i in range(1, self.planning_width):
                noise = np.random.normal(0, noise_std, self.nominal_trajectory.shape)
                actions[i] = self.nominal_trajectory + noise
                actions[i] = np.clip(actions[i], env.action_space.low, env.action_space.high)
        else:
            # --- Random Shooting ---
            # Generate all random trajectories at once using vectorization
            low = env.action_space.low
            high = env.action_space.high
            actions = np.random.uniform(
                low=low,
                high=high,
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
                cummulative_action[i_trajectory] += np.abs(action)  # Kept for compatibility
                current_obs = next_obs

        # Restore environment state
        if self.model_type == 'mujoco':
            self.restore_state(env, saved_qpos, saved_qvel)

        # Select best trajectory
        best_trajectory = np.argmax(cumulative_reward)
        best_action_trajectory = actions[best_trajectory]
        
        # Update nominal trajectory if using Predictive Sampling
        if hasattr(self, 'sampling_method') and self.sampling_method == 'predictive':
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


class RLShittyBird:
    """ Aspires to be better. """
    def __init__(self, n_states, n_actions, n_planning=1000, update_type='SARSA'):
        # Bird parameters
        self.gamma          = 0.9            # Discount factor
        self.alpha          = 0.05           # Learning rate
        self.epsilon        = 0.1            # Epsilon for epsilon-greedy policy
        self.smax_beta      = 0.5            # Softmax beta parameter
        self.n_states       = n_states
        self.n_actions      = n_actions
        self.action_values  = np.random.randn(n_states, n_actions)
        self.n_planning     = n_planning

        self.update_type    = update_type
        self.policy_type    = 'epsilon_greedy'

    # ---------------- Policy methods ------------------
    def policy(self, state):
        """ Wrapper for different policies. """
        if self.policy_type == 'epsilon_greedy':
            return self.epsilon_greedy_policy(state)
        elif self.policy_type == 'softmax':
            return self.softmax_policy(state)

    def epsilon_greedy_policy(self, state):
        """ Mostly greedy action selection, sometimes random. """
        # Roll the epsilon-dice
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) # Random
        else:
            return np.argmax(self.action_values[state,:]) # Greedy

    def softmax_policy(self, state):
        """ Softmax action selection. """
        # Compute action probabilities
        q_values = self.action_values[state,:]
        exp_q = np.exp(self.smax_beta * q_values)
        probs = exp_q / np.sum(exp_q)

        # Sample an action based on the probabilities
        return np.random.choice(self.n_actions, p=probs)
    
    # ---------------- Update methods ------------------
    def update(self, state, action, reward, next_state=None, next_action=None):
        """ Wrapper for whatever update method is used. """
        if self.update_type == "SARSA":
            self.SARSA_update(state, action, reward, next_state, next_action)
        elif self.update_type == "Dyna":
            self.dyna_update(state, action, reward)

    def SARSA_update(self, state, action, reward, next_state, next_action):
        """ Canonical update of Q-values using SARSA. """
        # Q values for current and next state
        Q_t0 = self.action_values[state, action]
        Q_t1 = self.action_values[next_state, next_action]

        # Update Q-values based on reward
        self.action_values[state, action] = (1-self.alpha)*Q_t0 + self.alpha*(reward + self.gamma*Q_t1)

    # ---------------- Dyna components ------------------
    def init_transition_model(self, world):
        """ Dyna Q model. """
        self.world_transition = world.transition
        self.world_reward     = world.reward

    def transition_model(self, state, action):
        """ Bird's model of the world's transition structure. """
        next_state = self.world_transition(state, action)
        reward     = self.world_reward(next_state)
        return next_state, reward

    def dyna_update(self, state, action, reward):
        """ Updates the transition model then the Q-values. """
        # Update the state-transition model
        #self.update_transition_model(state, action, next_state, reward)

        # Update Q-values based on rollout from the transition model
        self.dyna_planning_update(state, action, reward)

    def update_transition_model(self, state, action, next_state, reward):
        """ Standard Dyna update of transition model. """
        #self.transition_model(state, action) = (next_state, reward)
        pass

    def dyna_planning_update(self, state, action, reward):
        """ Update Q-values based on model rollouts. """
        # Rollout loop
        for _ in range(self.n_planning):
            # Sample a next state and reward from the model
            next_state, next_reward = self.transition_model(state, action)

            # Get the next action for doing a SARSA update
            next_action = self.policy(next_state)

            # Update Q-values based on the sampled transition
            self.SARSA_update(state, action, reward, next_state, next_action)

            # Cycle
            state, reward, action = next_state, next_reward, next_action

    # ---------------- Misc ------------------
    def register_action(self):
        """ Register that an action has been taken (compatibility method) """
        pass

