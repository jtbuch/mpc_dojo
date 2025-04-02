from configs import *

class WorldModel:
    """ What the bird thinks."""
    def __init__(self, n_states, n_actions):
        self.autonomous = np.randn(n_states, n_states)
        self.controlled = np.randn(n_states, n_actions)

    def transition(self, state, action):
        self.state = self.autonomous @ self.state + self.controlled @ self.action

class World:
    """ The way life actually is. The bird falls down unless it flaps."""
    def __init__(self, x_size, y_size, z_size):
        # Gridworld dimensions
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        self.n_states = x_size * y_size * z_size

        # Action definitions (6 possible actions)
        self.action_shifts = np.array([
            [ 0,  1, -1],  # north
            [ 0, -1, -1],  # south
            [-1,  0, -1],  # east
            [ 1,  0, -1],  # west
            [ 0,  0, -1],  # down
            [ 0,  0,  1],  # up
        ])
        self.n_actions = len(self.action_shifts)

    def in_bounds(self, x, y, z):
        """Check if (x,y,z) is within environment limits."""
        return (0 <= x < self.x_size) and \
               (0 <= y < self.y_size) and \
               (0 <= z < self.z_size)

    def state_to_index(self, x, y, z):
        """ Convert (x,y,z) in [0, x_size-1] x [0, y_size-1] x [0, z_size-1] to a single integer index. """
        return x + y*self.x_size + z*self.x_size*self.y_size

    def index_to_state(self, index):
        """ Inverse of the above: flatten -> (x,y,z). """
        z = index // (self.x_size * self.y_size)
        remainder = index % (self.x_size * self.y_size)
        y = remainder // self.x_size
        x = remainder % self.x_size
        return x, y, z

    def transition(self, state, action):
        """ Given a discrete state index and action index, return next_state_index. """
        # Get coordinates from state
        x, y, z = self.index_to_state(state)

        # Based on action, change coordinates
        dx, dy, dz = self.action_shifts[action]
        new_x = x + dx
        new_y = y + dy
        new_z = z + dz

        # If next state is out of bounds, ignore action but reduce z
        if not self.in_bounds(new_x, new_y, new_z):
            new_x, new_y = x, y
            new_z = max(0, z - 1)
            
        # Return state
        return self.state_to_index(new_x, new_y, new_z)

    def reward(self, state):
        """ Don't be on the ground or ceiling, but higher is better"""
        x, y, z = self.index_to_state(state)
        
        # Don't be on the ground
        if z == 0:
            return -5  # on the ground
        
        # Don't be at the ceiling
        if z == self.z_size - 1:
            return -5

        # Higher is better
        if z > 0 and z < self.z_size - 1:
            return z  # higher is better

class ShittyBird:
    """ Aspires to be better. """
    def __init__(self, n_states, n_actions):
        self.model = None
        self.gamma = 0.9
        self.alpha = 0.1
        self.n_states  = n_states
        self.n_actions = n_actions
        self.action_values = np.random.randn(n_states, n_actions)

        self.state = None
    
    def SARSA_update(self, state, action, reward, next_state, next_action):
        """ Canonical update of Q-values using SARSA. """
        # Q values for current and next state
        Q_t0 = self.action_values[state, action]
        Q_t1 = self.action_values[next_state, next_action]

        # Update Q-values based on reward
        self.action_values[state, action] = (1-self.alpha)*Q_t0 + self.alpha*(reward + self.gamma*Q_t1)

    def policy(self, state):
        """ Epsilon-greedy action selection. """
        # Roll the epsilon-dice
        if np.random.rand() < 0.1:
            # Random
            return np.random.randint(self.n_actions)
        else:
            # Greedy
            return np.argmax(self.action_values[state,:])

    def rollout(self, time_steps):
        """ Produces a trajectory given a policy. """
        if self.model is None:
            raise ValueError("Model not set. Cannot rollout.")
        
        # Rollout the model for a given policy
        state_list = []
        for t in range(time_steps):
            state = self.model.predict(state, self.policy(state))
            state_list.append(state)
            # TODO: add reward list

        return state_list
    
def simulate(bird, world, n_episodes, time_steps, train=False):
    """ Simulate bird, possible episodic SARSA. """
    # Episodic loop
    for _ in range(n_episodes):

        # Random initial state, convert to state index
        x, y, z = np.random.randint(5, size=3)
        state = world.state_to_index(x, y, z)

        # First bird action
        action = bird.policy(state)

        # Time-step loop
        for t in range(time_steps):
            next_state  = world.transition(state, action)
            next_action = bird.policy(next_state)
            reward      = world.reward(next_state)

            # Learning
            if train:
                bird.SARSA_update(state, action, reward, next_state, next_action)
            
            # Update current state and action
            state  = next_state
            action = next_action

if __name__ == "__main__":
    # Create environment
    world = World(x_size=5, y_size=5, z_size=5)

    # Create the bird
    bird = ShittyBird(n_states=world.n_states, n_actions=world.n_actions)

    # Train the bird
    simulate(bird, world, n_episodes=100, time_steps=1000, train=True)

    # Plot the birds state-action values
    values = unroll_action_values(bird, world)
    argmax_policy = np.argmax(values, axis=3)
    for z in range(world.z_size):
        plot_symbol_grid(argmax_policy[:,:,z])
        plt.title("Bird Policy at z={}".format(z))