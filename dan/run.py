from configs import *

def simulate(bird, world, n_episodes, time_steps, train=False):
    """ Simulate bird, possible episodic SARSA. """
    # Episodic loop
    for _ in range(n_episodes):

        # Random initial state, first bird action
        x, y, z = world.random_state_coords()
        state  = world.state_to_index(x, y, z)
        action = bird.policy(state)

        # Time-step loop
        for t in range(time_steps):
            # Transition state, get bird action and reward
            next_state  = world.transition(state, action)
            reward      = world.reward(next_state)
            next_action = bird.policy(next_state)

            # Learning
            if train: bird.update(state, action, reward, next_state, next_action)

            # Update current state and action
            state, action  = next_state, next_action

if __name__ == "__main__":
    # Close any old plots
    plt.close('all')

    # Create environment
    world = World(x_size = 5, y_size = 3, z_size = 5, wind = [0.2, 0.0, 0.0])

    # Create the bird and a world model for the bird
    bird = ShittyBird(n_states = world.n_states, n_actions = world.n_actions, n_planning = 1000, update_type='Dyna')
    bad_model = World(x_size = 5, y_size = 3, z_size = 5, wind = [0.0, 0.0, 0.0])
    bird.init_transition_model(bad_model)

    # Save the initial q_values
    q_values_prior = bird.action_values.flatten()

    # Train the bird
    simulate(bird, world, n_episodes=1000, time_steps=1, train=True)

    # Save the final q_values
    q_values_post = bird.action_values.flatten()

    # Plot action values
    plot_action_values(bird, world)