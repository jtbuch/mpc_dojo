from configs import *

def simulate(bird, world, n_episodes, time_steps, train=False):
    """ Simulate bird, possible episodic SARSA. """

    # Initialize trajectory tracking [episode, time_step, SAR]
    hist = np.zeros((n_episodes, time_steps, 3), dtype=int)

    # Episodic loop
    for epi in range(n_episodes):

        # Random initial state, first bird action
        x, y, z = world.random_state_coords()
        state  = world.state_to_index(x, y, z)
        action = bird.policy(state)
        bird.register_action()

        # Time-step loop
        for t in range(0, time_steps):
            # Transition state, get bird action and reward
            next_state  = world.transition(state, action)
            reward      = world.reward(next_state)
            next_action = bird.policy(next_state)
            bird.register_action()

            # Learning
            if train: bird.update(state, action, reward, next_state, next_action)

            # Save trajectory information
            hist[epi, t, :] = (state, action, reward)

            # Cycle current state and action
            state, action  = next_state, next_action

    return hist

if __name__ == "__main__":
    # Close any old plots
    plt.close('all')

    # Could take args, this is fine though
    use_RL_bird = False
    use_MPC_bird = False
    compare = True

    # Create environment
    world = World(x_size = 5, y_size = 3, z_size = 5, wind = [0.0, 0.0, 0.0])

    # RL simulation
    if use_RL_bird:
        # Create the bird and a world model for the bird
        rl_bird = RLShittyBird(n_states = world.n_states, n_actions = world.n_actions, n_planning = 1, update_type='SARSA')
        bad_model = World(x_size = 5, y_size = 3, z_size = 5, wind = [0.0, 0.0, 0.0])
        rl_bird.init_transition_model(bad_model)

        # Save the initial q_values
        q_values_prior = rl_bird.action_values.flatten()

        # Train the bird
        rl_bird_hist = simulate(rl_bird, world, n_episodes=1000, time_steps=1000, train=True)

        # Save the final q_values
        q_values_post = rl_bird.action_values.flatten()

        # Plot action values
        plot_action_values(rl_bird, world)

        # Plot bird's flight path on the final episode
        plot_flight_trajectory(rl_bird_hist, world)

        # Plot it's final episode state occupancy
        plot_state_occupancy_heatmap(rl_bird_hist, world)

    # MPC simulation
    if use_MPC_bird:
        # Create bird
        mpc_bird = MPCShittyBird(n_actions = world.n_actions, recompute = 1)
        mpc_bird.init_transition_model(world)

        # Get performance
        mpc_bird_hist = simulate(mpc_bird, world, n_episodes=5, time_steps=1000, train=True)

        # Plot bird's flight path on the final episode
        plot_flight_trajectory(mpc_bird_hist, world)

        # Plot it's final episode state occupancy
        plot_state_occupancy_heatmap(mpc_bird_hist, world)

        # Plot the average reward per time step
        reward_per_time = np.cumsum(mpc_bird_hist[:,:,2], axis=1)/np.arange(1, mpc_bird_hist.shape[1]+1)


    if compare:
        tsteps = 600
        reward_per_time = np.zeros((3, tsteps))

        for i in range(0,3):
            # Set recompute times
            recompute = i*2+1
            
            # Setup the world
            world = World(x_size = 5, y_size = 3, z_size = 5, wind = [0.0, 0.0, 0.0])

            # Create bird
            mpc_bird = MPCShittyBird(n_actions = world.n_actions, recompute = recompute)
            mpc_bird.init_transition_model(world)

            # Get performance
            mpc_bird_hist = simulate(mpc_bird, world, n_episodes=20, time_steps=tsteps, train=True)

            # Plot the average reward per time step
            reward_per_time[i,:] = np.mean(np.cumsum(mpc_bird_hist[:,:,2], axis=1)/np.arange(1, tsteps+1), axis=0)



        plt.figure()
        plt.plot(reward_per_time.T, label=['RT1', 'RT3', 'RT5'])
        plt.title('Average Reward Rate\nNoisy World, Correct Model')
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.grid()
        plt.legend()
        plt.tight_layout()


        plt.figure()
        plt.plot(reward_per_time.T, label=['RT1', 'RT3', 'RT5'])
        plt.title('Average Reward Rate\nNoiseless World, Correct Model')
        plt.xlabel('Time step')
        plt.ylabel('Reward Rate')
        plt.grid()
        plt.legend()
        plt.tight_layout()

