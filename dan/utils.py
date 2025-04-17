import numpy as np

def unroll_action_values(bird, world):
    """ Unroll action-value array over all states. """
    values = np.zeros((world.x_size, world.y_size, world.z_size, world.n_actions))
    for x in range(world.x_size):
        for y in range(world.y_size):
            for z in range(world.z_size):
                index = world.state_to_index(x, y, z)
                for a in range(world.n_actions):
                    values[x, y, z, a] = bird.action_values[index, a] 
    return values

def convert_hist_to_coords(hist, world, episode):
    """ Takes a history of states and rewards and unpacks them. """
    # Get the coordinates over time
    _, n_steps, _ = hist.shape

    # Loop over time-steps converting states to coordinates
    coords  = np.zeros((n_steps, 3), dtype=int)
    rewards = np.zeros(n_steps)
    for t in range(n_steps):
        # Extract state coordinates
        state = hist[episode, t, 0]
        x, y, z = world.index_to_state(state)
        coords[t, :] = (x, y, z)

        # Extract rewards
        rewards[t] = hist[episode, t, 2]

    return coords, rewards
