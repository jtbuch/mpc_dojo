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