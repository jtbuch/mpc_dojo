from matplotlib import pyplot as plt
import numpy as np

from utils import *

def plot_policy():
    pass

def plot_action_values(bird, world):
    """
    Unrolls the action values of the bird and plots them.
    The action values are unrolled into a 2D grid for each z slice.
    The action values are then plotted as a grid of arrows.
    """
    # Plot the birds state-action values
    values = unroll_action_values(bird, world)
    argmax_policy = np.argmax(values, axis=3)
    for z in range(world.z_size):
        plot_symbol_grid(argmax_policy[:,:,z])
        plt.title("Bird Policy at z={}".format(z))

def plot_symbol_grid(arr):
    """
    Plots symbols in the center of a grid of cells.
    Values 0-3 map to up/down/right/left arrows (blue).
    Value 4 maps to circle (green).
    Value 5 maps to cross (red).
    """

    # Markers and colors. Actions are north, south, east, west, up, down
    symbol_map = {0: '^', 1: 'v', 2: '>', 3: '<', 4: 'o', 5: 's'}
    color_map  = {0: 'blue', 1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'}

    # Create a new figure
    plt.figure()

    # Draw black grid lines (horizontal and vertical)
    num_cols, num_rows = arr.shape
    for i in range(num_rows + 1):
        plt.hlines(y=i, xmin=0, xmax=num_cols, color='black')
    
    for j in range(num_cols + 1):
        plt.vlines(x=j, ymin=0, ymax=num_rows, color='black')

    # Plot each cell with the appropriate marker and color at its center
    for i in range(num_rows):
        for j in range(num_cols):
            val    = arr[j, i]
            marker = symbol_map[val]
            color  = color_map[val]
            plt.scatter(j + 0.5, i + 0.5, marker=marker, c=color, s=200)

    # Turn off Matplotlib's built-in grid
    plt.grid(False)

    # Limit axes to the size of the grid
    plt.xlim(0, num_cols)
    plt.ylim(0, num_rows)
    plt.xlabel("Green-circle = up, Red-square = down")

    # Remove default tick marks to emphasize the black grid lines
    plt.xticks([])
    plt.yticks([])

    # Make cells square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_flight_trajectory(hist, world, episode=0):
    """
    Plots the flight trajectory of the bird in 3D space.
    The trajectory is plotted as a line in 3D space..
    """
    # Get number of time steps, coordinates over time
    n_steps = hist.shape[1]
    coords, _ = convert_hist_to_coords(hist, world, episode)

    # Plot the trajectory
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    jitter_scale = 0.1
    jitter = np.random.uniform(-jitter_scale, jitter_scale, [n_steps, 3])
    points = coords + jitter

    # Get color from blue to red representing time
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(0, n_steps)
    colors = cmap(norm(np.arange(n_steps)))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o')
    # Plot the trajectory line segments in average color
    for t in range(n_steps - 1):
        color = np.mean(colors[t:t+2], axis=0)
        ax.plot(points[t:t+2, 0], points[t:t+2, 1], points[t:t+2, 2], color=color)    

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Flight trajectory of the bird')
    plt.show()


def plot_state_occupancy_heatmap(hist, world, episode=-1):
    """
    Plots the state occupancy heatmap of the bird.
    The heatmap is plotted as a stack of 2D heatmaps.
    """
    n_steps = hist.shape[1]
    coords, _ = convert_hist_to_coords(hist, world, episode)
    occupancy = np.zeros((world.x_size, world.y_size, world.z_size))
    for t in range(n_steps):
        x, y, z = coords[t, :]
        occupancy[x, y, z] += 1

    # Normalize occupancy to [0, 1]
    occupancy /= np.max(occupancy)

    for z in range(world.z_size):
        plt.figure()
        
        # Plot the heatmap
        plt.imshow(occupancy[:, :, z].T, cmap='coolwarm', interpolation='none')
        plt.clim(0, 1)

        # Plot grid of cell outlines
        num_cols, num_rows = occupancy[:,:,z].shape
        for i in range(num_rows + 1):
            plt.hlines(y=i-0.5, xmin=-0.5, xmax=num_cols-0.5, color='black')
        
        for j in range(num_cols + 1):
            plt.vlines(x=j-0.5, ymin=-0.5, ymax=num_rows-0.5, color='black')

        # Print occupancy fractions in cells
        for i in range(num_rows):
            for j in range(num_cols):
                val = occupancy[j, i, z]
                plt.text(j, i, "{:.2f}".format(val), ha='center', va='center', fontsize=10, color='white')

        plt.xticks([])
        plt.yticks([])

        plt.title("Bird Occupancy at z={}".format(z))
