from matplotlib import pyplot as plt
import numpy as np

from utils import unroll_action_values

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