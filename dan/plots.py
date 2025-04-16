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
    # Plot the bird's state-action values
    values = unroll_action_values(bird, world)
    argmax_policy = np.argmax(values, axis=3)

    # Create vertical subplots (one column)
    n_rows = int(np.ceil(world.z_size))
    fig, axs = plt.subplots(n_rows, 1, figsize=(2, 2 * n_rows))
    axs = np.atleast_1d(axs)  # ensures axs is always iterable

    # Loop in reversed order so Z=0 is at the bottom
    for z, ax in zip(reversed(range(world.z_size)), axs):
        plot_symbol_grid(argmax_policy[:, :, z], ax)
        ax.set_title(f"Z = {z}")

    # Add a single label across bottom
    fig.text(0.5, 0.04, "Green-circle = up, Red-square = down", ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def plot_symbol_grid(arr, ax):
    """
    Plots symbols in the center of a grid of cells on the provided Axes object.
    Values 0-3 map to up/down/right/left arrows (blue).
    Value 4 maps to circle (green).
    Value 5 maps to cross (red).
    """
    symbol_map = {0: '^', 1: 'v', 2: '>', 3: '<', 4: 'o', 5: 's'}
    color_map  = {0: 'blue', 1: 'blue', 2: 'blue', 3: 'blue', 4: 'green', 5: 'red'}

    num_cols, num_rows = arr.shape

    # Draw grid lines
    for i in range(num_rows + 1):
        ax.hlines(y=i, xmin=0, xmax=num_cols, color='black')
    for j in range(num_cols + 1):
        ax.vlines(x=j, ymin=0, ymax=num_rows, color='black')

    # Plot markers
    for i in range(num_rows):
        for j in range(num_cols):
            val = arr[j, i]
            marker = symbol_map[val]
            color = color_map[val]
            ax.scatter(j + 0.5, i + 0.5, marker=marker, c=color, s=200)

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
