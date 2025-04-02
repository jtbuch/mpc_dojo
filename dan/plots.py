from matplotlib import pyplot as plt
import numpy as np

def plot_policy():
    pass

def plot_action_values():
    pass
import matplotlib.pyplot as plt
import numpy as np

def plot_symbol_grid(arr):
    """
    Plots a grid of symbols based on the contents of arr.
    Values 0-3 map to up/down/left/right arrows (blue).
    Value 4 maps to circle (green).
    Value 5 maps to cross (red).
    A black-line grid is drawn manually, with symbols in
    the center of each cell.
    """

    # Marker symbols for each value
    symbol_map = {
        0: '^',  # Up arrow
        1: 'v',  # Down arrow
        2: '<',  # Left arrow
        3: '>',  # Right arrow
        4: 's',  # Square (down)
        5: 'o'   # Circle (up)
    }

    # Colors for each value
    color_map = {
        0: 'blue',
        1: 'blue',
        2: 'blue',
        3: 'blue',
        4: 'red',
        5: 'green'
    }

    plt.figure()

    num_rows, num_cols = arr.shape

    # Draw black grid lines (horizontal and vertical)
    for i in range(num_rows + 1):
        plt.hlines(y=i, xmin=0, xmax=num_cols, color='black')
    for j in range(num_cols + 1):
        plt.vlines(x=j, ymin=0, ymax=num_rows, color='black')

    # Plot each cell with the appropriate marker and color at its center
    for i in range(num_rows):
        for j in range(num_cols):
            val = arr[i, j]
            marker = symbol_map[val]
            color = color_map[val]
            # Center each symbol in its cell by adding 0.5 to both x and y
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