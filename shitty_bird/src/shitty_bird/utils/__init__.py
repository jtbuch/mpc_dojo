"""Utility functions for ShittyBird."""

from .environment import check_package_installed, get_missing_packages, is_in_colab
from .visualization import fig_constructor, fig_updater, plot_state

__all__ = [
    "fig_constructor", 
    "fig_updater", 
    "plot_state",
    "is_in_colab", 
    "check_package_installed", 
    "get_missing_packages"
]