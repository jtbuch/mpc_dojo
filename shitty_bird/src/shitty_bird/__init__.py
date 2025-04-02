"""ShittyBird: A fluid dynamics simulation library built on JAX."""

__version__ = "0.1.0"

# Skip package installation if in Google Colab and packages are already installed
# Make subpackages available at the top level
from . import (
    core,  # noqa: F401
    simulations,  # noqa: F401
    utils,  # noqa: F401
)
from .utils.environment import get_missing_packages, is_in_colab

REQUIRED_JAX_PACKAGES = ["jax", "jaxlib", "chex"]

if is_in_colab():
    missing_packages = get_missing_packages(REQUIRED_JAX_PACKAGES)
    if missing_packages:
        # If packages are missing, they'll be installed by poetry
        # They're declared as optional in pyproject.toml
        pass
    else:
        # Colab with all packages already installed - no need to install them again
        print("Running in Colab with required JAX packages already installed.")