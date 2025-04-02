"""Environment detection utilities."""
import importlib.util
from typing import List


def is_in_colab() -> bool:
    """Check if the code is running in Google Colab.
    
    Returns:
        bool: True if running in Google Colab, False otherwise
    """
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ImportError:
        return False


def check_package_installed(package_name: str) -> bool:
    """Check if a package is already installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        bool: True if package is installed, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


def get_missing_packages(required_packages: List[str]) -> List[str]:
    """Get list of required packages that are not installed.
    
    Args:
        required_packages: List of package names to check
        
    Returns:
        List[str]: List of packages that are not installed
    """
    return [pkg for pkg in required_packages if not check_package_installed(pkg)]