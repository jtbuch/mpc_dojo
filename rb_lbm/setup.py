"""rb_lbm: A Rayleigh-Benard convection simulation library using Lattice Boltzmann Method built on JAX."""

from setuptools import setup, find_packages

setup(
    name="rb_lbm",
    version="0.1.0",
    description="A library for simulating Rayleigh-Benard convection using Lattice Boltzmann Method",
    author="Jatan Buch",
    author_email="jb4625@columbia.edu",
    license="MIT",
    packages=find_packages(),
    py_modules=['core', 'utils', 'simulations'],
    install_requires=[
        "jax>=0.4.23",
        "jaxlib>=0.4.23",
        "chex>=0.1.7",
        "numpy>=1.26.4",
        "matplotlib>=3.8.3",
        "tqdm>=4.66.2",
        "multipledispatch>=1.0.0",
        "moviepy>=1.0.3",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "mypy>=1.8.0",
            "ruff>=0.2.2",
        ],
        "notebook": [
            "ipython>=7.0.0",
        ],
        "default": [
            "jax>=0.4.23",
            "jaxlib>=0.4.23",
            "chex>=0.1.7",
        ],
    },
    python_requires=">=3.9, <3.12",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)