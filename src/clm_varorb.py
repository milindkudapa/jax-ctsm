"""
Translation of the Fortran module `clm_varorb` to Python using JAX.

This module defines orbital parameters.
"""

from typing import NamedTuple
import jax.numpy as jnp

class OrbitalParameters(NamedTuple):
    """
    Orbital parameters.

    Attributes:
        eccen: Orbital eccentricity factor.
        obliqr: Earth's obliquity in radians.
        lambm0: Mean longitude of perihelion at the vernal equinox (radians).
        mvelpp: Earth's moving vernal equinox longitude of perihelion plus pi (radians).
    """
    eccen: float
    obliqr: float
    lambm0: float
    mvelpp: float

# Initialize orbital parameters with default values
def initialize_orbital_parameters() -> OrbitalParameters:
    """
    Initialize orbital parameters with default values.

    Returns:
        OrbitalParameters: Initialized orbital parameters.
    """
    return OrbitalParameters(
        eccen=0.0,  # Default orbital eccentricity factor
        obliqr=0.0,  # Default Earth's obliquity in radians
        lambm0=0.0,  # Default mean longitude of perihelion at the vernal equinox
        mvelpp=0.0   # Default Earth's moving vernal equinox longitude of perihelion plus pi
    )