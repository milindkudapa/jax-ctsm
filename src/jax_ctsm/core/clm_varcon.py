"""
JAX translation of CLM Fortran module clm_varcon.

This module provides model constants, preserving the shr_kind_mod context by mapping r8 to jnp.float64.
All names are kept exactly as in the Fortran code. All constants are JIT-compatible and immutable.
"""

from typing import Final
import jax.numpy as jnp

# Fortran kind mapping: use shr_kind_mod, only : r8 => shr_kind_r8
# In JAX/Python, r8 is mapped to jnp.float64 for full precision.
r8: Final = jnp.float64

# Mathematical constants used in CLM
rpi: Final = r8(3.141592654)  # pi

# Physical constants used in CLM
tfrz: Final = r8(273.15)      # Freezing point of water (K)
sb: Final = r8(5.67e-08)      # Stefan-Boltzmann constant (W/m2/K4)
grav: Final = r8(9.80665)      # Gravitational acceleration (m/s2)
vkc: Final = r8(0.4)           # von Karman constant
denh2o: Final = r8(1000.0)     # Density of liquid water (kg/m3)
denice: Final = r8(917.0)      # Density of ice (kg/m3)
tkwat: Final = r8(0.57)        # Thermal conductivity of water (W/m/K)
tkice: Final = r8(2.29)        # Thermal conductivity of ice (W/m/K)
tkair: Final = r8(0.023)       # Thermal conductivity of air (W/m/K)
hfus: Final = r8(0.3337e6)     # Latent heat of fusion for water at 0 C (J/kg)
hvap: Final = r8(2.5010e6)     # Latent heat of evaporation (J/kg)
hsub: Final = r8(2.8347e6)     # Latent heat of sublimation (J/kg)
cpice: Final = r8(2.11727e3)   # Specific heat of ice (J/kg/K)
cpliq: Final = r8(4.188e3)     # Specific heat of water (J/kg/K)

# Constants used in CLM for bedrock
thk_bedrock: Final = r8(3.0)        # Thermal conductivity of saturated granitic rock (W/m/K)
csol_bedrock: Final = r8(2.0e6)      # Vol. heat capacity of granite/sandstone (J/m3/K)
zmin_bedrock: Final = r8(0.4)        # Minimum depth to bedrock (soil depth) [m]

# Special value flags used in CLM
spval: Final = r8(1.0e36)           # Special value for real data
ispval: Final = -9999               # Special value for integer data

__all__ = [
    "r8",
    "rpi",
    "tfrz",
    "sb",
    "grav",
    "vkc",
    "denh2o",
    "denice",
    "tkwat",
    "tkice",
    "tkair",
    "hfus",
    "hvap",
    "hsub",
    "cpice",
    "cpliq",
    "thk_bedrock",
    "csol_bedrock",
    "zmin_bedrock",
    "spval",
    "ispval",
]