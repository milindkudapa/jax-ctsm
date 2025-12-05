"""
JAX translation of CLM Fortran module spmdMod.

This module provides SPMD (Single Program Multiple Data) initialization constants.
In the original Fortran code, this handles parallel processing setup. 
For the JAX translation, we preserve the module structure and constants.

The module uses shr_kind_mod for type definitions, mapped to JAX types.
All names are kept exactly as in the Fortran code for consistency.
"""

from typing import Final
import jax.numpy as jnp

# Fortran kind mapping: use shr_kind_mod, only : r8 => shr_kind_r8
# In JAX/Python, r8 is mapped to jnp.float64 for full precision.
r8: Final = jnp.float64

# SPMD initialization constants
# proc 0 logical for printing msgs
masterproc: Final = True

__all__ = [
    "r8",
    "masterproc",
]
