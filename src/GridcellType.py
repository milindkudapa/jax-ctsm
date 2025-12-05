"""
JAX translation of CLM Fortran module GridcellType.

This module provides the gridcell data type for storing spatial coordinates.
In Fortran, this uses allocatable pointers. In JAX, we use immutable arrays.
All names are kept exactly as in the Fortran code.
"""

from typing import Final, NamedTuple
import jax.numpy as jnp

# Fortran kind mapping: use shr_kind_mod, only : r8 => shr_kind_r8
# In JAX/Python, r8 is mapped to jnp.float64 for full precision.
r8: Final = jnp.float64

# Import special value from clm_varcon: use clm_varcon, only : nan => spval
# In the Fortran code, this is used to initialize arrays
try:
    from jax_ctsm.core.clm_varcon import spval as nan
except ImportError:
    # Fallback if module structure is different
    nan: Final = r8(1.0e36)


class gridcell_type(NamedTuple):
    """
    Gridcell data type for spatial coordinates.
    
    In Fortran, this is a derived type with allocatable pointers.
    In JAX, we use an immutable NamedTuple for JIT compatibility.
    
    Attributes:
        latdeg: Latitude in degrees [n_gridcells]
        londeg: Longitude in degrees [n_gridcells]
    """
    latdeg: jnp.ndarray  # [n_gridcells]
    londeg: jnp.ndarray  # [n_gridcells]


def Init(begg: int, endg: int) -> gridcell_type:
    """
    Initialize gridcell data structure.
    
    Creates arrays of latitude and longitude for the specified range
    of gridcell indices, initialized to NaN (special value).
    
    In Fortran, this allocates pointer arrays. In JAX, we create
    immutable arrays with the specified shape.
    
    Args:
        begg: Beginning gridcell index (inclusive)
        endg: Ending gridcell index (inclusive)
    
    Returns:
        gridcell_type: Initialized gridcell structure with lat/lon
                      arrays filled with NaN values.
    
    Examples:
        >>> grc = Init(0, 9)
        >>> grc.latdeg.shape
        (10,)
        >>> grc.londeg.shape
        (10,)
        
        >>> # Initialize for single gridcell
        >>> grc = Init(0, 0)
        >>> grc.latdeg.shape
        (1,)
    
    Note:
        In Fortran, arrays use indices [begg:endg]. In JAX/Python,
        we use 0-based indexing, so the array size is (endg - begg + 1).
    """
    # Calculate array size (Fortran uses inclusive indexing)
    n_gridcells = endg - begg + 1
    
    # Initialize arrays with NaN (special value)
    latdeg = jnp.full(n_gridcells, nan, dtype=r8)
    londeg = jnp.full(n_gridcells, nan, dtype=r8)
    
    return gridcell_type(latdeg=latdeg, londeg=londeg)


# Create a module-level instance analogous to Fortran's 'grc'
# In actual use, this should be initialized by calling Init()
# This is a placeholder following Fortran's pattern: type(gridcell_type), public, target :: grc
grc: gridcell_type = None  # Should be initialized via Init() before use


__all__ = [
    "r8",
    "nan",
    "gridcell_type",
    "Init",
    "grc",
]
