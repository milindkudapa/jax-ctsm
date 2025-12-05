"""
JAX translation of CLM Fortran module clm_varpar.

This module provides model parameters for vertical layer structure.
All names are kept exactly as in the Fortran code. Uses immutable NamedTuple
to represent mutable Fortran module variables in a JAX-compatible way.
"""

from typing import Final, NamedTuple, Literal
import jax.numpy as jnp

# Fortran kind mapping: use shr_kind_mod, only : r8 => shr_kind_r8
# In JAX/Python, r8 is mapped to jnp.float64 for full precision.
r8: Final = jnp.float64

# Radiation waveband parameters (constants)
numrad: Final = 2   # Number of radiation wavebands
ivis: Final = 1     # Visible waveband index
inir: Final = 2     # Near-infrared waveband index
mxpft: Final = 78   # Maximum number of plant functional types


class ClmVarpar(NamedTuple):
    """
    CLM vertical layer parameters.
    
    In Fortran, these are module-level variables that get initialized.
    In JAX, we use an immutable NamedTuple for JIT compatibility.
    
    Attributes:
        nlevsno: Maximum number of snow layers
        nlevsoi: Number of hydrologically active soil layers
        nlevgrnd: Number of ground layers (including hydrologically inactive)
    """
    nlevsno: int
    nlevsoi: int
    nlevgrnd: int


def clm_varpar_init(clm_phys: Literal["CLM5_0", "CLM4_5"] = "CLM5_0") -> ClmVarpar:
    """
    Initialize module variables based on CLM physics version.
    
    CLM4.5 and CLM5 have different snow/soil layer configurations:
    - CLM5_0: 12 snow layers, 20 soil layers, 25 total ground layers
    - CLM4_5: 5 snow layers, 10 soil layers, 15 total ground layers
    
    Args:
        clm_phys: CLM physics version string. Either "CLM5_0" or "CLM4_5".
                  Default is "CLM5_0".
    
    Returns:
        ClmVarpar: Initialized parameter structure with layer counts.
    
    Examples:
        >>> params = clm_varpar_init("CLM5_0")
        >>> params.nlevsno
        12
        >>> params.nlevsoi
        20
        >>> params.nlevgrnd
        25
        
        >>> params = clm_varpar_init("CLM4_5")
        >>> params.nlevsno
        5
        >>> params.nlevsoi
        10
        >>> params.nlevgrnd
        15
    """
    if clm_phys == "CLM5_0":
        nlevsno = 12
        nlevsoi = 20
        nlevgrnd = nlevsoi + 5
    elif clm_phys == "CLM4_5":
        nlevsno = 5
        nlevsoi = 10
        nlevgrnd = 15
    else:
        raise ValueError(f"Unknown clm_phys version: {clm_phys}. Must be 'CLM5_0' or 'CLM4_5'.")
    
    return ClmVarpar(
        nlevsno=nlevsno,
        nlevsoi=nlevsoi,
        nlevgrnd=nlevgrnd
    )


__all__ = [
    "r8",
    "numrad",
    "ivis",
    "inir",
    "mxpft",
    "ClmVarpar",
    "clm_varpar_init",
]
