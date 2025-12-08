"""Module containing core type instances and definitions for JAX-CTSM.

This module provides the central data structures and type definitions that represent
the key components of the land model state. All types are implemented as immutable
frozen dataclasses to maintain JAX compatibility with pure functions, vectorization,
and JIT compilation.

Original Fortran: clm_instMod.F90

The module serves three primary purposes:
1. Define a unified LandModelState dataclass that holds all component states
2. Provide initialization functions following JAX best practices (pure, immutable)
3. Provide restart I/O functions for defining, writing, and reading state

Key Design Decisions:
- Uses frozen dataclass for immutability and JAX compatibility
- Pure functions replace Fortran subroutines with mutable state
- Restart operations support 'define', 'write', and 'read' modes
- All array types use jax.numpy for JIT compilation
- Field names use '_inst' suffix to match original Fortran naming
- Provides Fortran-style function names for compatibility

JAX Best Practices Applied:
- Pure functions: No side effects, no mutations
- Immutable state: LandModelState is frozen dataclass
- JIT-compatible: All operations use jnp arrays, no Python control flow with arrays
- Vectorized: All operations work on spatial grids via jnp
- Type hints: Full type annotations for all functions and fields
"""

from dataclasses import dataclass
from typing import Any, Optional
import jax.numpy as jnp

# Type aliases for array types
Array = jnp.ndarray

# jax_ctsm is the root package, core is the subpackage, x.py is the module import <class> and aliases
# from jax_ctsm.core.x import <class> as <alias>
# Analogous to "use decompMod, only : bounds_type" in Fortran
from jax_ctsm.core.decomp import Bounds as bounds_type
# Atm2Lnd should be the class which has the method for init_atm2lnd, so just importing class, creating obj and call init method on that object
# should have been the approach.
"""AI: from jax_ctsm.core.atmosphere import Atm2Lnd as atm2lnd_type, init_atm2lnd"""
from jax_ctsm.core.atmosphere import Atm2Lnd as atm2lnd_type # Corrected code
# Similar to previous comment
from jax_ctsm.core.soil import SoilState as soilstate_type # Corrected code
"""AI: from jax_ctsm.core.soil import SoilState as soilstate_type, init_soil_state, init_soil_time_const"""
# Below Waterflux and WaterState type from different modules/ .F90 fiels. If we are keeping flux and state modules
# separate as in Fortran, then we should keep them separate here as well.
from jax_ctsm.core.waterstate import WaterState as waterstate_type # Corrected code
from jax_ctsm.core.waterflux import WaterFlux as waterflux_type # Corrected code
"""AI: from jax_ctsm.core.water import (
    WaterState as waterstate_type,
    WaterFlux as waterflux_type,
    init_water_state,
    init_water_flux,
)"""
# Similar to previous comment
from jax_ctsm.core.canopystate import CanopyState as canopystate_type # Corrected code
from jax_ctsm.core.mlcanopyfluxes import MLCanopyFluxesType as mlcanopy_type # Corrected code
"""AI:from jax_ctsm.core.canopy import (
    CanopyState as canopystate_type,
    MLCanopyState as mlcanopy_type,
    init_canopy_state,
    init_mlcanopy,
)"""
# init_temperature etc need not be imported separately if we are importing Temperature class from temperature module. Similarly for other modules.
"""AI: from jax_ctsm.core.temperature import Temperature as temperature_type, init_temperature
from jax_ctsm.core.energy import EnergyFlux as energyflux_type, init_energy_flux
from jax_ctsm.core.friction import FrictionVelocity as frictionvel_type, init_friction_velocity"""
from jax_ctsm.core.temperature import Temperature as temperature_type # Corrected code
from jax_ctsm.core.energy import EnergyFlux as energyflux_type # Corrected code
from jax_ctsm.core.friction import FrictionVelocity as frictionvel_type # Corrected code
# SurfaceAlbedoType, SolarAbsorbedType, and SurfaceAlbedoMod are in different modules in Fortran, so we should keep them separate here as well.
""" AI: from jax_ctsm.core.albedo import (
    SurfaceAlbedo as surfalb_type,
    SolarAbsorbed as solarabs_type,
    init_surface_albedo,
    init_solar_absorbed,
    init_surface_albedo_time_const,
)"""
from jax_ctsm.core.surfacealbedo import SurfaceAlbedo as surfalb_type # Corrected code
from jax_ctsm.core.solarabsorbed import SolarAbsorbed as solarabs_type # Corrected code 
from jax_ctsm.core.vertical import init_vertical


# MISSED IMPORTS:
from jax_ctsm.core.surfacealbedomod import SurfaceAlbedoMod as SurfaceAlbedoInitTimeConst # Corrected code
from jax_ctsm.core.soilstatetimeconstmod import SoilStateInitTimeConstMod # Corrected code

# TYPE_CHECKING reference (kept for documentation only):
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from jax_ctsm.core.decomp import Bounds as bounds_type
#     from jax_ctsm.core.atmosphere import Atm2Lnd as atm2lnd_type
#     from jax_ctsm.core.soil import SoilState as soilstate_type
#     from jax_ctsm.core.water import WaterState as waterstate_type, WaterFlux as waterflux_type
#     from jax_ctsm.core.canopy import CanopyState as canopystate_type, MLCanopyState as mlcanopy_type
#     from jax_ctsm.core.temperature import Temperature as temperature_type
#     from jax_ctsm.core.energy import EnergyFlux as energyflux_type
#     from jax_ctsm.core.friction import FrictionVelocity as frictionvel_type
#     from jax_ctsm.core.albedo import SurfaceAlbedo as surfalb_type, SolarAbsorbed as solarabs_type


@dataclass(frozen=True)
class LandModelState:
    """Container for all land model state variables.
    
    This immutable dataclass holds all component states for the land model.
    Field names use the '_inst' suffix to match the original Fortran module
    instance variable naming (e.g., `atm2lnd_inst`, `soilstate_inst`, etc.),
    maintaining alignment with the Fortran code for easier comparison and
    validation.
    
    Following JAX best practices, this is a frozen dataclass ensuring immutability.
    State updates always return new LandModelState objects.

    Attributes:
        atm2lnd_inst: Atmosphere-to-land coupling variables and fluxes
        soilstate_inst: Soil state (water content, temperature profile)
        waterstate_inst: Water state (canopy interception, etc.)
        canopystate_inst: Canopy state (LAI, vegetation, etc.)
        temperature_inst: Temperature state (surface, profile, etc.)
        energyflux_inst: Energy flux variables (latent, sensible heat, etc.)
        waterflux_inst: Water flux (infiltration, runoff, etc.)
        frictionvel_inst: Friction velocity and aerodynamic properties
        surfalb_inst: Surface albedo (visible and near-IR bands)
        solarabs_inst: Solar radiation absorption
        mlcanopy_inst: Multi-layer canopy state (CLM-ml specific)
    """
    # Creating the instances of the imported classes
    atm2lnd_inst: atm2lnd_type
    soilstate_inst: soilstate_type
    waterstate_inst: waterstate_type
    canopystate_inst: canopystate_type
    temperature_inst: temperature_type
    energyflux_inst: energyflux_type
    waterflux_inst: waterflux_type
    frictionvel_inst: frictionvel_type
    surfalb_inst: surfalb_type
    solarabs_inst: solarabs_type
    mlcanopy_inst: mlcanopy_type


def clm_instInit(bounds: bounds_type) -> LandModelState:
    """Initialize the full land model state.
    
    Fortran equivalent: clm_instInit subroutine in clm_instMod.F90
    
    This pure function replaces the Fortran subroutine. It creates and returns
    a new immutable LandModelState with all component states initialized.
    The initialization order matches the original Fortran code to preserve any
    implicit dependencies.

    The function demonstrates JAX best practices:
    - Pure function (no side effects, no mutations)
    - Returns immutable state object
    - All arrays use jax.numpy for JIT compilation
    - Suitable for jax.jit, jax.vmap, and gradient computation

    Args:
        bounds: Domain decomposition bounds specifying grid dimensions and
                spatial indices for this computational unit

    Returns:
        LandModelState: Immutable object with all initialized component states

    Notes:
        Initialization sequence (matching original Fortran clm_instInit):
        1. initVertical - Initialize vertical grid structure
        2. atm2lnd_inst%Init - Atmosphere-land coupling
        3. soilstate_inst%Init + SoilStateInitTimeConst - Soil with constants
        4. waterstate_inst%Init - Water state
        5. canopystate_inst%Init - Canopy state
        6. temperature_inst%Init - Temperature
        7. energyflux_inst%Init - Energy fluxes
        8. waterflux_inst%Init - Water fluxes
        9. frictionvel_inst%Init - Friction velocity
        10. surfalb_inst%Init + SurfaceAlbedoInitTimeConst - Albedo with constants
        11. mlcanopy_inst%Init - Multi-layer canopy
        
        This order may be significant if later components depend on earlier ones.

    Example:
        >>> from jax_ctsm.core.decomp import Bounds
        >>> bounds = Bounds(...)  # Create domain bounds
        >>> state = clm_instInit(bounds)
        >>> print(state.temperature_inst)  # Access temperature component
    """
    # Step 1: Initialize vertical grid structure
    # (Used by all subsequent components)
    _ = init_vertical(bounds)

    # Step 2: Initialize atmosphere-land coupling
    atm2lnd_inst = atm2lnd_type() # Corrected code
    atm2lnd_inst = atm2lnd_inst.init(bounds) # Corrected code
    """AI: atm2lnd_inst = init_atm2lnd(bounds)"""

    # Step 3: Initialize soil state with time-invariant properties
    soilstate_inst = soilstate_type() # Corrected code
    soilstate_inst = soilstate_inst.init(bounds) # Corrected code
    # call SoilStateInitTimeConst     (bounds, soilstate_inst) below is analogous
    soilconst = SoilStateInitTimeConstMod() # Corrected code
    soilconst.SoilStateInitTimeConst(bounds, soilstate_inst) # Corrected code
    """AI: soilstate_inst = init_soil_state(bounds)
    soilstate_inst = init_soil_time_const(bounds, soilstate_inst)"""

    # Step 4-9: Initialize remaining core states
    waterstate_inst = waterstate_type() # Corrected code
    waterstate_inst = waterstate_inst.init(bounds) # Corrected code
    canopystate_inst = canopystate_type() # Corrected code
    canopystate_inst = canopystate_inst.init(bounds) # Corrected code
    temperature_inst = temperature_type() # Corrected code
    temperature_inst = temperature_inst.init(bounds) # Corrected code
    energyflux_inst = energyflux_type() # Corrected code
    energyflux_inst = energyflux_inst.init(bounds) # Corrected code
    waterflux_inst = waterflux_type() # Corrected code
    waterflux_inst = waterflux_inst.init(bounds) # Corrected code
    frictionvel_inst = frictionvel_type() # Corrected code
    frictionvel_inst = frictionvel_inst.init(bounds) # Corrected code
    """AI:waterstate_inst = init_water_state(bounds)
    canopystate_inst = init_canopy_state(bounds)
    temperature_inst = init_temperature(bounds)
    energyflux_inst = init_energy_flux(bounds)
    waterflux_inst = init_water_flux(bounds)
    frictionvel_inst = init_friction_velocity(bounds)"""

    # Step 10: Initialize surface properties with time-invariant constants
    suffalb_inst = surfalb_type() # Corrected code
    suffalb_inst = suffalb_inst.init(bounds) # Corrected code
    solarabs_inst = solarabs_type() # Corrected code
    solarabs_inst = solarabs_inst.init(bounds) # Corrected code
    surfacealbedoconst = SurfaceAlbedoInitTimeConst() # Corrected code
    surfacealbedoconst.SurfaceAlbedoInitTimeConst(bounds) # Corrected code
    """AI:surfalb_inst = init_surface_albedo(bounds)
    solarabs_inst = init_solar_absorbed(bounds)
    init_surface_albedo_time_const(bounds)"""

    # Step 11: Initialize multi-layer canopy state (CLM-ml specific)
    mlcanopy_inst = mlcanopy_type() # Corrected code
    mlcanopy_inst = mlcanopy_inst.init(bounds) # Corrected code
    """AI:mlcanopy_inst = init_mlcanopy(bounds)"""

    # Combine all initialized components into immutable state object
    return LandModelState(
        atm2lnd_inst=atm2lnd_inst,
        soilstate_inst=soilstate_inst,
        waterstate_inst=waterstate_inst,
        canopystate_inst=canopystate_inst,
        temperature_inst=temperature_inst,
        energyflux_inst=energyflux_inst,
        waterflux_inst=waterflux_inst,
        frictionvel_inst=frictionvel_inst,
        surfalb_inst=surfalb_inst,
        solarabs_inst=solarabs_inst,
        mlcanopy_inst=mlcanopy_inst,
    )


def clm_instRest(
    bounds: bounds_type,
    ncid: Any,  # netCDF4.Dataset or similar
    flag: str,
) -> Optional[LandModelState]:
    """Define/write/read CLM restart file.
    
    Fortran equivalent: clm_instRest subroutine in clm_instMod.F90
    
    This pure function handles restart file operations. It replaces the Fortran
    subroutine which used a mutable `flag` parameter. Following JAX principles,
    this function:
    - Returns None for 'define' and 'write' modes
    - Returns new LandModelState for 'read' mode
    - Has no side effects beyond file I/O

    Args:
        bounds: Domain decomposition bounds
        ncid: NetCDF file handle (netCDF4.Dataset or similar)
              Expected to be in 'define' mode for flag='define'
              Expected to be in 'write' mode for flag='write'
              Expected to be in 'read' mode for flag='read'
        flag: Operation mode - one of:
              - 'define': Define variables in NetCDF file (returns None)
              - 'write': Write current state to file (returns None)
              - 'read': Read state from file (returns LandModelState)

    Returns:
        None for 'define' and 'write' modes
        New LandModelState for 'read' mode

    Raises:
        ValueError: If flag is not one of 'define', 'write', 'read'
        RuntimeError: If file I/O fails

    Notes:
        In the original Fortran code, most component restart routines are
        commented out. Only mlcanopy_inst%restart is active. This Python
        version maintains that pattern.
        
        Commented-out components that could be enabled in future:
        - atm2lnd_inst%restart
        - soilstate_inst%restart
        - canopystate_inst%restart
        - waterflux_inst%restart

    Example:
        >>> import netCDF4 as nc
        >>> from jax_ctsm.core.decomp import Bounds
        >>> bounds = Bounds(...)
        >>> state = clm_instInit(bounds)
        >>> 
        >>> # Write restart
        >>> with nc.Dataset('restart.nc', 'w') as ds:
        ...     clm_instRest(bounds, ds, 'define')
        ...     clm_instRest(bounds, ds, 'write')
        >>> 
        >>> # Read restart
        >>> with nc.Dataset('restart.nc', 'r') as ds:
        ...     restored_state = clm_instRest(bounds, ds, 'read')
    """
    if flag not in ('define', 'write', 'read'):
        raise ValueError(
            f"Invalid restart flag: '{flag}'. "
            "Must be one of: 'define', 'write', 'read'"
        )

    # Handle 'define' and 'write' modes (both handle component restart operations)
    if flag in ('define', 'write'):
        # Note: Only the MLCanopy component is currently enabled
        # (matching the original Fortran code which has other calls commented out)
        
        # TODO: Re-enable other components as they are translated:
        # _call_component_restart(atm2lnd_inst, bounds, ncid, flag)
        # _call_component_restart(soilstate_inst, bounds, ncid, flag)
        # _call_component_restart(canopystate_inst, bounds, ncid, flag)
        # _call_component_restart(waterflux_inst, bounds, ncid, flag)
        
        _mlcanopy_restart(bounds, ncid, flag) """Should be implemented in respective field and imported here"""
        return None

    # Handle 'read' mode: create new state from restart file
    elif flag == 'read':
        # Initialize with default state first
        state = clm_instInit(bounds)

        # Then override MLCanopy with data from restart file
        # Note: Only MLCanopy is currently enabled in original code
        mlcanopy_inst = _read_mlcanopy_restart(bounds, ncid) """Should be implemented in respective field and imported here"""

        # Return new state with restarted components
        return LandModelState(
            atm2lnd_inst=state.atm2lnd_inst,
            soilstate_inst=state.soilstate_inst,
            waterstate_inst=state.waterstate_inst,
            canopystate_inst=state.canopystate_inst,
            temperature_inst=state.temperature_inst,
            energyflux_inst=state.energyflux_inst,
            waterflux_inst=state.waterflux_inst,
            frictionvel_inst=state.frictionvel_inst,
            surfalb_inst=state.surfalb_inst,
            solarabs_inst=state.solarabs_inst,
            mlcanopy_inst=mlcanopy_inst,
        )



"""Below are AI generated. This wont be required once these modules are translated in their respective files."""
# Stub functions for component restart operations
# These will be implemented when the respective component modules are translated

def _mlcanopy_restart(bounds: bounds_type, ncid: Any, flag: str) -> None:
    """Handle MLCanopy restart (define/write operation).
    
    Args:
        bounds: Domain decomposition bounds
        ncid: NetCDF file handle
        flag: 'define' or 'write'
    """
    # TODO: Implement when MLCanopyFluxesType is translated
    pass


def _read_mlcanopy_restart(bounds: bounds_type, ncid: Any) -> mlcanopy_type:
    """Read MLCanopy state from restart file.
    
    Args:
        bounds: Domain decomposition bounds
        ncid: NetCDF file handle open in read mode
    
    Returns:
        MLCanopyState read from file, or default initialization if not in file
    """
    # TODO: Implement when MLCanopyFluxesType is translated
    # For now, return initialized state
    return init_mlcanopy(bounds)


# Fortran-style function name aliases for compatibility
# These enable calling code that uses Fortran-style names to work with Python implementations
# Maybe be required???
def initVertical(bounds: bounds_type) -> jnp.ndarray:
    """Fortran-compatible alias for vertical grid initialization.
    
    Args:
        bounds: Domain decomposition bounds
    
    Returns:
        Initialized vertical grid array
    """
    return init_vertical(bounds)


def SoilStateInitTimeConst(bounds: bounds_type, soil_state: soilstate_type) -> soilstate_type:
    """Fortran-compatible alias for soil time-constant initialization.
    
    Args:
        bounds: Domain decomposition bounds
        soil_state: Soil state to update with time-constant properties
    
    Returns:
        Updated soil state with time-constant properties
    """
    return init_soil_time_const(bounds, soil_state)


def SurfaceAlbedoInitTimeConst(bounds: bounds_type) -> None:
    """Fortran-compatible alias for surface albedo time-constant initialization.
    
    Args:
        bounds: Domain decomposition bounds
    """
    init_surface_albedo_time_const(bounds)


# Public interface for module initialization
__all__ = [
    'LandModelState',
    'clm_instInit',
    'clm_instRest',
    'initVertical',
    'SoilStateInitTimeConst',
    'SurfaceAlbedoInitTimeConst',
]
