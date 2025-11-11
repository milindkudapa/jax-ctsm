"""JAX-CTSM: Community Terrestrial Systems Model in JAX.

A differentiable, GPU-accelerated reimplementation of CTSM ecosystem dynamics.
"""

__version__ = "0.1.0"

# Core data structures
from jax_ctsm.core import (
    PatchState,
    ColumnState,
    NitrogenState,
    CarbonState,
    CanopyState,
    TemperatureState,
    SoilState,
    CarbonFlux,
    SpatialInfo,
)

# Physics modules
from jax_ctsm.physics import calculate_maintenance_respiration

# Parameters
from jax_ctsm.params import RespirationParams

# Driver
from jax_ctsm.driver import (
    ecosystem_timestep,
    ecosystem_timestep_jit,
    ecosystem_timestep_batch,
    run_simulation,
    run_simulation_scan,
    EcosystemParams,
    EcosystemFluxes,
)

__all__ = [
    # Core data structures
    "PatchState",
    "ColumnState", 
    "NitrogenState",
    "CarbonState",
    "CanopyState",
    "TemperatureState",
    "SoilState",
    "CarbonFlux",
    "SpatialInfo",
    # Physics
    "calculate_maintenance_respiration",
    # Parameters
    "RespirationParams",
    # Driver
    "ecosystem_timestep",
    "ecosystem_timestep_jit",
    "ecosystem_timestep_batch",
    "run_simulation",
    "run_simulation_scan",
    "EcosystemParams",
    "EcosystemFluxes",
]
