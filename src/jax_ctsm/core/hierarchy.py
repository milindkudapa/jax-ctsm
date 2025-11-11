"""
Spatial Hierarchy for JAX-CTSM.

CTSM uses a nested spatial hierarchy:
    Gridcell → Landunit → Column → Patch

- Gridcell: ~1 degree lat/lon grid cell
- Landunit: Land cover type within gridcell (vegetated, crop, urban, glacier, lake)
- Column: Soil/snow column (usually 1:1 with landunit, but crops can have multiple)
- Patch: Plant functional type (PFT) - the fundamental unit for vegetation

This module defines PyTree-compatible data structures for each level.
All arrays use batch dimension first for vmap compatibility.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp


class SpatialInfo(NamedTuple):
    """Spatial indexing and hierarchy information.
    
    Attributes:
        patch_index: Patch index within simulation [n_patches]
        column_index: Parent column index for each patch [n_patches]
        landunit_index: Parent landunit index for each column [n_columns]
        gridcell_index: Parent gridcell index for each landunit [n_landunits]
        weights: Area weights for aggregation [n_patches]
        pft_type: Plant functional type index [n_patches]
        is_vegetated: Whether patch is vegetated [n_patches]
    """
    patch_index: jnp.ndarray  # [n_patches]
    column_index: jnp.ndarray  # [n_patches]
    landunit_index: jnp.ndarray  # [n_columns]
    gridcell_index: jnp.ndarray  # [n_landunits]
    weights: jnp.ndarray  # [n_patches]
    pft_type: jnp.ndarray  # [n_patches] (integer)
    is_vegetated: jnp.ndarray  # [n_patches] (boolean)


class NitrogenState(NamedTuple):
    """Patch-level nitrogen state pools.
    
    All units in gN/m2 unless otherwise specified.
    Arrays have shape [n_patches] or [n_patches, n_pools].
    
    Attributes:
        leafn: Leaf nitrogen
        frootn: Fine root nitrogen  
        livestemn: Live stem nitrogen (woody plants)
        livecrootn: Live coarse root nitrogen (woody plants)
        deadstemn: Dead stem nitrogen (woody plants)
        deadcrootn: Dead coarse root nitrogen (woody plants)
        reproductiven: Reproductive tissue nitrogen (crops) [n_patches, n_repr]
        retransn: Retranslocation pool
    """
    leafn: jnp.ndarray
    frootn: jnp.ndarray
    livestemn: jnp.ndarray
    livecrootn: jnp.ndarray
    deadstemn: jnp.ndarray
    deadcrootn: jnp.ndarray
    reproductiven: jnp.ndarray  # [n_patches, n_repr]
    retransn: jnp.ndarray


class CarbonState(NamedTuple):
    """Patch-level carbon state pools.
    
    All units in gC/m2 unless otherwise specified.
    
    Attributes:
        leafc: Leaf carbon
        frootc: Fine root carbon
        livestemc: Live stem carbon
        livecrootc: Live coarse root carbon
        deadstemc: Dead stem carbon
        deadcrootc: Dead coarse root carbon
        reproductivec: Reproductive carbon (crops) [n_patches, n_repr]
        cpool: Temporary photosynthate pool
        xsmrpool: Excess maintenance respiration pool
    """
    leafc: jnp.ndarray
    frootc: jnp.ndarray
    livestemc: jnp.ndarray
    livecrootc: jnp.ndarray
    deadstemc: jnp.ndarray
    deadcrootc: jnp.ndarray
    reproductivec: jnp.ndarray  # [n_patches, n_repr]
    cpool: jnp.ndarray
    xsmrpool: jnp.ndarray


class CarbonFlux(NamedTuple):
    """Patch-level carbon fluxes.
    
    All units in gC/m2/s unless otherwise specified.
    
    Attributes:
        gpp: Gross primary production
        leaf_mr: Leaf maintenance respiration
        froot_mr: Fine root maintenance respiration
        livestem_mr: Live stem maintenance respiration
        livecroot_mr: Live coarse root maintenance respiration
        reproductive_mr: Reproductive tissue maintenance respiration [n_patches, n_repr]
    """
    gpp: jnp.ndarray
    leaf_mr: jnp.ndarray
    froot_mr: jnp.ndarray
    livestem_mr: jnp.ndarray
    livecroot_mr: jnp.ndarray
    reproductive_mr: jnp.ndarray  # [n_patches, n_repr]


class CanopyState(NamedTuple):
    """Canopy structure state.
    
    Attributes:
        lai_sun: Sunlit leaf area index [m2/m2]
        lai_shade: Shaded leaf area index [m2/m2]
        lmr_sun: Sunlit leaf MR from photosynthesis [umol CO2/m2/s]
        lmr_shade: Shaded leaf MR from photosynthesis [umol CO2/m2/s]
        frac_veg_nosno: Fraction vegetation not covered by snow [0 or 1]
    """
    lai_sun: jnp.ndarray
    lai_shade: jnp.ndarray
    lmr_sun: jnp.ndarray
    lmr_shade: jnp.ndarray
    frac_veg_nosno: jnp.ndarray


class TemperatureState(NamedTuple):
    """Temperature state at patch and column levels.
    
    Attributes:
        t_ref2m: 2m air temperature [K] - patch level [n_patches]
        t_10day: 10-day running mean 2m temperature [K] - patch level [n_patches]
        t_soisno: Soil/snow temperature [K] - column level [n_columns, n_levgrnd]
            In CTSM, soil temperature is shared across patches in the same column.
            Patches access their column's temperature via column_index mapping.
    """
    t_ref2m: jnp.ndarray  # [n_patches]
    t_10day: jnp.ndarray  # [n_patches]
    t_soisno: jnp.ndarray  # [n_columns, n_levgrnd]


class SoilState(NamedTuple):
    """Soil state at column level.
    
    Attributes:
        crootfr: Fraction of roots in each layer [n_patches, n_levgrnd]
        depth: Soil layer depths [m] [n_levgrnd]
    """
    crootfr: jnp.ndarray  # [n_patches, n_levgrnd]
    depth: jnp.ndarray  # [n_levgrnd]


class PatchState(NamedTuple):
    """Complete state for a patch (PFT level).
    
    This is the main data structure for ecosystem processes.
    All processes operate on PatchState and return updated states/fluxes.
    
    Attributes:
        carbon: Carbon pools
        nitrogen: Nitrogen pools
        canopy: Canopy structure
        temperature: Temperature state
        soil: Soil properties
        spatial: Spatial hierarchy information
        pft_params: PFT-specific parameters (woody, is_crop, etc.)
    """
    carbon: CarbonState
    nitrogen: NitrogenState
    canopy: CanopyState
    temperature: TemperatureState
    soil: SoilState
    spatial: SpatialInfo
    pft_params: dict  # PFT-specific parameters


class ColumnState(NamedTuple):
    """State for a soil column.
    
    Attributes:
        soil: Soil biogeochemistry state
        temperature: Column-level temperature
        spatial: Column spatial info
    """
    soil: SoilState
    temperature: jnp.ndarray  # [n_columns, n_levgrnd]
    spatial: dict  # Column indexing info


class LandunitState(NamedTuple):
    """State for a landunit (land cover type).
    
    Attributes:
        ltype: Landunit type (vegetated=1, crop=2, etc.)
        weight: Area weight within gridcell
        columns: Child column states
    """
    ltype: int
    weight: float
    columns: list[ColumnState]


class GridcellState(NamedTuple):
    """State for a gridcell (top of hierarchy).
    
    Attributes:
        lat: Latitude [degrees]
        lon: Longitude [degrees]
        landunits: Child landunit states
    """
    lat: float
    lon: float
    landunits: list[LandunitState]


# Utility functions for spatial aggregation
def patch_to_column(
    patch_values: jnp.ndarray,
    patch_weights: jnp.ndarray,
    column_indices: jnp.ndarray,
    n_columns: int
) -> jnp.ndarray:
    """Aggregate patch-level values to column level.
    
    Args:
        patch_values: Values at patch level [n_patches]
        patch_weights: Patch area weights [n_patches]
        column_indices: Parent column index for each patch [n_patches]
        n_columns: Total number of columns
        
    Returns:
        Column-level aggregated values [n_columns]
    """
    weighted_values = patch_values * patch_weights
    
    # Use scatter-add for aggregation (differentiable)
    column_values = jnp.zeros(n_columns)
    column_values = column_values.at[column_indices].add(weighted_values)
    
    # Normalize by total weight per column
    column_weights = jnp.zeros(n_columns)
    column_weights = column_weights.at[column_indices].add(patch_weights)
    
    # Avoid division by zero
    return jnp.where(column_weights > 0, column_values / column_weights, 0.0)


def column_to_patch(
    column_values: jnp.ndarray,
    column_indices: jnp.ndarray
) -> jnp.ndarray:
    """Broadcast column-level values to patch level.
    
    Args:
        column_values: Values at column level [n_columns]
        column_indices: Parent column index for each patch [n_patches]
        
    Returns:
        Patch-level values [n_patches]
    """
    return column_values[column_indices]
