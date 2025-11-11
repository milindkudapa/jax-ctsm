# CTSM Spatial Hierarchy in JAX

## Overview

CTSM uses a nested spatial hierarchy to represent the heterogeneity of land surface processes. This document explains how the hierarchy is implemented in JAX-CTSM and how to work with it.

## The Four-Level Hierarchy

```
Gridcell (lat/lon grid cell, ~1°×1°)
    │
    └── Landunit (land cover type)
            │
            └── Column (soil/snow vertical column)
                    │
                    └── Patch (Plant Functional Type - PFT)
```

### 1. Gridcell

- **Definition**: Geographic grid cell (typically 1° latitude × 1° longitude)
- **Purpose**: Top-level spatial unit for global simulations
- **Contains**: Multiple landunits

**Example**: A gridcell might contain 60% vegetated land, 30% crop, 10% urban

### 2. Landunit

- **Definition**: Land cover type within a gridcell
- **Types**:
  - Vegetated (natural vegetation)
  - Crop (managed agriculture)
  - Urban
  - Glacier
  - Lake
- **Purpose**: Separate fundamentally different land cover types
- **Contains**: One or more columns

### 3. Column

- **Definition**: Vertical soil/snow column
- **Purpose**: Track vertical processes (hydrology, soil biogeochemistry)
- **Usually**: 1:1 relationship with landunit (except crops can have multiple)
- **Contains**: One or more patches

**Key Variables at Column Level**:
- Soil temperature (by layer)
- Soil moisture
- Soil biogeochemistry
- Snow pack

### 4. Patch

- **Definition**: Plant Functional Type (PFT) - the finest resolution
- **Purpose**: Track vegetation-specific processes
- **Examples**: 
  - Needleleaf Evergreen Temperate Tree
  - Broadleaf Deciduous Tropical Tree  
  - C3 Crop
  - C3 Arctic Grass

**Key Variables at Patch Level**:
- Vegetation carbon and nitrogen pools
- Leaf area index (LAI)
- Canopy structure
- Photosynthesis
- Maintenance respiration

## Implementation in JAX

### Data Structure Design

JAX-CTSM uses **flat arrays** with **index mappings** instead of nested objects for performance:

```python
# Instead of nested objects:
gridcell.landunits[i].columns[j].patches[k]

# We use flat arrays with indices:
patch_values[patch_idx]
column_index[patch_idx]  # Parent column
landunit_index[column_idx]  # Parent landunit
```

### Spatial Info Structure

```python
@dataclass
class SpatialInfo:
    patch_index: jnp.ndarray      # [n_patches] Patch ID
    column_index: jnp.ndarray     # [n_patches] Parent column for each patch
    landunit_index: jnp.ndarray   # [n_columns] Parent landunit for each column
    gridcell_index: jnp.ndarray   # [n_landunits] Parent gridcell
    weights: jnp.ndarray          # [n_patches] Area weights
    pft_type: jnp.ndarray         # [n_patches] PFT type ID
    is_vegetated: jnp.ndarray     # [n_patches] Boolean mask
```

### Example: Creating Spatial Info

```python
import jax.numpy as jnp
from jax_ctsm.core.hierarchy import SpatialInfo

# Simple example: 1 gridcell, 1 landunit, 1 column, 3 patches
spatial = SpatialInfo(
    patch_index=jnp.array([0, 1, 2]),
    column_index=jnp.array([0, 0, 0]),      # All patches in same column
    landunit_index=jnp.array([0]),          # Column in landunit 0
    gridcell_index=jnp.array([0]),          # Landunit in gridcell 0
    weights=jnp.array([0.5, 0.3, 0.2]),     # 50%, 30%, 20% area
    pft_type=jnp.array([1, 1, 6]),          # PFT IDs
    is_vegetated=jnp.array([True, True, True]),
)
```

## Aggregation: Patch → Column → Landunit → Gridcell

Many variables need to be aggregated up the hierarchy (e.g., for mass balance or output).

### Patch to Column

```python
from jax_ctsm.core.hierarchy import patch_to_column

# Aggregate patch-level values to column using area weights
column_values = patch_to_column(
    patch_values=patch_mr,           # [n_patches]
    patch_weights=weights,            # [n_patches]
    column_indices=column_index,      # [n_patches]
    n_columns=n_columns,
)
# Returns: [n_columns]
```

**Example**:
```python
# 3 patches in 1 column with weights [0.5, 0.3, 0.2]
patch_mr = jnp.array([1.0, 2.0, 1.5])  # gC/m2/s
weights = jnp.array([0.5, 0.3, 0.2])

column_mr = patch_to_column(patch_mr, weights, jnp.array([0, 0, 0]), 1)
# Result: [0.5*1.0 + 0.3*2.0 + 0.2*1.5] = [1.4]
```

### Broadcasting: Column → Patch

Sometimes we need to broadcast column-level values to patches:

```python
from jax_ctsm.core.hierarchy import column_to_patch

# Broadcast column values to all patches
patch_values = column_to_patch(
    column_values=soil_temp,         # [n_columns, n_layers]
    column_indices=column_index,      # [n_patches]
)
# Returns: [n_patches, n_layers]
```

## Process-Level Considerations

### Where Does Each Process Operate?

| Process | Level | Why |
|---------|-------|-----|
| Photosynthesis | Patch | PFT-specific leaf properties |
| Maintenance Respiration | Patch | PFT-specific tissue N content |
| Stomatal Conductance | Patch | PFT-specific responses |
| Phenology | Patch | PFT-specific leaf dynamics |
| Soil Decomposition | Column | Shared soil environment |
| Soil Hydrology | Column | Vertical water flow |
| Nitrification/Denitrification | Column | Soil microbial processes |
| Soil Temperature | Column | Shared thermal environment |

### Example: Maintenance Respiration

Maintenance respiration operates at **patch level** but needs **column-level** soil temperature:

```python
def root_maintenance_respiration(
    root_n: jnp.ndarray,              # [n_patches] - patch level
    soil_temp: jnp.ndarray,           # [n_columns, n_layers] - column level
    column_indices: jnp.ndarray,      # [n_patches] - mapping
    ...
) -> jnp.ndarray:
    # Get soil temp for each patch from its parent column
    patch_soil_temp = soil_temp[column_indices, :]  # [n_patches, n_layers]
    
    # Calculate MR with patch-specific N and column soil temp
    ...
```

## Filtering and Subsetting

CTSM uses **filters** to operate on subsets of patches/columns:

```python
# Filter for vegetated patches only
veg_filter = jnp.where(spatial.is_vegetated)[0]
veg_patches = patch_values[veg_filter]

# Filter for soil columns (exclude lake, glacier)
soil_filter = jnp.where(column_type == SOIL)[0]
soil_columns = column_values[soil_filter]
```

### Vectorization with Filters

Use `vmap` to vectorize over filtered subsets:

```python
import jax

# Calculate only for vegetated patches
veg_indices = jnp.where(spatial.is_vegetated)[0]

def process_patch(idx):
    return calculate_mr(patch_state[idx], params)

# Vectorize over vegetated patches
veg_results = jax.vmap(process_patch)(veg_indices)
```

## Performance Tips

### ✅ DO

1. **Use flat arrays** with index mappings
2. **Batch operations** across patches/columns
3. **Use `vmap`** for vectorization
4. **Pre-allocate** output arrays

```python
# Good: Vectorized over all patches
mr_all = jax.vmap(calculate_mr)(patch_states, params)
```

### ❌ DON'T

1. **Avoid Python loops** over spatial dimensions
2. **Don't use nested objects** (slow, not JIT-compatible)
3. **Don't dynamically change shapes** inside JIT

```python
# Bad: Python loop
for p in range(n_patches):
    mr[p] = calculate_mr(patch_states[p], params)
```

## Practical Examples

### Example 1: Single Patch Simulation

```python
# Minimal setup for testing
spatial = SpatialInfo(
    patch_index=jnp.array([0]),
    column_index=jnp.array([0]),
    landunit_index=jnp.array([0]),
    gridcell_index=jnp.array([0]),
    weights=jnp.array([1.0]),
    pft_type=jnp.array([1]),
    is_vegetated=jnp.array([True]),
)
```

### Example 2: Multi-PFT Column

```python
# One column with 3 PFTs (e.g., mixed forest)
spatial = SpatialInfo(
    patch_index=jnp.array([0, 1, 2]),
    column_index=jnp.array([0, 0, 0]),  # All same column
    landunit_index=jnp.array([0]),
    gridcell_index=jnp.array([0]),
    weights=jnp.array([0.6, 0.3, 0.1]),  # 60% tree1, 30% tree2, 10% grass
    pft_type=jnp.array([1, 2, 12]),
    is_vegetated=jnp.array([True, True, True]),
)
```

### Example 3: Multiple Columns in Gridcell

```python
# 2 columns (veg + crop) with multiple patches each
spatial = SpatialInfo(
    patch_index=jnp.array([0, 1, 2, 3]),
    column_index=jnp.array([0, 0, 1, 1]),  # First 2 in col 0, next 2 in col 1
    landunit_index=jnp.array([0, 1]),      # Col 0 in landunit 0, col 1 in landunit 1
    gridcell_index=jnp.array([0, 0]),      # Both landunits in gridcell 0
    weights=jnp.array([0.35, 0.25, 0.3, 0.1]),
    pft_type=jnp.array([1, 6, 15, 16]),    # Trees, grass, crop1, crop2
    is_vegetated=jnp.array([True, True, True, True]),
)
```

## Debugging Tips

### Visualize Hierarchy

```python
def print_hierarchy(spatial: SpatialInfo):
    """Print spatial hierarchy structure."""
    for p in range(len(spatial.patch_index)):
        c = spatial.column_index[p]
        l = spatial.landunit_index[c] if c < len(spatial.landunit_index) else -1
        g = spatial.gridcell_index[l] if l >= 0 and l < len(spatial.gridcell_index) else -1
        print(f"Patch {p} (PFT {spatial.pft_type[p]}, wt={spatial.weights[p]:.2f}) "
              f"→ Column {c} → Landunit {l} → Gridcell {g}")
```

### Check Area Weights

```python
def check_weights(spatial: SpatialInfo):
    """Verify weights sum to 1.0 per column."""
    for c in range(spatial.column_index.max() + 1):
        mask = spatial.column_index == c
        total = spatial.weights[mask].sum()
        print(f"Column {c}: weight sum = {total:.6f}")
        assert jnp.allclose(total, 1.0), f"Weights don't sum to 1.0 for column {c}"
```

## Further Reading

- CTSM Technical Note: [CLM5.0 Documentation](https://escomp.github.io/ctsm-docs/)
- Original CTSM spatial hierarchy: `CTSM/src/main/subgridMod.F90`
- JAX transformations: [JAX Documentation](https://jax.readthedocs.io/)
