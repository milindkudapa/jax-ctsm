# CTSM Driver: Fortran vs JAX

This document explains the **driver** concept in CTSM and how it translates to JAX.

---

## What is a Driver?

The **driver** is the main orchestrator that:
1. Takes the current state (carbon, nitrogen, temperature, etc.)
2. Calls all physics modules in the correct order
3. Updates the state based on calculated fluxes
4. Returns the new state

Think of it as the **conductor of an orchestra** - it doesn't play any instruments, but coordinates all the musicians (physics modules) to create the symphony (ecosystem simulation).

---

## Fortran CTSM Driver

### Location
`CTSM/src/biogeochem/CNDriverMod.F90`

### Main Subroutine
`CNDriverNoLeaching` - executes one timestep of ecosystem dynamics

### Structure

```fortran
subroutine CNDriverNoLeaching(...)
  ! Inputs: All state variables, forcing, parameters
  ! Outputs: Updated state variables
  
  ! 1. Zero all fluxes
  call SetValues(carbonflux_inst, 0.0)
  call SetValues(nitrogenflux_inst, 0.0)
  
  ! 2. Nitrogen inputs
  call CNNDeposition(...)      ! Atmospheric deposition
  call CNNFixation(...)        ! Biological fixation
  call CNNFert(...)            ! Fertilizer (crops)
  
  ! 3. Maintenance respiration
  call CNMResp(...)
  
  ! 4. Soil biogeochemistry
  call decomp_rate_constants(...)
  call SoilBiogeochemPotential(...)
  call SoilBiogeochemNitrifDenitrif(...)
  
  ! 5. Phenology (phase 1)
  call CNPhenology(..., phase=1)
  
  ! 6. Allocation
  call calc_gpp_mr_availc(...)
  call calc_allometry(...)
  call calc_plant_nutrient_demand(...)
  call SoilBiogeochemCompetition(...)
  call calc_plant_nutrient_competition(...)
  
  ! 7. Growth respiration
  call CNGResp(...)
  
  ! 8. Phenology (phase 2)
  call CNPhenology(..., phase=2)
  
  ! 9. Mortality
  call CNGapMortality(...)
  
  ! 10. Harvest
  call CNHarvest(...)
  
  ! 11. Fire
  call CNFire(...)
  
  ! 12. State updates (3 stages)
  call CStateUpdate1(...)  ! Growth & allocation
  call NStateUpdate1(...)
  
  call CStateUpdate2(...)  ! Mortality & harvest
  call NStateUpdate2(...)
  
  call CStateUpdate3(...)  ! Fire & other
  call NStateUpdate3(...)
  
  ! 13. C14 decay (if enabled)
  call C14Decay(...)
  
end subroutine
```

### Characteristics

**Pros:**
- Well-tested over decades
- Comprehensive (all processes)
- Modular (each process is separate subroutine)

**Cons:**
- In-place mutation (modifies state directly)
- Not differentiable (can't compute gradients)
- Hard to parallelize
- Difficult to JIT compile
- Sequential execution only

---

## JAX-CTSM Driver

### Location
`jax-ctsm/src/jax_ctsm/driver.py` (just created)

### Main Function
`ecosystem_timestep()` - pure functional equivalent

### Structure (Conceptual)

```python
def ecosystem_timestep(
    patch_state: PatchState,
    column_state: ColumnState,
    params: EcosystemParams,
    dt: float,
) -> Tuple[PatchState, ColumnState, EcosystemFluxes]:
    """One timestep of ecosystem dynamics (pure function)."""
    
    # 1. N inputs (TODO)
    # n_deposition = calculate_n_deposition(...)
    
    # 2. Maintenance respiration (✅ IMPLEMENTED)
    mr_fluxes = calculate_maintenance_respiration(patch_state, params.respiration)
    
    # 3. Soil biogeochemistry (TODO)
    # soil_fluxes = soil_biogeochemistry(...)
    
    # 4. Phenology (TODO)
    # phenology_state, phenology_fluxes = phenology(...)
    
    # 5. Allocation (TODO)
    # allocation_fluxes = allocation(...)
    
    # 6. Growth respiration (TODO)
    # gr_fluxes = growth_respiration(...)
    
    # 7. Mortality (TODO)
    # mortality_fluxes = gap_mortality(...)
    
    # 8. Fire (TODO)
    # fire_fluxes = fire_dynamics(...)
    
    # 9. State updates (functional - creates new state)
    new_patch_state = update_patch_state(patch_state, all_fluxes, dt)
    new_column_state = update_column_state(column_state, all_fluxes, dt)
    
    # 10. Return new state + fluxes
    return new_patch_state, new_column_state, fluxes
```

### Characteristics

**Pros:**
- Pure functions (no side effects)
- Differentiable (automatic gradients with `jax.grad`)
- Parallelizable (automatic with `jax.vmap`)
- JIT-compilable (100-1000× speedup)
- GPU-ready (no code changes)
- Immutable state (easier to reason about)

**Cons:**
- Not yet complete (only MR implemented)
- Requires different mindset (functional programming)
- New codebase (less tested than Fortran)

---

## Key Differences: Fortran vs JAX

### 1. State Mutation vs Immutability

**Fortran (mutation):**
```fortran
! Modifies state in-place
carbonstate%leafc(p) = carbonstate%leafc(p) + flux * dt
```

**JAX (immutable):**
```python
# Creates new state
new_leafc = state.carbon.leafc + flux * dt
new_carbon = state.carbon._replace(leafc=new_leafc)
new_state = state._replace(carbon=new_carbon)
```

### 2. Subroutines vs Pure Functions

**Fortran:**
```fortran
subroutine CNMResp(state_inst, flux_inst)
  ! Modifies flux_inst directly
  flux_inst%leaf_mr(p) = ...
end subroutine
```

**JAX:**
```python
def calculate_mr(state: PatchState) -> CarbonFlux:
    # Returns new flux object
    return CarbonFlux(leaf_mr=..., froot_mr=...)
```

### 3. Filters vs Vectorization

**Fortran (filtered loops):**
```fortran
do fp = 1, num_soilp
  p = filter_soilp(fp)  ! Only vegetated patches
  mr(p) = calculate_mr(...)
end do
```

**JAX (vectorized with masking):**
```python
# Compute for all, mask non-vegetated
mr = calculate_mr(state)  # Vectorized over all patches
mr = jnp.where(is_vegetated, mr, 0.0)  # Mask
```

### 4. Sequential vs Composable

**Fortran (sequential calls):**
```fortran
call CNMResp(...)
call CNGResp(...)
call CNAllocation(...)
! Each modifies state
```

**JAX (functional composition):**
```python
# Chain of pure functions
fluxes_mr = maintenance_respiration(state, params)
fluxes_gr = growth_respiration(state, params)
fluxes_alloc = allocation(state, params)

# Combine and update
state = update_state(state, [fluxes_mr, fluxes_gr, fluxes_alloc], dt)
```

---

## Current Implementation Status

### ✅ Implemented in JAX

1. **Spatial hierarchy** - Complete
   - `PatchState`, `ColumnState`, etc.
   - Aggregation functions (`patch_to_column`)
   
2. **Maintenance respiration** - Complete
   - Leaf, root, stem, grain
   - Temperature response
   - All tissue types

3. **Infrastructure** - Complete
   - Testing framework
   - Validation utilities
   - Documentation

### ⏳ TODO (Next Steps)

1. **Allocation** 
   - Available C calculation
   - Allocation coefficients
   - C/N distribution

2. **Growth respiration** 
   - Based on allocation

3. **Phenology** 
   - Leaf onset/offset
   - Crop phenology
   - LAI dynamics

4. **Soil biogeochemistry** 
   - Decomposition
   - Nitrification/denitrification
   - Vertical transport

5. **Fire** 
   - Fire probability
   - Emissions
   - Mortality

### Estimated Timeline

- **TBD**: Core processes (allocation, phenology, soil)
- **TBD**: Complete CN cycle
- **TBD**: Full parity with Fortran CTSM

---

## Example: How to Use JAX Driver (Future)

### Single Timestep

```python
from jax_ctsm.driver import ecosystem_timestep
from jax_ctsm.params import EcosystemParams

# Initialize
params = EcosystemParams(
    respiration=RespirationParams(),
    # allocation=AllocationParams(),
    # etc.
)

# One timestep
new_patch, new_column, fluxes = ecosystem_timestep(
    patch_state, 
    column_state, 
    params,
    dt=1800  # 30 minutes
)

# Check fluxes
print(f"Total MR: {fluxes.carbon.leaf_mr.sum() * 1800:.2f} gC/m2")
```

### Multi-Timestep Simulation

```python
from jax_ctsm.driver import run_simulation

# Run for 1 day (48 timesteps of 30 min)
final_patch, final_column, flux_history = run_simulation(
    initial_patch_state,
    initial_column_state,
    params,
    n_timesteps=48,
    dt=1800
)

# Analyze daily totals
daily_mr = sum([f.carbon.leaf_mr.sum() for f in flux_history]) * 1800
print(f"Daily MR: {daily_mr:.2f} gC/m2/day")
```

### JIT Compilation (100× speedup)

```python
from jax_ctsm.driver import ecosystem_timestep_jit

# First call: slow (compilation)
result = ecosystem_timestep_jit(patch, column, params, dt=1800)

# Subsequent calls: fast!
for _ in range(100):
    result = ecosystem_timestep_jit(patch, column, params, dt=1800)
```

### Vectorized (Batch Processing)

```python
from jax_ctsm.driver import ecosystem_timestep_batch

# Process 1000 patches simultaneously
batch_patches = ...  # [1000 patches]
batch_columns = ...  # [1000 columns]

# Automatically parallelized
new_patches, new_columns, fluxes = ecosystem_timestep_batch(
    batch_patches,
    batch_columns,
    params,
    dt=1800
)
```

### Automatic Differentiation

```python
import jax

def loss_fn(param_value):
    """Optimize respiration Q10 to match observations."""
    params = EcosystemParams(
        respiration=RespirationParams(q10=param_value)
    )
    _, _, fluxes = ecosystem_timestep(patch, column, params)
    
    # Minimize error vs observations
    predicted_mr = fluxes.carbon.leaf_mr.sum()
    return (predicted_mr - observed_mr) ** 2

# Compute gradient
grad_fn = jax.grad(loss_fn)
gradient = grad_fn(1.5)  # dLoss/dQ10

print(f"Gradient: {gradient}")
```

---

## Design Philosophy

### Fortran CTSM
**Imperative programming**: "Do this, then do that"
- Modify state in-place
- Sequential execution
- Side effects everywhere

### JAX-CTSM  
**Functional programming**: "What is the result?"
- Pure functions
- Immutable data
- No side effects

**Benefits of functional approach:**
1. **Easier to test** - pure functions are deterministic
2. **Easier to parallelize** - no shared state to worry about
3. **Differentiable** - automatic gradients
4. **Composable** - functions are like LEGO blocks

---

## Summary

| Aspect | Fortran CTSM | JAX-CTSM |
|--------|--------------|----------|
| **Driver** | `CNDriverNoLeaching` | `ecosystem_timestep` |
| **Style** | Imperative, mutable | Functional, immutable |
| **State updates** | In-place mutation | New state objects |
| **Performance** | Fast (compiled) | Very fast (JIT + GPU) |
| **Gradients** | ❌ Not available | ✅ Automatic |
| **Parallelization** | Manual (MPI) | Automatic (vmap) |
| **GPU** | Requires ports | Automatic |
| **Status** | ✅ Complete | ⏳ In progress (5% done) |

---

## Next Steps

To complete the JAX driver, we need to implement (in order):

1. ✅ **Maintenance respiration** (DONE)
2. **Growth respiration** - depends on allocation
3. **Allocation** - depends on available C
4. **Phenology** - leaf dynamics
5. **Soil biogeochemistry** - decomposition
6. **Fire** - disturbance

Each module follows the same pattern established by maintenance respiration:
- Pure functions
- PyTree-compatible data structures  
- Comprehensive tests
- Documentation

**The driver will automatically work once we have all the pieces!**
