"""
JAX-CTSM Main Driver - POC Implementation

This is the JAX equivalent of CTSM's CNDriverNoLeaching in Fortran.
It orchestrates all ecosystem processes for one timestep.

CURRENT STATUS: Working POC with maintenance respiration
- ✅ Maintenance respiration fully functional
- ✅ State updates for carbon pools
- ✅ Multi-timestep simulation
- ✅ JIT compilation
- ✅ Batch processing

FUTURE: Add allocation, phenology, soil biogeochem, fire, etc.

DESIGN NOTE - Column State:
  Currently, the driver operates only on PatchState. Column-level data
  (e.g., soil temperature) is embedded in PatchState for convenience.
  
  When implementing soil biogeochemistry, the driver will need to be
  extended to accept and return ColumnState as well:
  
    def ecosystem_timestep(
        patch_state: PatchState,
        column_state: ColumnState,  # <- Add this
        params: EcosystemParams,
        dt: float
    ) -> Tuple[PatchState, ColumnState, EcosystemFluxes]:  # <- And this
  
  This is intentionally deferred to keep the POC simple.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List

from jax_ctsm.core.hierarchy import PatchState, CarbonFlux, CarbonState
from jax_ctsm.physics.maintenance_respiration import calculate_maintenance_respiration
from jax_ctsm.params.respiration import RespirationParams


class EcosystemParams(NamedTuple):
    """Complete parameter set for all ecosystem processes.
    
    Attributes:
        respiration: Maintenance respiration parameters
    """
    respiration: RespirationParams
    # Future additions:
    # allocation: AllocationParams
    # phenology: PhenologyParams
    # soil_bgc: SoilBiogeochemParams
    # fire: FireParams


class EcosystemFluxes(NamedTuple):
    """All ecosystem fluxes for one timestep.
    
    Attributes:
        carbon: Carbon fluxes (gC/m2/s)
        total_respiration: Total ecosystem respiration (gC/m2/s)
    """
    carbon: CarbonFlux
    total_respiration: jnp.ndarray  # Sum of all respiration terms


def update_carbon_pools(
    carbon_state: CarbonState,
    carbon_flux: CarbonFlux,
    dt: float,
) -> CarbonState:
    """Update carbon pools based on fluxes.
    
    Args:
        carbon_state: Current carbon state
        carbon_flux: Carbon fluxes (gC/m2/s)
        dt: Timestep (seconds)
        
    Returns:
        Updated carbon state
        
    Note:
        In full implementation, this would include:
        - GPP additions to cpool
        - Allocation from cpool to tissues
        - Mortality losses
        - Fire losses
        
        For POC, we only have respiration losses.
    """
    # Calculate total MR per patch
    total_mr = (
        carbon_flux.leaf_mr + 
        carbon_flux.froot_mr + 
        carbon_flux.livestem_mr + 
        carbon_flux.livecroot_mr +
        jnp.sum(carbon_flux.reproductive_mr, axis=-1)  # Sum over reproductive pools
    )
    
    # In a full model:
    # 1. GPP would add to cpool
    # 2. Allocation would move from cpool to tissues
    # 3. MR would come from cpool (if available) or tissues
    # 4. Growth respiration would consume from cpool
    
    # For POC: Respiration comes from cpool
    # If cpool is insufficient, track the deficit in xsmrpool
    new_cpool = carbon_state.cpool - total_mr * dt
    
    # xsmrpool tracks cumulative carbon deficit (excess MR demand)
    # When cpool goes negative, we:
    # 1. Calculate the deficit (negative value)
    # 2. Add the absolute value to xsmrpool (since deficit is negative, subtracting it adds)
    # 3. Zero out cpool (can't be negative)
    # This matches CTSM convention: xsmrpool accumulates carbon that should have
    # been respired but wasn't available, representing a "carbon debt"
    deficit = jnp.minimum(new_cpool, 0.0)  # deficit ≤ 0 (negative when cpool insufficient)
    new_xsmrpool = carbon_state.xsmrpool - deficit  # Subtracting negative adds to pool
    new_cpool = jnp.maximum(new_cpool, 0.0)  # Floor at zero
    
    # Create updated carbon state
    return carbon_state._replace(
        cpool=new_cpool,
        xsmrpool=new_xsmrpool,
    )


def ecosystem_timestep(
    patch_state: PatchState,
    params: EcosystemParams,
    dt: float = 1800.0,
) -> Tuple[PatchState, EcosystemFluxes]:
    """Execute one ecosystem dynamics timestep.
    
    This is the JAX equivalent of CTSM's CNDriverNoLeaching.
    Current POC version includes only maintenance respiration.
    
    Args:
        patch_state: Current patch-level state
        params: All ecosystem parameters
        dt: Timestep in seconds (default: 1800s = 30 min)
        
    Returns:
        (new_patch_state, fluxes): Updated state and calculated fluxes
        
    Note:
        - Pure function (no side effects)
        - Immutable state (creates new state objects)
        - GPU-compatible
        - Differentiable
        
    Example:
        >>> params = EcosystemParams(respiration=RespirationParams())
        >>> new_state, fluxes = ecosystem_timestep(patch_state, params, dt=1800)
        >>> print(f"Total MR: {fluxes.total_respiration.sum() * 1800:.2f} gC/m2")
    """
    
    # ============================================
    # PHASE 1: NITROGEN INPUTS (Future)
    # ============================================
    # Future: N deposition, fixation, fertilizer
    
    # ============================================
    # PHASE 2: MAINTENANCE RESPIRATION ✅
    # ============================================
    mr_fluxes = calculate_maintenance_respiration(patch_state, params.respiration)
    
    # ============================================
    # PHASE 3-7: OTHER PROCESSES (Future)
    # ============================================
    # Future: soil biogeochem, phenology, allocation, growth resp, mortality, fire
    
    # ============================================
    # PHASE 8: CALCULATE TOTAL FLUXES
    # ============================================
    # Calculate total respiration
    total_mr = (
        mr_fluxes.leaf_mr + 
        mr_fluxes.froot_mr + 
        mr_fluxes.livestem_mr + 
        mr_fluxes.livecroot_mr +
        jnp.sum(mr_fluxes.reproductive_mr, axis=-1)
    )
    
    # Combine fluxes
    carbon_fluxes = CarbonFlux(
        gpp=jnp.zeros_like(total_mr),  # Future: from photosynthesis
        leaf_mr=mr_fluxes.leaf_mr,
        froot_mr=mr_fluxes.froot_mr,
        livestem_mr=mr_fluxes.livestem_mr,
        livecroot_mr=mr_fluxes.livecroot_mr,
        reproductive_mr=mr_fluxes.reproductive_mr,
    )
    
    fluxes = EcosystemFluxes(
        carbon=carbon_fluxes,
        total_respiration=total_mr,
    )
    
    # ============================================
    # PHASE 9: UPDATE STATE
    # ============================================
    # Update carbon pools based on fluxes
    new_carbon = update_carbon_pools(patch_state.carbon, carbon_fluxes, dt)
    new_patch_state = patch_state._replace(carbon=new_carbon)
    
    return new_patch_state, fluxes


# ============================================
# JIT-COMPILED VERSION FOR SPEED
# ============================================

def ecosystem_timestep_jit(
    patch_state: PatchState,
    params: EcosystemParams,
    dt: float = 1800.0,
) -> Tuple[PatchState, EcosystemFluxes]:
    """JIT-compiled version of ecosystem_timestep for speed.
    
    First call will be slow (compilation), subsequent calls will be fast.
    The dt parameter is static, so changing it will trigger recompilation.
    
    Args:
        patch_state: Patch state
        params: Ecosystem parameters
        dt: Timestep (seconds) - static argument, recompiles on change
        
    Returns:
        (new_state, fluxes)
        
    Example:
        >>> # First call: compiles for dt=1800
        >>> new_state, fluxes = ecosystem_timestep_jit(state, params, dt=1800)
        >>> 
        >>> # Subsequent calls with same dt: fast!
        >>> for _ in range(100):
        >>>     new_state, fluxes = ecosystem_timestep_jit(state, params, dt=1800)
        >>>
        >>> # Different dt: recompiles
        >>> new_state, fluxes = ecosystem_timestep_jit(state, params, dt=3600)
    """
    return ecosystem_timestep(patch_state, params, dt)


# Create the JIT-compiled version with static_argnames
ecosystem_timestep_jit = jax.jit(ecosystem_timestep_jit, static_argnames=['dt'])


# ============================================
# VECTORIZED VERSION FOR BATCHES
# ============================================

def ecosystem_timestep_batch(
    patch_states: PatchState,  # Batched over patches
    params: EcosystemParams,
    dt: float = 1800.0,
) -> Tuple[PatchState, EcosystemFluxes]:
    """Vectorized timestep over multiple patches.
    
    Uses JAX vmap for efficient parallel processing.
    
    Args:
        patch_states: Batch of patch states (all arrays have leading batch dim)
        params: Shared parameters (same for all patches)
        dt: Timestep (seconds)
        
    Returns:
        (batched_states, batched_fluxes)
        
    Example:
        >>> # Process 100 patches simultaneously
        >>> batch_states = ...  # PatchState with arrays [100, ...]
        >>> new_states, fluxes = ecosystem_timestep_batch(batch_states, params)
        >>> print(f"Mean MR: {fluxes.total_respiration.mean():.6f} gC/m2/s")
    """
    # Vectorize over first dimension (batch of patches)
    batched_step = jax.vmap(
        lambda p: ecosystem_timestep(p, params, dt),
        in_axes=0,  # Vectorize patch state over first axis
    )
    
    return batched_step(patch_states)


# ============================================
# MULTI-TIMESTEP SIMULATION
# ============================================

def run_simulation(
    initial_patch_state: PatchState,
    params: EcosystemParams,
    n_timesteps: int,
    dt: float = 1800.0,
    store_history: bool = True,
) -> Tuple[PatchState, List[EcosystemFluxes]]:
    """Run multi-timestep simulation.
    
    Args:
        initial_patch_state: Starting patch state
        params: Ecosystem parameters
        n_timesteps: Number of timesteps to run
        dt: Timestep size (seconds)
        store_history: Whether to store flux history (default: True)
        
    Returns:
        (final_patch_state, flux_history)
        
    Example:
        >>> # Run for 1 day (48 timesteps of 30 min)
        >>> final_state, flux_history = run_simulation(
        >>>     patch_state, params, n_timesteps=48, dt=1800
        >>> )
        >>> 
        >>> # Calculate daily total MR
        >>> daily_mr = sum([f.total_respiration.sum() for f in flux_history]) * 1800
        >>> print(f"Daily MR: {daily_mr:.2f} gC/m2/day")
    """
    patch_state = initial_patch_state
    flux_history = [] if store_history else None
    
    for step in range(n_timesteps):
        patch_state, fluxes = ecosystem_timestep(patch_state, params, dt)
        if store_history:
            flux_history.append(fluxes)
    
    return patch_state, flux_history if store_history else []


# ============================================
# SCAN-BASED SIMULATION (More efficient for long runs)
# ============================================

def run_simulation_scan(
    initial_patch_state: PatchState,
    params: EcosystemParams,
    n_timesteps: int,
    dt: float = 1800.0,
) -> Tuple[PatchState, EcosystemFluxes]:
    """Run simulation using JAX scan (more memory efficient).
    
    More efficient than Python loop for long simulations.
    Uses JAX's scan to compile the entire loop.
    
    Args:
        initial_patch_state: Starting state
        params: Ecosystem parameters  
        n_timesteps: Number of steps
        dt: Timestep (seconds)
        
    Returns:
        (final_state, flux_history): where flux_history is an EcosystemFluxes
            with arrays having leading dimension [n_timesteps, ...]
        
    Note:
        This is faster and more memory efficient than run_simulation()
        for long runs, but takes longer to compile on first call.
        flux_history arrays have shape [n_timesteps, n_patches] instead of
        being a list of EcosystemFluxes objects.
        
    Example:
        >>> # Run for 1 year (17520 timesteps)
        >>> final_state, flux_history = run_simulation_scan(
        >>>     patch_state, params, n_timesteps=17520, dt=1800
        >>> )
        >>> # Access total MR for all timesteps
        >>> annual_mr = flux_history.total_respiration.sum() * 1800
    """
    def step_fn(state, _):
        """Single timestep function for scan."""
        new_state, fluxes = ecosystem_timestep(state, params, dt)
        return new_state, fluxes
    
    # Run scan over timesteps
    final_state, flux_history = jax.lax.scan(
        step_fn, 
        initial_patch_state, 
        jnp.arange(n_timesteps)
    )
    
    return final_state, flux_history


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("JAX-CTSM ECOSYSTEM DRIVER - POC")
    print("="*70)
    
    print("\n📋 Implementation Status:")
    print("  ✅ Maintenance respiration (all tissues)")
    print("  ✅ State updates (carbon pools)")
    print("  ✅ Multi-timestep simulation")
    print("  ✅ JIT compilation")
    print("  ✅ Batch processing (vmap)")
    print("  ⏳ Allocation (future)")
    print("  ⏳ Phenology (future)")
    print("  ⏳ Soil biogeochemistry (future)")
    print("  ⏳ Fire (future)")
    
    print("\n🔧 Available Functions:")
    print("  • ecosystem_timestep()        - Single timestep")
    print("  • ecosystem_timestep_jit()    - JIT-compiled version")
    print("  • ecosystem_timestep_batch()  - Vectorized over patches")
    print("  • run_simulation()            - Multi-timestep (Python loop)")
    print("  • run_simulation_scan()       - Multi-timestep (JAX scan)")
    
    print("\n📖 Example Usage:")
    print("""
    from jax_ctsm.driver import ecosystem_timestep, EcosystemParams
    from jax_ctsm.params import RespirationParams
    
    # Setup
    params = EcosystemParams(respiration=RespirationParams())
    
    # Single timestep
    new_state, fluxes = ecosystem_timestep(patch_state, params, dt=1800)
    
    # Multi-timestep (1 day)
    final_state, flux_history = run_simulation(
        patch_state, params, n_timesteps=48, dt=1800
    )
    
    # Analyze
    daily_mr = sum([f.total_respiration.sum() for f in flux_history]) * 1800
    print(f"Daily MR: {daily_mr:.2f} gC/m2/day")
    """)
    
    print("\n" + "="*70)
    print("See examples/basic_maintenance_respiration.py for full example")
    print("="*70 + "\n")
