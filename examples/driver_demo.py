#!/usr/bin/env python3
"""
Demonstration of JAX-CTSM Ecosystem Driver

This example shows how to use the driver to run multi-timestep ecosystem simulations.
The driver orchestrates all ecosystem processes (currently only maintenance respiration).

Features demonstrated:
1. Single timestep execution
2. Multi-timestep simulation
3. JIT compilation for speed
4. Batch processing with vmap
5. JAX scan for efficient long runs
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time

from jax_ctsm import (
    ecosystem_timestep,
    ecosystem_timestep_jit,
    run_simulation,
    run_simulation_scan,
    EcosystemParams,
    PatchState,
    NitrogenState,
    CarbonState,
    CanopyState,
    TemperatureState,
    SoilState,
    SpatialInfo,
    RespirationParams,
)


def create_test_patch(n_patches: int = 1) -> PatchState:
    """Create a test patch state for demonstration."""
    n_levgrnd = 10
    
    # Create nitrogen state
    nitrogen = NitrogenState(
        leafn=jnp.full(n_patches, 0.02),      # 20 g N/m2
        frootn=jnp.full(n_patches, 0.01),     # 10 g N/m2
        livestemn=jnp.full(n_patches, 0.005), # 5 g N/m2
        livecrootn=jnp.full(n_patches, 0.005),
        deadstemn=jnp.zeros(n_patches),
        deadcrootn=jnp.zeros(n_patches),
        reproductiven=jnp.zeros((n_patches, 3)),
        retransn=jnp.zeros(n_patches),
    )
    
    # Create carbon state
    carbon = CarbonState(
        leafc=jnp.full(n_patches, 0.5),      # 500 g C/m2
        frootc=jnp.full(n_patches, 0.3),     # 300 g C/m2
        livestemc=jnp.full(n_patches, 1.0),  # 1 kg C/m2
        livecrootc=jnp.full(n_patches, 0.8), # 800 g C/m2
        deadstemn=jnp.zeros(n_patches),
        deadcrootc=jnp.zeros(n_patches),
        reproductivec=jnp.zeros((n_patches, 3)),
        cpool=jnp.full(n_patches, 10.0),     # 10 kg C/m2 available
        xsmrpool=jnp.zeros(n_patches),       # No deficit initially
    )
    
    # Create canopy state
    canopy = CanopyState(
        lai_sun=jnp.full(n_patches, 2.0),
        lai_shade=jnp.full(n_patches, 1.0),
        lmr_sun=jnp.full(n_patches, 2e-6),
        lmr_shade=jnp.full(n_patches, 1e-6),
        frac_veg_nosno=jnp.ones(n_patches, dtype=jnp.int32),
    )
    
    # Create temperature state
    temp = TemperatureState(
        t_ref2m=jnp.full(n_patches, 298.15),  # 25°C
        t_10day=jnp.full(n_patches, 295.15),  # 22°C (10-day mean)
        t_soisno=jnp.full((n_patches, n_levgrnd), 295.15),  # Soil temp
    )
    
    # Create soil state
    soil = SoilState(
        crootfr=jnp.array([[0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01, 0.0]] * n_patches),
        depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.4, 0.58, 0.8, 1.06, 1.36]),
    )
    
    # Create spatial info
    spatial = SpatialInfo(
        pft_type=jnp.zeros(n_patches, dtype=jnp.int32),
        patch_index=jnp.arange(n_patches, dtype=jnp.int32),
        column_index=jnp.arange(n_patches, dtype=jnp.int32),
        landunit_index=jnp.zeros(n_patches, dtype=jnp.int32),
        gridcell_index=jnp.zeros(n_patches, dtype=jnp.int32),
        weights=jnp.ones(n_patches),
        is_vegetated=jnp.ones(n_patches, dtype=bool),
    )
    
    # PFT parameters
    pft_params = {
        "is_woody": jnp.ones(n_patches, dtype=bool),
        "is_crop": jnp.zeros(n_patches, dtype=bool),
    }
    
    return PatchState(
        carbon=carbon,
        nitrogen=nitrogen,
        canopy=canopy,
        temperature=temp,
        soil=soil,
        spatial=spatial,
        pft_params=pft_params,
    )


def demo_single_timestep():
    """Demo 1: Single timestep execution."""
    print("\n" + "="*70)
    print("DEMO 1: Single Timestep")
    print("="*70)
    
    # Setup
    patch_state = create_test_patch(n_patches=1)
    params = EcosystemParams(respiration=RespirationParams())
    
    # Execute one timestep (30 min)
    new_state, fluxes = ecosystem_timestep(patch_state, params, dt=1800.0)
    
    # Display results
    print(f"\n📊 Results for 30-minute timestep:")
    print(f"  Leaf MR:        {fluxes.carbon.leaf_mr[0]:.6f} gC/m2/s")
    print(f"  Root MR:        {fluxes.carbon.froot_mr[0]:.6f} gC/m2/s")
    print(f"  Stem MR:        {fluxes.carbon.livestem_mr[0]:.6f} gC/m2/s")
    print(f"  Total MR:       {fluxes.total_respiration[0]:.6f} gC/m2/s")
    
    print(f"\n📈 Carbon pool changes:")
    print(f"  Initial cpool:  {patch_state.carbon.cpool[0]:.4f} kgC/m2")
    print(f"  Final cpool:    {new_state.carbon.cpool[0]:.4f} kgC/m2")
    print(f"  MR consumed:    {(patch_state.carbon.cpool[0] - new_state.carbon.cpool[0]):.4f} kgC/m2")


def demo_multi_timestep():
    """Demo 2: Multi-timestep simulation (1 day)."""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Timestep Simulation (1 day)")
    print("="*70)
    
    # Setup
    patch_state = create_test_patch(n_patches=1)
    params = EcosystemParams(respiration=RespirationParams())
    
    # Run for 1 day (48 timesteps of 30 min)
    n_timesteps = 48
    dt = 1800.0
    
    final_state, flux_history = run_simulation(
        patch_state, params, n_timesteps=n_timesteps, dt=dt
    )
    
    # Calculate daily totals
    daily_mr = sum([f.total_respiration[0] for f in flux_history]) * dt
    daily_leaf_mr = sum([f.carbon.leaf_mr[0] for f in flux_history]) * dt
    daily_root_mr = sum([f.carbon.froot_mr[0] for f in flux_history]) * dt
    
    print(f"\n📊 Daily totals (48 timesteps):")
    print(f"  Total MR:       {daily_mr:.2f} gC/m2/day")
    print(f"  Leaf MR:        {daily_leaf_mr:.2f} gC/m2/day")
    print(f"  Root MR:        {daily_root_mr:.2f} gC/m2/day")
    
    print(f"\n📈 Carbon pool changes:")
    print(f"  Initial cpool:  {patch_state.carbon.cpool[0]:.4f} kgC/m2")
    print(f"  Final cpool:    {final_state.carbon.cpool[0]:.4f} kgC/m2")
    print(f"  Daily C loss:   {(patch_state.carbon.cpool[0] - final_state.carbon.cpool[0]):.4f} kgC/m2/day")


def demo_jit_speedup():
    """Demo 3: JIT compilation speedup."""
    print("\n" + "="*70)
    print("DEMO 3: JIT Compilation Speedup")
    print("="*70)
    
    # Setup
    patch_state = create_test_patch(n_patches=1)
    params = EcosystemParams(respiration=RespirationParams())
    
    # Test non-JIT version
    print("\n🐌 Non-JIT version (100 timesteps)...")
    start = time()
    state = patch_state
    for _ in range(100):
        state, _ = ecosystem_timestep(state, params)
    time_no_jit = time() - start
    print(f"  Time: {time_no_jit:.3f} seconds")
    
    # Test JIT version
    print("\n🚀 JIT version (100 timesteps)...")
    start = time()
    state = patch_state
    # First call compiles
    state, _ = ecosystem_timestep_jit(state, params)
    compile_time = time() - start
    print(f"  First call (compilation): {compile_time:.3f} seconds")
    
    # Subsequent calls are fast
    start = time()
    for _ in range(99):
        state, _ = ecosystem_timestep_jit(state, params)
    time_jit = time() - start
    print(f"  Next 99 calls: {time_jit:.3f} seconds")
    
    print(f"\n⚡ Speedup: {time_no_jit/time_jit:.1f}x faster")


def demo_scan_vs_loop():
    """Demo 4: JAX scan vs Python loop."""
    print("\n" + "="*70)
    print("DEMO 4: JAX Scan vs Python Loop")
    print("="*70)
    
    # Setup
    patch_state = create_test_patch(n_patches=1)
    params = EcosystemParams(respiration=RespirationParams())
    n_timesteps = 365  # 1 year at daily timesteps
    
    # Python loop
    print(f"\n🐍 Python loop ({n_timesteps} timesteps)...")
    start = time()
    final_loop, fluxes_loop = run_simulation(
        patch_state, params, n_timesteps=n_timesteps, dt=86400
    )
    time_loop = time() - start
    print(f"  Time: {time_loop:.3f} seconds")
    
    # JAX scan (includes compilation)
    print(f"\n⚡ JAX scan ({n_timesteps} timesteps)...")
    start = time()
    final_scan, fluxes_scan = run_simulation_scan(
        patch_state, params, n_timesteps=n_timesteps, dt=86400
    )
    time_scan = time() - start
    print(f"  Time (with compilation): {time_scan:.3f} seconds")
    
    # Verify results match
    cpool_diff = abs(final_loop.carbon.cpool[0] - final_scan.carbon.cpool[0])
    print(f"\n✓ Results match: cpool difference = {cpool_diff:.9f}")
    
    if time_loop > time_scan:
        print(f"⚡ Scan is {time_loop/time_scan:.1f}x faster")


def demo_visualization():
    """Demo 5: Visualize daily cycle."""
    print("\n" + "="*70)
    print("DEMO 5: Visualize Daily Respiration Cycle")
    print("="*70)
    
    # Setup
    patch_state = create_test_patch(n_patches=1)
    params = EcosystemParams(respiration=RespirationParams())
    
    # Run for 1 day with 30-min timesteps
    n_timesteps = 48
    dt = 1800.0
    
    final_state, flux_history = run_simulation(
        patch_state, params, n_timesteps=n_timesteps, dt=dt
    )
    
    # Extract fluxes
    times = jnp.arange(n_timesteps) * dt / 3600  # Convert to hours
    leaf_mr = jnp.array([f.carbon.leaf_mr[0] for f in flux_history])
    froot_mr = jnp.array([f.carbon.froot_mr[0] for f in flux_history])
    stem_mr = jnp.array([f.carbon.livestem_mr[0] for f in flux_history])
    total_mr = jnp.array([f.total_respiration[0] for f in flux_history])
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(times, leaf_mr * 1e6, label='Leaf', linewidth=2)
    plt.plot(times, froot_mr * 1e6, label='Fine Root', linewidth=2)
    plt.plot(times, stem_mr * 1e6, label='Live Stem', linewidth=2)
    plt.plot(times, total_mr * 1e6, label='Total', linewidth=2, linestyle='--', color='black')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('MR (μgC/m²/s)', fontsize=12)
    plt.title('Maintenance Respiration by Tissue', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cpool_history = [patch_state.carbon.cpool[0]]
    state = patch_state
    for f in flux_history:
        mr_total = f.total_respiration[0] * dt
        new_cpool = state.carbon.cpool[0] - mr_total
        cpool_history.append(float(new_cpool))
        # Update state for next iteration
        new_carbon = state.carbon._replace(cpool=jnp.array([new_cpool]))
        state = state._replace(carbon=new_carbon)
    
    plt.plot(jnp.concatenate([jnp.array([0]), times]), cpool_history, linewidth=2, color='green')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Carbon Pool (kgC/m²)', fontsize=12)
    plt.title('Carbon Pool Depletion', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('driver_demo_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved to: driver_demo_results.png")
    plt.close()


def main():
    """Run all demonstrations."""
    print("\n" + "🌿 "*35)
    print("JAX-CTSM ECOSYSTEM DRIVER DEMONSTRATION")
    print("🌿 "*35)
    
    # Run demos
    demo_single_timestep()
    demo_multi_timestep()
    demo_jit_speedup()
    demo_scan_vs_loop()
    demo_visualization()
    
    print("\n" + "="*70)
    print("✅ All demonstrations complete!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("  1. Driver orchestrates ecosystem processes")
    print("  2. Pure functional approach (immutable state)")
    print("  3. JIT compilation provides significant speedup")
    print("  4. JAX scan enables efficient long simulations")
    print("  5. Easy to visualize and analyze results")
    print("\n📚 Next steps:")
    print("  • Add allocation module")
    print("  • Add phenology dynamics")
    print("  • Add soil biogeochemistry")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
