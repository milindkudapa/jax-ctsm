"""
Basic example of maintenance respiration calculation.

This example demonstrates:
1. Creating patch state from scratch
2. Calculating maintenance respiration
3. Visualizing results
4. Parameter sensitivity analysis
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax_ctsm.core.hierarchy import (
    PatchState,
    NitrogenState,
    CarbonState,
    CanopyState,
    TemperatureState,
    SoilState,
    SpatialInfo,
)
from jax_ctsm.physics.maintenance_respiration import calculate_maintenance_respiration
from jax_ctsm.params.respiration import RespirationParams


def create_simple_patch(
    temperature_c: float = 25.0,
    leaf_n: float = 50.0,
    root_n: float = 30.0,
    stem_n: float = 100.0,
    is_woody: bool = True,
) -> PatchState:
    """Create a simple single-patch state for testing.
    
    Args:
        temperature_c: Air temperature in Celsius
        leaf_n: Leaf nitrogen [gN/m2]
        root_n: Root nitrogen [gN/m2]
        stem_n: Stem nitrogen [gN/m2]
        is_woody: Whether patch is woody
        
    Returns:
        Single-patch state
    """
    n_levgrnd = 10
    n_repr = 2
    
    # Convert temperature to Kelvin
    temp_k = temperature_c + 273.15
    
    # Nitrogen pools
    nitrogen = NitrogenState(
        leafn=jnp.array([leaf_n]),
        frootn=jnp.array([root_n]),
        livestemn=jnp.array([stem_n if is_woody else 0.0]),
        livecrootn=jnp.array([stem_n * 0.8 if is_woody else 0.0]),
        deadstemn=jnp.zeros(1),
        deadcrootn=jnp.zeros(1),
        reproductiven=jnp.zeros((1, n_repr)),
        retransn=jnp.zeros(1),
    )
    
    # Carbon pools (not used for MR but required)
    carbon = CarbonState(
        leafc=jnp.zeros(1),
        frootc=jnp.zeros(1),
        livestemc=jnp.zeros(1),
        livecrootc=jnp.zeros(1),
        deadstemn=jnp.zeros(1),
        deadcrootc=jnp.zeros(1),
        reproductivec=jnp.zeros((1, n_repr)),
        cpool=jnp.zeros(1),
        xsmrpool=jnp.zeros(1),
    )
    
    # Canopy
    canopy = CanopyState(
        lai_sun=jnp.array([2.5]),
        lai_shade=jnp.array([1.5]),
        lmr_sun=jnp.array([1.0]),  # umol/m2/s
        lmr_shade=jnp.array([0.7]),
        frac_veg_nosno=jnp.array([1.0]),
    )
    
    # Temperature
    # Soil temp decreases with depth
    soil_temps = jnp.array([[temp_k - i * 0.5 for i in range(n_levgrnd)]])
    
    temperature = TemperatureState(
        t_ref2m=jnp.array([temp_k]),
        t_10day=jnp.array([temp_k - 2.0]),  # Slightly cooler 10-day mean
        t_soisno=soil_temps,
    )
    
    # Soil (exponential root distribution)
    root_fracs = np.exp(-np.arange(n_levgrnd) * 0.3)
    root_fracs = root_fracs / root_fracs.sum()
    
    soil = SoilState(
        crootfr=jnp.array([root_fracs]),
        depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.40, 0.58, 0.80, 1.06, 1.36]),
    )
    
    # Spatial info
    spatial = SpatialInfo(
        patch_index=jnp.array([0]),
        column_index=jnp.array([0]),
        landunit_index=jnp.array([0]),
        gridcell_index=jnp.array([0]),
        weights=jnp.array([1.0]),
        pft_type=jnp.array([1]),
        is_vegetated=jnp.array([True]),
    )
    
    # PFT params
    pft_params = {
        "is_woody": jnp.array([is_woody]),
        "is_crop": jnp.array([False]),
    }
    
    return PatchState(
        carbon=carbon,
        nitrogen=nitrogen,
        canopy=canopy,
        temperature=temperature,
        soil=soil,
        spatial=spatial,
        pft_params=pft_params,
    )


def example_basic_calculation():
    """Example 1: Basic MR calculation."""
    print("=" * 60)
    print("Example 1: Basic Maintenance Respiration")
    print("=" * 60)
    
    # Create a temperate forest patch at 20°C
    patch = create_simple_patch(temperature_c=20.0, is_woody=True)
    params = RespirationParams()
    
    # Calculate MR
    fluxes = calculate_maintenance_respiration(patch, params)
    
    # Print results
    print(f"\nPatch conditions:")
    print(f"  Temperature: 20°C (reference)")
    print(f"  Leaf N: {patch.nitrogen.leafn[0]:.1f} gN/m2")
    print(f"  Root N: {patch.nitrogen.frootn[0]:.1f} gN/m2")
    print(f"  Stem N: {patch.nitrogen.livestemn[0]:.1f} gN/m2")
    
    print(f"\nMaintenance Respiration (gC/m2/s):")
    print(f"  Leaf MR:      {fluxes.leaf_mr[0]:.8f}")
    print(f"  Root MR:      {fluxes.froot_mr[0]:.8f}")
    print(f"  Stem MR:      {fluxes.livestem_mr[0]:.8f}")
    print(f"  Coarse root:  {fluxes.livecroot_mr[0]:.8f}")
    print(f"  Total MR:     {(fluxes.leaf_mr[0] + fluxes.froot_mr[0] + fluxes.livestem_mr[0] + fluxes.livecroot_mr[0]):.8f}")
    
    # Convert to daily and annual
    total_mr_daily = (fluxes.leaf_mr[0] + fluxes.froot_mr[0] + 
                      fluxes.livestem_mr[0] + fluxes.livecroot_mr[0]) * 86400
    total_mr_annual = total_mr_daily * 365
    
    print(f"\n  Daily total:  {total_mr_daily:.3f} gC/m2/day")
    print(f"  Annual total: {total_mr_annual:.1f} gC/m2/year")


def example_temperature_response():
    """Example 2: Temperature sensitivity."""
    print("\n" + "=" * 60)
    print("Example 2: Temperature Response (Q10 = 1.5)")
    print("=" * 60)
    
    # Temperature range
    temps = np.arange(-5, 35, 2)  # -5°C to 35°C
    params = RespirationParams()
    
    # Calculate MR at each temperature
    total_mr = []
    for temp_c in temps:
        patch = create_simple_patch(temperature_c=temp_c, is_woody=True)
        fluxes = calculate_maintenance_respiration(patch, params)
        mr_sum = (fluxes.leaf_mr[0] + fluxes.froot_mr[0] + 
                  fluxes.livestem_mr[0] + fluxes.livecroot_mr[0])
        total_mr.append(float(mr_sum))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(temps, np.array(total_mr) * 86400, 'b-', linewidth=2, label='Total MR')
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Maintenance Respiration (gC/m²/day)', fontsize=12)
    plt.title('Temperature Sensitivity of Maintenance Respiration', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('mr_temperature_response.png', dpi=150)
    print("\nPlot saved to: mr_temperature_response.png")
    
    # Print some key values
    idx_0 = np.argmin(np.abs(temps - 0))
    idx_20 = np.argmin(np.abs(temps - 20))
    idx_30 = np.argmin(np.abs(temps - 30))
    
    print(f"\nMR at different temperatures:")
    print(f"   0°C: {total_mr[idx_0] * 86400:.3f} gC/m2/day")
    print(f"  20°C: {total_mr[idx_20] * 86400:.3f} gC/m2/day")
    print(f"  30°C: {total_mr[idx_30] * 86400:.3f} gC/m2/day")
    print(f"\nQ10 check (20→30°C): {total_mr[idx_30] / total_mr[idx_20]:.3f} (expected: 1.5)")


def example_parameter_sensitivity():
    """Example 3: Parameter sensitivity."""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Sensitivity")
    print("=" * 60)
    
    # Base case
    patch = create_simple_patch(temperature_c=20.0, is_woody=True)
    
    # Vary Q10
    q10_values = [1.2, 1.5, 2.0, 2.5]
    
    print("\nQ10 Sensitivity (at 20°C):")
    for q10 in q10_values:
        params = RespirationParams(q10=q10)
        fluxes = calculate_maintenance_respiration(patch, params)
        total = (fluxes.leaf_mr[0] + fluxes.froot_mr[0] + 
                fluxes.livestem_mr[0] + fluxes.livecroot_mr[0]) * 86400
        print(f"  Q10 = {q10:.1f}: {total:.3f} gC/m2/day")
    
    # Vary base rate
    br_multipliers = [0.5, 1.0, 1.5, 2.0]
    
    print("\nBase Rate Sensitivity (at 20°C):")
    base_br = RespirationParams().br
    for mult in br_multipliers:
        params = RespirationParams(br=base_br * mult, br_root=base_br * mult)
        fluxes = calculate_maintenance_respiration(patch, params)
        total = (fluxes.leaf_mr[0] + fluxes.froot_mr[0] + 
                fluxes.livestem_mr[0] + fluxes.livecroot_mr[0]) * 86400
        print(f"  {mult:.1f}x base rate: {total:.3f} gC/m2/day")


def example_woody_vs_grass():
    """Example 4: Woody vs grass comparison."""
    print("\n" + "=" * 60)
    print("Example 4: Woody Plant vs Grass")
    print("=" * 60)
    
    temp_c = 20.0
    params = RespirationParams()
    
    # Woody plant (tree)
    woody_patch = create_simple_patch(
        temperature_c=temp_c,
        leaf_n=50.0,
        root_n=30.0,
        stem_n=100.0,
        is_woody=True,
    )
    woody_fluxes = calculate_maintenance_respiration(woody_patch, params)
    woody_total = (woody_fluxes.leaf_mr[0] + woody_fluxes.froot_mr[0] + 
                   woody_fluxes.livestem_mr[0] + woody_fluxes.livecroot_mr[0])
    
    # Grass (no woody tissue)
    grass_patch = create_simple_patch(
        temperature_c=temp_c,
        leaf_n=50.0,
        root_n=30.0,
        stem_n=0.0,
        is_woody=False,
    )
    grass_fluxes = calculate_maintenance_respiration(grass_patch, params)
    grass_total = grass_fluxes.leaf_mr[0] + grass_fluxes.froot_mr[0]
    
    print(f"\nWoody Plant (Tree):")
    print(f"  Leaf:        {woody_fluxes.leaf_mr[0] * 86400:.3f} gC/m2/day")
    print(f"  Fine root:   {woody_fluxes.froot_mr[0] * 86400:.3f} gC/m2/day")
    print(f"  Live stem:   {woody_fluxes.livestem_mr[0] * 86400:.3f} gC/m2/day")
    print(f"  Coarse root: {woody_fluxes.livecroot_mr[0] * 86400:.3f} gC/m2/day")
    print(f"  TOTAL:       {woody_total * 86400:.3f} gC/m2/day")
    
    print(f"\nGrass (Non-woody):")
    print(f"  Leaf:        {grass_fluxes.leaf_mr[0] * 86400:.3f} gC/m2/day")
    print(f"  Fine root:   {grass_fluxes.froot_mr[0] * 86400:.3f} gC/m2/day")
    print(f"  TOTAL:       {grass_total * 86400:.3f} gC/m2/day")
    
    print(f"\nWoody tissue adds {(woody_total - grass_total) * 86400:.3f} gC/m2/day")
    print(f"({(woody_total - grass_total) / grass_total * 100:.1f}% increase)")


if __name__ == "__main__":
    # Run all examples
    example_basic_calculation()
    example_temperature_response()
    example_parameter_sensitivity()
    example_woody_vs_grass()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
