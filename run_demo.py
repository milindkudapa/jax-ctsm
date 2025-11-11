#!/usr/bin/env python
"""
Quick demo of JAX-CTSM maintenance respiration.

Run: python run_demo.py
"""

import jax.numpy as jnp
import numpy as np
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


def create_demo_patch():
    """Create a simple patch for demo."""
    n_levgrnd = 10
    n_repr = 2
    
    # Nitrogen pools for a temperate forest
    nitrogen = NitrogenState(
        leafn=jnp.array([55.0]),
        frootn=jnp.array([35.0]),
        livestemn=jnp.array([110.0]),
        livecrootn=jnp.array([90.0]),
        deadstemn=jnp.zeros(1),
        deadcrootn=jnp.zeros(1),
        reproductiven=jnp.zeros((1, n_repr)),
        retransn=jnp.zeros(1),
    )
    
    carbon = CarbonState(
        leafc=jnp.zeros(1),
        frootc=jnp.zeros(1),
        livestemc=jnp.zeros(1),
        livecrootc=jnp.zeros(1),
        deadstemc=jnp.zeros(1),
        deadcrootc=jnp.zeros(1),
        reproductivec=jnp.zeros((1, n_repr)),
        cpool=jnp.zeros(1),
        xsmrpool=jnp.zeros(1),
    )
    
    canopy = CanopyState(
        lai_sun=jnp.array([2.8]),
        lai_shade=jnp.array([1.7]),
        lmr_sun=jnp.array([1.1]),
        lmr_shade=jnp.array([0.75]),
        frac_veg_nosno=jnp.array([1.0]),
    )
    
    temperature = TemperatureState(
        t_ref2m=jnp.array([295.15]),  # 22°C
        t_10day=jnp.array([293.15]),  # 20°C
        t_soisno=jnp.array([[293.15 - i*0.5 for i in range(n_levgrnd)]]),
    )
    
    # Exponential root distribution
    root_fracs = np.exp(-np.arange(n_levgrnd) * 0.3)
    root_fracs = root_fracs / root_fracs.sum()
    
    soil = SoilState(
        crootfr=jnp.array([root_fracs]),
        depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.40, 0.58, 0.80, 1.06, 1.36]),
    )
    
    spatial = SpatialInfo(
        patch_index=jnp.array([0]),
        column_index=jnp.array([0]),
        landunit_index=jnp.array([0]),
        gridcell_index=jnp.array([0]),
        weights=jnp.array([1.0]),
        pft_type=jnp.array([1]),
        is_vegetated=jnp.array([True]),
    )
    
    pft_params = {
        "is_woody": jnp.array([True]),
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


def main():
    print("=" * 70)
    print("JAX-CTSM Maintenance Respiration Demo")
    print("=" * 70)
    
    # Create patch state
    patch = create_demo_patch()
    params = RespirationParams()
    
    print("\nPatch State:")
    print(f"  PFT: Needleleaf Evergreen Tree (woody)")
    print(f"  Temperature: 22°C")
    print(f"  Leaf N: {patch.nitrogen.leafn[0]:.1f} gN/m²")
    print(f"  Root N: {patch.nitrogen.frootn[0]:.1f} gN/m²")
    print(f"  Stem N: {patch.nitrogen.livestemn[0]:.1f} gN/m²")
    print(f"  Coarse Root N: {patch.nitrogen.livecrootn[0]:.1f} gN/m²")
    print(f"  LAI: {patch.canopy.lai_sun[0] + patch.canopy.lai_shade[0]:.1f} m²/m²")
    
    print("\nParameters:")
    print(f"  Base rate (br): {params.br:.3e} gC/gN/s")
    print(f"  Q10: {params.q10}")
    print(f"  Reference temp: {params.reference_temp}°C")
    
    # Calculate MR
    print("\nCalculating maintenance respiration...")
    fluxes = calculate_maintenance_respiration(patch, params)
    
    # Display results
    print("\n" + "=" * 70)
    print("Maintenance Respiration Results")
    print("=" * 70)
    
    print(f"\nFluxes (gC/m²/s):")
    print(f"  Leaf MR:        {fluxes.leaf_mr[0]:.8f}")
    print(f"  Fine Root MR:   {fluxes.froot_mr[0]:.8f}")
    print(f"  Live Stem MR:   {fluxes.livestem_mr[0]:.8f}")
    print(f"  Coarse Root MR: {fluxes.livecroot_mr[0]:.8f}")
    
    total_mr = (fluxes.leaf_mr[0] + fluxes.froot_mr[0] + 
                fluxes.livestem_mr[0] + fluxes.livecroot_mr[0])
    
    print(f"\n  TOTAL MR:       {total_mr:.8f} gC/m²/s")
    
    # Convert to more intuitive units
    print(f"\nDaily and Annual Totals:")
    print(f"  Daily:  {total_mr * 86400:.3f} gC/m²/day")
    print(f"  Annual: {total_mr * 86400 * 365:.1f} gC/m²/year")
    
    # Tissue breakdown
    print(f"\nTissue Breakdown (%):")
    print(f"  Leaf:        {fluxes.leaf_mr[0]/total_mr*100:.1f}%")
    print(f"  Fine Root:   {fluxes.froot_mr[0]/total_mr*100:.1f}%")
    print(f"  Live Stem:   {fluxes.livestem_mr[0]/total_mr*100:.1f}%")
    print(f"  Coarse Root: {fluxes.livecroot_mr[0]/total_mr*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run examples/basic_maintenance_respiration.py for more examples")
    print("  - Run: python -m pytest tests/ -v  (to run tests)")
    print("  - See docs/IMPLEMENTATION_SUMMARY.md for complete documentation")
    

if __name__ == "__main__":
    main()
