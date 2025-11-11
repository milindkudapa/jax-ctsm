"""Tests for the ecosystem driver."""

import jax.numpy as jnp
import pytest

from jax_ctsm.driver import (
    ecosystem_timestep,
    ecosystem_timestep_jit,
    run_simulation,
    run_simulation_scan,
    update_carbon_pools,
    EcosystemParams,
    EcosystemFluxes,
)
from jax_ctsm.core.hierarchy import (
    PatchState,
    NitrogenState,
    CarbonState,
    CanopyState,
    TemperatureState,
    SoilState,
    SpatialInfo,
    CarbonFlux,
)
from jax_ctsm.params.respiration import RespirationParams


def create_test_patch():
    """Create a minimal test patch state."""
    n_patches = 2
    n_levgrnd = 10
    
    nitrogen = NitrogenState(
        leafn=jnp.array([0.02, 0.03]),
        frootn=jnp.array([0.01, 0.015]),
        livestemn=jnp.array([0.005, 0.008]),
        livecrootn=jnp.array([0.005, 0.008]),
        deadstemn=jnp.zeros(n_patches),
        deadcrootn=jnp.zeros(n_patches),
        reproductiven=jnp.zeros((n_patches, 3)),
        retransn=jnp.zeros(n_patches),
    )
    
    carbon = CarbonState(
        leafc=jnp.array([0.5, 0.6]),
        frootc=jnp.array([0.3, 0.4]),
        livestemc=jnp.array([1.0, 1.5]),
        livecrootc=jnp.array([0.8, 1.2]),
        deadstemc=jnp.zeros(n_patches),
        deadcrootc=jnp.zeros(n_patches),
        reproductivec=jnp.zeros((n_patches, 3)),
        cpool=jnp.array([10.0, 12.0]),
        xsmrpool=jnp.zeros(n_patches),
    )
    
    canopy = CanopyState(
        lai_sun=jnp.array([2.0, 2.5]),
        lai_shade=jnp.array([1.0, 1.2]),
        lmr_sun=jnp.array([2e-6, 2.5e-6]),
        lmr_shade=jnp.array([1e-6, 1.2e-6]),
        frac_veg_nosno=jnp.ones(n_patches, dtype=jnp.int32),
    )
    
    temp = TemperatureState(
        t_ref2m=jnp.full(n_patches, 298.15),
        t_10day=jnp.full(n_patches, 295.15),
        t_soisno=jnp.full((n_patches, n_levgrnd), 295.15),  # Note: Using n_patches here as proxy for n_columns (1:1 mapping in this test)
    )
    
    soil = SoilState(
        crootfr=jnp.tile(jnp.array([0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01, 0.0]), (n_patches, 1)),
        depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.4, 0.58, 0.8, 1.06, 1.36]),
    )
    
    spatial = SpatialInfo(
        pft_type=jnp.zeros(n_patches, dtype=jnp.int32),
        patch_index=jnp.arange(n_patches, dtype=jnp.int32),
        column_index=jnp.arange(n_patches, dtype=jnp.int32),
        landunit_index=jnp.zeros(n_patches, dtype=jnp.int32),
        gridcell_index=jnp.zeros(n_patches, dtype=jnp.int32),
        weights=jnp.ones(n_patches),
        is_vegetated=jnp.ones(n_patches, dtype=bool),
    )
    
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


def test_update_carbon_pools():
    """Test carbon pool updates."""
    carbon_state = CarbonState(
        leafc=jnp.array([0.5, 0.6]),
        frootc=jnp.array([0.3, 0.4]),
        livestemc=jnp.array([1.0, 1.5]),
        livecrootc=jnp.array([0.8, 1.2]),
        deadstemc=jnp.zeros(2),
        deadcrootc=jnp.zeros(2),
        reproductivec=jnp.zeros((2, 3)),
        cpool=jnp.array([10.0, 12.0]),
        xsmrpool=jnp.zeros(2),
    )
    
    fluxes = CarbonFlux(
        gpp=jnp.zeros(2),
        leaf_mr=jnp.array([1e-6, 1.5e-6]),
        froot_mr=jnp.array([0.5e-6, 0.8e-6]),
        livestem_mr=jnp.array([0.3e-6, 0.4e-6]),
        livecroot_mr=jnp.array([0.2e-6, 0.3e-6]),
        reproductive_mr=jnp.zeros((2, 3)),
    )
    
    dt = 1800.0  # 30 minutes
    
    new_carbon = update_carbon_pools(carbon_state, fluxes, dt)
    
    # Check that cpool decreased
    assert jnp.all(new_carbon.cpool < carbon_state.cpool)
    
    # Check conservation (approximately)
    total_mr = (fluxes.leaf_mr + fluxes.froot_mr + 
                fluxes.livestem_mr + fluxes.livecroot_mr)
    expected_loss = total_mr * dt
    actual_loss = carbon_state.cpool - new_carbon.cpool
    
    # Use absolute tolerance for float32 precision
    assert jnp.allclose(actual_loss, expected_loss, atol=1e-6)


def test_ecosystem_timestep():
    """Test single timestep execution."""
    patch_state = create_test_patch()
    params = EcosystemParams(respiration=RespirationParams())
    
    new_state, fluxes = ecosystem_timestep(patch_state, params, dt=1800.0)
    
    # Check that we got valid outputs
    assert isinstance(new_state, PatchState)
    assert isinstance(fluxes, EcosystemFluxes)
    
    # Check that carbon pools changed
    assert not jnp.array_equal(new_state.carbon.cpool, patch_state.carbon.cpool)
    
    # Check that fluxes are positive (respiration)
    assert jnp.all(fluxes.total_respiration >= 0)
    assert jnp.all(fluxes.carbon.leaf_mr >= 0)
    assert jnp.all(fluxes.carbon.froot_mr >= 0)


def test_ecosystem_timestep_jit():
    """Test JIT-compiled timestep."""
    patch_state = create_test_patch()
    params = EcosystemParams(respiration=RespirationParams())
    
    # JIT version should give same results
    new_state_jit, fluxes_jit = ecosystem_timestep_jit(patch_state, params, dt=1800.0)
    new_state_normal, fluxes_normal = ecosystem_timestep(patch_state, params, dt=1800.0)
    
    # Results should match
    assert jnp.allclose(new_state_jit.carbon.cpool, new_state_normal.carbon.cpool)
    assert jnp.allclose(fluxes_jit.total_respiration, fluxes_normal.total_respiration)


def test_run_simulation():
    """Test multi-timestep simulation."""
    patch_state = create_test_patch()
    params = EcosystemParams(respiration=RespirationParams())
    
    n_timesteps = 10
    final_state, flux_history = run_simulation(
        patch_state, params, n_timesteps=n_timesteps, dt=1800.0
    )
    
    # Check that we got the right number of timesteps
    assert len(flux_history) == n_timesteps
    
    # Check that carbon pools decreased monotonically
    initial_cpool = patch_state.carbon.cpool
    final_cpool = final_state.carbon.cpool
    assert jnp.all(final_cpool < initial_cpool)
    
    # Check that total C loss matches sum of fluxes
    total_mr = sum([f.total_respiration for f in flux_history])
    expected_loss = total_mr * 1800.0
    actual_loss = initial_cpool - final_cpool
    # Use loose tolerance for accumulated float32 rounding errors
    assert jnp.allclose(actual_loss, expected_loss, atol=2e-6)


def test_run_simulation_scan():
    """Test scan-based simulation."""
    patch_state = create_test_patch()
    params = EcosystemParams(respiration=RespirationParams())
    
    n_timesteps = 10
    final_scan, flux_scan = run_simulation_scan(
        patch_state, params, n_timesteps=n_timesteps, dt=1800.0
    )
    final_loop, flux_loop = run_simulation(
        patch_state, params, n_timesteps=n_timesteps, dt=1800.0
    )
    
    # Results should match loop version
    assert jnp.allclose(final_scan.carbon.cpool, final_loop.carbon.cpool)
    
    # Check flux history shapes match
    assert flux_scan.total_respiration.shape[0] == n_timesteps


def test_state_immutability():
    """Test that original state is not modified."""
    patch_state = create_test_patch()
    params = EcosystemParams(respiration=RespirationParams())
    
    # Store original values
    original_cpool = patch_state.carbon.cpool.copy()
    
    # Run timestep
    new_state, _ = ecosystem_timestep(patch_state, params, dt=1800.0)
    
    # Original should be unchanged
    assert jnp.array_equal(patch_state.carbon.cpool, original_cpool)
    
    # New state should be different
    assert not jnp.array_equal(new_state.carbon.cpool, original_cpool)


def test_carbon_conservation():
    """Test that carbon is conserved."""
    patch_state = create_test_patch()
    params = EcosystemParams(respiration=RespirationParams())
    
    # Run simulation
    n_timesteps = 5
    dt = 1800.0
    final_state, flux_history = run_simulation(
        patch_state, params, n_timesteps=n_timesteps, dt=dt
    )
    
    # Calculate total C lost to respiration
    total_mr_flux = sum([f.total_respiration for f in flux_history]) * dt
    
    # Calculate change in C pools
    initial_total = patch_state.carbon.cpool + patch_state.carbon.xsmrpool
    final_total = final_state.carbon.cpool + final_state.carbon.xsmrpool
    pool_change = initial_total - final_total
    
    # Should match (conservation)
    assert jnp.allclose(total_mr_flux, pool_change, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
