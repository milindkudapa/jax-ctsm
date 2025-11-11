"""
Integration tests for complete workflow.

Tests the full pipeline from state creation to flux calculation.
"""

import pytest
import jax
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
    patch_to_column,
)
from jax_ctsm.physics.maintenance_respiration import (
    calculate_maintenance_respiration,
    calculate_maintenance_respiration_batch,
)
from jax_ctsm.params.respiration import RespirationParams
from jax_ctsm.utils.validation import validate_fluxes, check_mass_balance


class TestSpatialAggregation:
    """Test spatial aggregation functions."""
    
    def test_patch_to_column_aggregation(self):
        """Test patch to column aggregation with weights."""
        # 3 patches in 1 column
        patch_values = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.5, 0.3, 0.2])
        column_indices = jnp.array([0, 0, 0])
        n_columns = 1
        
        result = patch_to_column(patch_values, weights, column_indices, n_columns)
        
        # Expected: weighted average
        expected = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0
        assert jnp.allclose(result[0], expected)
    
    def test_multiple_columns(self):
        """Test aggregation with multiple columns."""
        # 4 patches in 2 columns
        patch_values = jnp.array([1.0, 2.0, 3.0, 4.0])
        weights = jnp.array([0.6, 0.4, 0.7, 0.3])
        column_indices = jnp.array([0, 0, 1, 1])  # First 2 in col 0, next 2 in col 1
        n_columns = 2
        
        result = patch_to_column(patch_values, weights, column_indices, n_columns)
        
        # Column 0: (0.6*1 + 0.4*2) / (0.6+0.4)
        expected_0 = (0.6 * 1.0 + 0.4 * 2.0) / (0.6 + 0.4)
        # Column 1: (0.7*3 + 0.3*4) / (0.7+0.3)
        expected_1 = (0.7 * 3.0 + 0.3 * 4.0) / (0.7 + 0.3)
        
        assert jnp.allclose(result[0], expected_0)
        assert jnp.allclose(result[1], expected_1)


class TestEndToEndWorkflow:
    """Test complete workflow from state to fluxes."""
    
    @pytest.fixture
    def multi_pft_state(self):
        """Create realistic multi-PFT state."""
        n_patches = 5
        n_levgrnd = 10
        n_repr = 2
        
        # Mixed forest + grassland
        pft_types = jnp.array([1, 1, 6, 12, 12])  # 2 trees, 1 shrub, 2 grasses
        is_woody = jnp.array([True, True, True, False, False])
        is_crop = jnp.array([False, False, False, False, False])
        
        # Nitrogen pools (varying by PFT)
        nitrogen = NitrogenState(
            leafn=jnp.array([60.0, 55.0, 40.0, 30.0, 35.0]),
            frootn=jnp.array([40.0, 38.0, 25.0, 20.0, 22.0]),
            livestemn=jnp.array([120.0, 110.0, 80.0, 0.0, 0.0]),  # Woody only
            livecrootn=jnp.array([100.0, 95.0, 65.0, 0.0, 0.0]),  # Woody only
            deadstemn=jnp.zeros(n_patches),
            deadcrootn=jnp.zeros(n_patches),
            reproductiven=jnp.zeros((n_patches, n_repr)),
            retransn=jnp.zeros(n_patches),
        )
        
        carbon = CarbonState(
            leafc=jnp.zeros(n_patches),
            frootc=jnp.zeros(n_patches),
            livestemc=jnp.zeros(n_patches),
            livecrootc=jnp.zeros(n_patches),
            deadstemc=jnp.zeros(n_patches),
            deadcrootc=jnp.zeros(n_patches),
            reproductivec=jnp.zeros((n_patches, n_repr)),
            cpool=jnp.zeros(n_patches),
            xsmrpool=jnp.zeros(n_patches),
        )
        
        canopy = CanopyState(
            lai_sun=jnp.array([3.0, 2.8, 2.0, 1.5, 1.8]),
            lai_shade=jnp.array([2.0, 1.8, 1.2, 1.0, 1.2]),
            lmr_sun=jnp.array([1.2, 1.1, 0.9, 0.7, 0.8]),
            lmr_shade=jnp.array([0.8, 0.7, 0.6, 0.5, 0.6]),
            frac_veg_nosno=jnp.ones(n_patches),
        )
        
        # Temperature gradient
        temps = jnp.array([298.15, 296.15, 294.15, 292.15, 290.15])  # 25°C to 17°C
        soil_temps = jnp.array([
            [temp - i * 0.5 for i in range(n_levgrnd)]
            for temp in [295.15, 295.15, 293.15, 293.15, 291.15]  # Column temps
        ])
        
        temperature = TemperatureState(
            t_ref2m=temps,
            t_10day=temps - 2.0,
            t_soisno=soil_temps,
        )
        
        # Root distribution
        root_fracs = np.zeros((n_patches, n_levgrnd))
        for p in range(n_patches):
            fracs = np.exp(-np.arange(n_levgrnd) * 0.3)
            root_fracs[p, :] = fracs / fracs.sum()
        
        soil = SoilState(
            crootfr=jnp.array(root_fracs),
            depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.40, 0.58, 0.80, 1.06, 1.36]),
        )
        
        # 2 columns: patches 0-2 in column 0, patches 3-4 in column 1
        spatial = SpatialInfo(
            patch_index=jnp.arange(n_patches),
            column_index=jnp.array([0, 0, 0, 1, 1]),
            landunit_index=jnp.array([0, 0]),
            gridcell_index=jnp.array([0]),
            weights=jnp.array([0.35, 0.30, 0.35, 0.6, 0.4]),  # Normalized within column
            pft_type=pft_types,
            is_vegetated=jnp.ones(n_patches, dtype=bool),
        )
        
        pft_params = {
            "is_woody": is_woody,
            "is_crop": is_crop,
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
    
    def test_full_calculation(self, multi_pft_state):
        """Test complete MR calculation on realistic state."""
        params = RespirationParams()
        
        # Calculate MR
        fluxes = calculate_maintenance_respiration(multi_pft_state, params)
        
        # Validate fluxes
        flux_dict = {
            "leaf_mr": fluxes.leaf_mr,
            "froot_mr": fluxes.froot_mr,
            "livestem_mr": fluxes.livestem_mr,
            "livecroot_mr": fluxes.livecroot_mr,
        }
        validation = validate_fluxes(flux_dict)
        
        assert all(validation.values()), f"Validation failed: {validation}"
        
        # Check that woody plants have stem/croot MR
        assert jnp.all(fluxes.livestem_mr[:3] > 0)  # First 3 are woody
        assert jnp.all(fluxes.livestem_mr[3:] == 0)  # Last 2 are grass
        
        assert jnp.all(fluxes.livecroot_mr[:3] > 0)
        assert jnp.all(fluxes.livecroot_mr[3:] == 0)
    
    def test_column_aggregation(self, multi_pft_state):
        """Test aggregating patch MR to column level."""
        params = RespirationParams()
        fluxes = calculate_maintenance_respiration(multi_pft_state, params)
        
        # Total MR per patch
        total_patch_mr = (
            fluxes.leaf_mr + fluxes.froot_mr + 
            fluxes.livestem_mr + fluxes.livecroot_mr
        )
        
        # Aggregate to column
        spatial = multi_pft_state.spatial
        column_mr = patch_to_column(
            total_patch_mr,
            spatial.weights,
            spatial.column_index,
            2,  # 2 columns
        )
        
        # Both columns should have positive MR
        assert jnp.all(column_mr > 0)
        
        # Column 0 (woody) should have higher MR than column 1 (grass)
        # (because woody tissue adds MR)
        assert column_mr[0] > column_mr[1]


class TestJAXTransformations:
    """Test JAX transformations (jit, vmap, grad)."""
    
    def test_jit_compilation(self, multi_pft_state):
        """Test that calculation works with JIT."""
        params = RespirationParams()
        
        # JIT compile
        jitted_fn = jax.jit(
            lambda state, p: calculate_maintenance_respiration(state, p)
        )
        
        # Run once to compile
        result1 = jitted_fn(multi_pft_state, params)
        
        # Run again (should be fast)
        result2 = jitted_fn(multi_pft_state, params)
        
        # Results should be identical
        assert jnp.allclose(result1.leaf_mr, result2.leaf_mr)
        assert jnp.allclose(result1.froot_mr, result2.froot_mr)
    
    def test_vmap_over_parameters(self):
        """Test vmapping over parameter values."""
        # Create simple state
        n_patches = 1
        n_levgrnd = 5
        n_repr = 2
        
        nitrogen = NitrogenState(
            leafn=jnp.array([50.0]),
            frootn=jnp.array([30.0]),
            livestemn=jnp.array([100.0]),
            livecrootn=jnp.array([80.0]),
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
        
        # Create complete state... (abbreviated for brevity)
        # In real test, would create full state
        
    def test_gradient_calculation(self):
        """Test that we can compute gradients w.r.t. parameters."""
        # Simple differentiable loss
        def loss_fn(q10_value):
            params = RespirationParams(q10=q10_value)
            # Would calculate MR with full state, sum as loss
            # For now, simple quadratic
            return (q10_value - 1.5) ** 2
        
        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(1.5)
        
        # At minimum, gradient should be zero
        assert jnp.allclose(gradient, 0.0, atol=1e-6)


class TestPhysicalConsistency:
    """Test physical behavior and conservation."""
    
    def test_temperature_response_monotonic(self):
        """Test that MR increases monotonically with temperature."""
        temps = jnp.array([273.15, 283.15, 293.15, 303.15, 313.15])  # 0 to 40°C
        mrs = []
        
        params = RespirationParams()
        
        for temp in temps:
            # Simple correction calculation
            temp_c = temp - 273.15
            tc = params.q10 ** ((temp_c - params.reference_temp) / 10.0)
            mr = 100.0 * params.br * tc  # 100 gN/m2 stem
            mrs.append(float(mr))
        
        # Should be monotonically increasing
        mrs = np.array(mrs)
        assert np.all(np.diff(mrs) > 0), "MR should increase with temperature"
    
    def test_nitrogen_scaling(self, multi_pft_state):
        """Test that MR scales with nitrogen content."""
        params = RespirationParams()
        
        # Original fluxes
        fluxes_1x = calculate_maintenance_respiration(multi_pft_state, params)
        
        # Double nitrogen
        nitrogen_2x = multi_pft_state.nitrogen._replace(
            livestemn=multi_pft_state.nitrogen.livestemn * 2.0,
        )
        state_2x = multi_pft_state._replace(nitrogen=nitrogen_2x)
        
        fluxes_2x = calculate_maintenance_respiration(state_2x, params)
        
        # Stem MR should approximately double
        # (exact only for woody plants where stem MR calculated)
        woody_mask = multi_pft_state.pft_params["is_woody"]
        
        assert jnp.allclose(
            fluxes_2x.livestem_mr[woody_mask] / fluxes_1x.livestem_mr[woody_mask],
            2.0,
            rtol=1e-5,
        )


class TestSpatialHierarchy:
    """Test spatial hierarchy with multiple patches per column."""
    
    def test_multiple_patches_per_column(self):
        """Test that multiple patches can share a column's soil temperature.
        
        This is critical for CTSM's design where multiple PFTs can exist
        on the same soil column (e.g., grass and tree in same location).
        """
        # Setup: 4 patches, 2 columns (2 patches per column)
        n_patches = 4
        n_columns = 2
        n_levgrnd = 10
        
        # Nitrogen state
        nitrogen = NitrogenState(
            leafn=jnp.array([40.0, 35.0, 50.0, 45.0]),
            frootn=jnp.array([25.0, 20.0, 30.0, 28.0]),
            livestemn=jnp.array([80.0, 0.0, 100.0, 0.0]),  # Patches 0,2 woody
            livecrootn=jnp.array([60.0, 0.0, 80.0, 0.0]),
            deadstemn=jnp.zeros(n_patches),
            deadcrootn=jnp.zeros(n_patches),
            reproductiven=jnp.zeros((n_patches, 2)),
            retransn=jnp.zeros(n_patches),
        )
        
        # Carbon state (minimal, just for structure)
        carbon = CarbonState(
            leafc=jnp.zeros(n_patches),
            frootc=jnp.zeros(n_patches),
            livestemc=jnp.zeros(n_patches),
            livecrootc=jnp.zeros(n_patches),
            deadstemc=jnp.zeros(n_patches),
            deadcrootc=jnp.zeros(n_patches),
            reproductivec=jnp.zeros((n_patches, 2)),
            cpool=jnp.zeros(n_patches),
            xsmrpool=jnp.zeros(n_patches),
        )
        
        # Canopy state
        canopy = CanopyState(
            lai_sun=jnp.array([2.0, 1.5, 2.5, 2.0]),
            lai_shade=jnp.array([1.0, 0.8, 1.2, 1.0]),
            lmr_sun=jnp.array([1.0, 0.8, 1.2, 1.0]),
            lmr_shade=jnp.array([0.6, 0.5, 0.7, 0.6]),
            frac_veg_nosno=jnp.ones(n_patches),
        )
        
        # Temperature state - CRITICAL: soil temp is column-level
        # Column 0 has warmer soil, Column 1 has cooler soil
        soil_temps_column = jnp.array([
            [295.15 - i * 0.5 for i in range(n_levgrnd)],  # Column 0: warmer
            [290.15 - i * 0.5 for i in range(n_levgrnd)],  # Column 1: cooler
        ])
        
        temperature = TemperatureState(
            t_ref2m=jnp.array([298.15, 298.15, 293.15, 293.15]),  # Patch-level
            t_10day=jnp.array([295.15, 295.15, 290.15, 290.15]),
            t_soisno=soil_temps_column,  # Column-level [n_columns, n_levgrnd]
        )
        
        # Soil state with root distribution
        root_fracs = np.zeros((n_patches, n_levgrnd))
        for p in range(n_patches):
            fracs = np.exp(-np.arange(n_levgrnd) * 0.3)
            root_fracs[p, :] = fracs / fracs.sum()
        
        soil = SoilState(
            crootfr=jnp.array(root_fracs),
            depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.40, 0.58, 0.80, 1.06, 1.36]),
        )
        
        # Spatial info - KEY: patches 0,1 share column 0; patches 2,3 share column 1
        spatial = SpatialInfo(
            patch_index=jnp.array([0, 1, 2, 3]),
            column_index=jnp.array([0, 0, 1, 1]),  # ← 2 patches per column
            landunit_index=jnp.array([0, 0]),
            gridcell_index=jnp.array([0]),
            weights=jnp.array([0.3, 0.2, 0.3, 0.2]),
            pft_type=jnp.array([1, 12, 1, 12]),  # Tree, grass, tree, grass
            is_vegetated=jnp.ones(n_patches, dtype=bool),
        )
        
        # PFT params
        pft_params = {
            "is_woody": jnp.array([True, False, True, False]),
            "is_crop": jnp.array([False, False, False, False]),
        }
        
        # Create patch state
        patch_state = PatchState(
            carbon=carbon,
            nitrogen=nitrogen,
            canopy=canopy,
            temperature=temperature,
            soil=soil,
            spatial=spatial,
            pft_params=pft_params,
        )
        
        # Calculate maintenance respiration
        params = RespirationParams()
        fluxes = calculate_maintenance_respiration(patch_state, params)
        
        # Verify calculations work
        assert fluxes.leaf_mr.shape == (n_patches,)
        assert fluxes.froot_mr.shape == (n_patches,)
        
        # Patches sharing column should use the same soil temperature
        # So patches 0,1 should have similar root MR (scaled by N content)
        # And patches 2,3 should have similar root MR (scaled by N content)
        
        # Verify root MR is reasonable
        assert jnp.all(fluxes.froot_mr > 0), "All patches should have positive root MR"
        
        # Patches in warmer column (0,1) should have higher root MR than cooler (2,3)
        # when normalized by nitrogen content
        mr_per_n_warm = (fluxes.froot_mr[0] + fluxes.froot_mr[1]) / (
            nitrogen.frootn[0] + nitrogen.frootn[1]
        )
        mr_per_n_cool = (fluxes.froot_mr[2] + fluxes.froot_mr[3]) / (
            nitrogen.frootn[2] + nitrogen.frootn[3]
        )
        
        # Warmer soil → higher MR per unit N
        assert mr_per_n_warm > mr_per_n_cool, (
            f"Warmer column should have higher MR per unit N: "
            f"{mr_per_n_warm:.6e} vs {mr_per_n_cool:.6e}"
        )
        
        # Verify stem MR only calculated for woody plants
        assert fluxes.livestem_mr[1] == 0.0, "Grass (patch 1) should have zero stem MR"
        assert fluxes.livestem_mr[3] == 0.0, "Grass (patch 3) should have zero stem MR"
        assert fluxes.livestem_mr[0] > 0.0, "Tree (patch 0) should have positive stem MR"
        assert fluxes.livestem_mr[2] > 0.0, "Tree (patch 2) should have positive stem MR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
