"""
Tests for maintenance respiration calculations.

Tests against known values from CTSM Fortran implementation and
validates physical behavior.
"""

import pytest
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
from jax_ctsm.physics.maintenance_respiration import (
    calculate_maintenance_respiration,
    leaf_maintenance_respiration,
    root_maintenance_respiration,
    stem_maintenance_respiration,
)
from jax_ctsm.params.respiration import RespirationParams, RYAN_1991_PARAMS


@pytest.fixture
def default_params():
    """Default respiration parameters."""
    return RespirationParams()


@pytest.fixture
def sample_patch_state():
    """Create a sample patch state for testing."""
    n_patches = 3
    n_levgrnd = 10
    n_repr = 2
    
    # Create nitrogen state
    nitrogen = NitrogenState(
        leafn=jnp.array([50.0, 30.0, 40.0]),
        frootn=jnp.array([30.0, 20.0, 25.0]),
        livestemn=jnp.array([100.0, 0.0, 80.0]),  # Woody only
        livecrootn=jnp.array([80.0, 0.0, 60.0]),  # Woody only
        deadstemn=jnp.zeros(n_patches),
        deadcrootn=jnp.zeros(n_patches),
        reproductiven=jnp.array([[0.0, 0.0], [15.0, 10.0], [0.0, 0.0]]),  # Crop only
        retransn=jnp.zeros(n_patches),
    )
    
    # Create carbon state (placeholder)
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
    
    # Create canopy state
    canopy = CanopyState(
        lai_sun=jnp.array([2.5, 1.5, 2.0]),
        lai_shade=jnp.array([1.5, 1.0, 1.2]),
        lmr_sun=jnp.array([1.2, 0.8, 1.0]),  # umol CO2/m2 leaf/s
        lmr_shade=jnp.array([0.8, 0.5, 0.6]),
        frac_veg_nosno=jnp.array([1.0, 1.0, 0.0]),  # Last patch snow covered
    )
    
    # Create temperature state
    # Soil temp: 15°C at surface, decreasing with depth
    soil_temps = jnp.array([
        [288.15 - i * 0.5 for i in range(n_levgrnd)],  # Column 0
        [288.15 - i * 0.5 for i in range(n_levgrnd)],  # Column 1
        [288.15 - i * 0.5 for i in range(n_levgrnd)],  # Column 2
    ])
    
    temperature = TemperatureState(
        t_ref2m=jnp.array([298.15, 293.15, 288.15]),  # 25°C, 20°C, 15°C
        t_10day=jnp.array([296.15, 291.15, 286.15]),  # 10-day means
        t_soisno=soil_temps,
    )
    
    # Create soil state
    # Root fraction decreasing with depth (sum to 1.0)
    root_fracs = np.zeros((n_patches, n_levgrnd))
    for p in range(n_patches):
        fracs = np.exp(-np.arange(n_levgrnd) * 0.3)
        root_fracs[p, :] = fracs / fracs.sum()
    
    soil = SoilState(
        crootfr=jnp.array(root_fracs),
        depth=jnp.array([0.01, 0.04, 0.09, 0.16, 0.26, 0.40, 0.58, 0.80, 1.06, 1.36]),
    )
    
    # Create spatial info
    spatial = SpatialInfo(
        patch_index=jnp.array([0, 1, 2]),
        column_index=jnp.array([0, 1, 2]),  # 1:1 mapping
        landunit_index=jnp.array([0, 0, 0]),
        gridcell_index=jnp.array([0]),
        weights=jnp.array([0.4, 0.3, 0.3]),
        pft_type=jnp.array([1, 15, 1]),  # PFT types
        is_vegetated=jnp.array([True, True, True]),
    )
    
    # PFT parameters
    pft_params = {
        "is_woody": jnp.array([True, False, True]),
        "is_crop": jnp.array([False, True, False]),
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


class TestLeafMaintenanceRespiration:
    """Test leaf MR calculations."""
    
    def test_basic_calculation(self, default_params):
        """Test basic leaf MR calculation."""
        lai_sun = jnp.array([2.5])
        lai_shade = jnp.array([1.5])
        lmr_sun = jnp.array([1.2])  # umol/m2 leaf/s
        lmr_shade = jnp.array([0.8])
        frac_veg_nosno = jnp.array([1.0])
        
        leaf_mr = leaf_maintenance_respiration(
            lai_sun, lai_shade, lmr_sun, lmr_shade, frac_veg_nosno, default_params
        )
        
        # Expected: (1.2 * 2.5 + 0.8 * 1.5) * 12.011e-6
        expected = (1.2 * 2.5 + 0.8 * 1.5) * 12.011e-6
        assert jnp.allclose(leaf_mr[0], expected, rtol=1e-6)
    
    def test_snow_cover(self, default_params):
        """Test that leaf MR is zero when snow covered."""
        lai_sun = jnp.array([2.5])
        lai_shade = jnp.array([1.5])
        lmr_sun = jnp.array([1.2])
        lmr_shade = jnp.array([0.8])
        frac_veg_nosno = jnp.array([0.0])  # Snow covered
        
        leaf_mr = leaf_maintenance_respiration(
            lai_sun, lai_shade, lmr_sun, lmr_shade, frac_veg_nosno, default_params
        )
        
        assert leaf_mr[0] == 0.0


class TestStemMaintenanceRespiration:
    """Test stem MR calculations."""
    
    def test_woody_stem_mr(self, default_params):
        """Test stem MR for woody plants."""
        stem_n = jnp.array([100.0])  # gN/m2
        temperature = jnp.array([293.15])  # 20°C (reference temp)
        t_10day = jnp.array([293.15])
        is_woody = jnp.array([True])
        
        stem_mr = stem_maintenance_respiration(
            stem_n, temperature, t_10day, is_woody, default_params
        )
        
        # At reference temp (20°C), Q10 correction = 1.0
        # MR = stem_n * br * 1.0
        expected = 100.0 * default_params.br * 1.0
        assert jnp.allclose(stem_mr[0], expected, rtol=1e-6)
    
    def test_non_woody_zero_mr(self, default_params):
        """Test that non-woody plants have zero stem MR."""
        stem_n = jnp.array([100.0])
        temperature = jnp.array([293.15])
        t_10day = jnp.array([293.15])
        is_woody = jnp.array([False])
        
        stem_mr = stem_maintenance_respiration(
            stem_n, temperature, t_10day, is_woody, default_params
        )
        
        assert stem_mr[0] == 0.0
    
    def test_temperature_sensitivity(self, default_params):
        """Test Q10 temperature response."""
        stem_n = jnp.array([100.0])
        is_woody = jnp.array([True])
        t_10day = jnp.array([293.15])
        
        # At 20°C (reference)
        mr_20c = stem_maintenance_respiration(
            stem_n, jnp.array([293.15]), t_10day, is_woody, default_params
        )
        
        # At 30°C (10° higher)
        mr_30c = stem_maintenance_respiration(
            stem_n, jnp.array([303.15]), t_10day, is_woody, default_params
        )
        
        # Should be Q10 times higher (Q10=1.5 -> 1.5x)
        assert jnp.allclose(mr_30c[0] / mr_20c[0], default_params.q10, rtol=1e-6)


class TestRootMaintenanceRespiration:
    """Test fine root MR with depth distribution."""
    
    def test_single_layer(self, default_params):
        """Test with all roots in one layer."""
        root_n = jnp.array([30.0])
        root_frac = jnp.array([[1.0, 0.0, 0.0]])  # All in top layer
        soil_temp = jnp.array([[293.15, 288.15, 283.15]])  # 20°C in top
        column_indices = jnp.array([0])
        t_10day = jnp.array([293.15])
        
        root_mr = root_maintenance_respiration(
            root_n, root_frac, soil_temp, column_indices, t_10day, default_params
        )
        
        # At 20°C, Q10 correction = 1.0
        expected = 30.0 * default_params.br_root * 1.0
        assert jnp.allclose(root_mr[0], expected, rtol=1e-6)
    
    def test_depth_distribution(self, default_params):
        """Test with roots distributed across layers."""
        root_n = jnp.array([30.0])
        root_frac = jnp.array([[0.5, 0.3, 0.2]])  # Distributed
        # Varying temps: 30°C, 20°C, 10°C
        soil_temp = jnp.array([[303.15, 293.15, 283.15]])
        column_indices = jnp.array([0])
        t_10day = jnp.array([293.15])
        
        root_mr = root_maintenance_respiration(
            root_n, root_frac, soil_temp, column_indices, t_10day, default_params
        )
        
        # Calculate expected weighted sum
        br = default_params.br_root
        q10 = default_params.q10
        # Layer 0: 30°C -> Q10^1.0
        # Layer 1: 20°C -> Q10^0.0 = 1.0
        # Layer 2: 10°C -> Q10^-1.0
        tc0 = q10 ** ((30.0 - 20.0) / 10.0)
        tc1 = q10 ** ((20.0 - 20.0) / 10.0)
        tc2 = q10 ** ((10.0 - 20.0) / 10.0)
        
        expected = 30.0 * br * (0.5 * tc0 + 0.3 * tc1 + 0.2 * tc2)
        assert jnp.allclose(root_mr[0], expected, rtol=1e-5)


class TestIntegratedMaintenanceRespiration:
    """Test full integrated MR calculation."""
    
    def test_full_calculation(self, sample_patch_state, default_params):
        """Test complete MR calculation with realistic state."""
        fluxes = calculate_maintenance_respiration(sample_patch_state, default_params)
        
        # Check dimensions
        assert fluxes.leaf_mr.shape == (3,)
        assert fluxes.froot_mr.shape == (3,)
        assert fluxes.livestem_mr.shape == (3,)
        assert fluxes.livecroot_mr.shape == (3,)
        assert fluxes.reproductive_mr.shape == (3, 2)
        
        # Check that all fluxes are non-negative
        assert jnp.all(fluxes.leaf_mr >= 0)
        assert jnp.all(fluxes.froot_mr >= 0)
        assert jnp.all(fluxes.livestem_mr >= 0)
        assert jnp.all(fluxes.livecroot_mr >= 0)
        assert jnp.all(fluxes.reproductive_mr >= 0)
        
        # Check woody plant has stem/croot MR
        assert fluxes.livestem_mr[0] > 0  # Patch 0 is woody
        assert fluxes.livecroot_mr[0] > 0
        assert fluxes.livestem_mr[1] == 0  # Patch 1 is crop
        assert fluxes.livecroot_mr[1] == 0
        
        # Check crop has reproductive MR
        assert jnp.sum(fluxes.reproductive_mr[1, :]) > 0  # Patch 1 is crop
        assert jnp.sum(fluxes.reproductive_mr[0, :]) == 0  # Patch 0 is woody
    
    def test_snow_cover_effect(self, sample_patch_state, default_params):
        """Test that snow cover zeros out leaf MR."""
        fluxes = calculate_maintenance_respiration(sample_patch_state, default_params)
        
        # Patch 2 is snow covered (frac_veg_nosno=0)
        assert fluxes.leaf_mr[2] == 0.0
        
        # Other patches should have non-zero leaf MR
        assert fluxes.leaf_mr[0] > 0
        assert fluxes.leaf_mr[1] > 0
    
    def test_mass_balance(self, sample_patch_state, default_params):
        """Test that MR scales with nitrogen pools."""
        fluxes = calculate_maintenance_respiration(sample_patch_state, default_params)
        
        # Total MR should increase if we double nitrogen
        nitrogen_doubled = sample_patch_state.nitrogen._replace(
            frootn=sample_patch_state.nitrogen.frootn * 2,
            livestemn=sample_patch_state.nitrogen.livestemn * 2,
        )
        state_doubled = sample_patch_state._replace(nitrogen=nitrogen_doubled)
        fluxes_doubled = calculate_maintenance_respiration(state_doubled, default_params)
        
        # Froot and stem MR should approximately double
        # (not exact due to LAI-based leaf MR)
        assert jnp.allclose(
            fluxes_doubled.froot_mr / fluxes.froot_mr,
            2.0,
            rtol=1e-5,
            atol=1e-10
        )


class TestParameterSensitivity:
    """Test sensitivity to parameter changes."""
    
    def test_q10_sensitivity(self, sample_patch_state):
        """Test that changing Q10 affects temperature response."""
        # High Q10
        params_high = RespirationParams(q10=2.0)
        fluxes_high = calculate_maintenance_respiration(sample_patch_state, params_high)
        
        # Low Q10
        params_low = RespirationParams(q10=1.2)
        fluxes_low = calculate_maintenance_respiration(sample_patch_state, params_low)
        
        # At temperatures above reference, high Q10 should give higher MR
        # Patch 0 is at 25°C (above 20°C reference)
        assert fluxes_high.livestem_mr[0] > fluxes_low.livestem_mr[0]
    
    def test_base_rate_scaling(self, sample_patch_state):
        """Test that MR scales linearly with base rate."""
        params_2x = RespirationParams(br=2 * RespirationParams().br)
        
        fluxes_1x = calculate_maintenance_respiration(sample_patch_state, RespirationParams())
        fluxes_2x = calculate_maintenance_respiration(sample_patch_state, params_2x)
        
        # Stem MR should double (woody patch)
        assert jnp.allclose(
            fluxes_2x.livestem_mr[0],
            fluxes_1x.livestem_mr[0] * 2,
            rtol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
