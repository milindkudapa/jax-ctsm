"""
Parameters for maintenance respiration calculations.

Based on CTSM's CNMRespMod.F90
Reference: M. Ryan, 1991. Effects of climate change on plant respiration.
           Ecological Applications, 1(2), 157-167.
"""

from typing import NamedTuple
import jax.numpy as jnp


class RespirationParams(NamedTuple):
    """Parameters for maintenance respiration.
    
    Attributes:
        br: Base rate for aboveground tissues [gC/gN/s]
            Original: 0.0106 molC/(molN h) = 2.525e-6 gC/(gN s)
            
        br_root: Base rate for fine roots [gC/gN/s]
            Can be set independently to tune root respiration
            
        q10: Temperature sensitivity coefficient [-]
            Reduced from 2.0 to 1.5 in CTSM for better seasonal CO2 cycle
            
        reference_temp: Reference temperature for Q10 [°C]
            Temperature at which base rates are defined
            
        acclimation_coef: Acclimation coefficient for root/stem [1/°C]
            Used when rootstem_acc=True: 10^(-0.00794*(T10-25))
            
        use_acclimation: Whether to apply temperature acclimation
            For boreal forests, this significantly affects productivity
            
        umol_to_gC: Conversion factor [gC/umol CO2]
            12.011e-6 gC/umol (for leaf MR from photosynthesis)
    """
    br: float = 2.525e-6  # gC/gN/s
    br_root: float = 2.525e-6  # gC/gN/s (can differ from br)
    q10: float = 1.5  # Dimensionless
    reference_temp: float = 20.0  # Degrees C
    acclimation_coef: float = 0.00794  # 1/°C
    use_acclimation: bool = False
    umol_to_gC: float = 12.011e-6  # gC/umol
    
    def get_temp_correction(
        self,
        temperature: jnp.ndarray,
        reference_temp_kelvin: float = 293.15  # 20°C in Kelvin
    ) -> jnp.ndarray:
        """Calculate Q10-based temperature correction factor.
        
        Args:
            temperature: Temperature in Kelvin [n_patches]
            reference_temp_kelvin: Reference temperature in Kelvin
            
        Returns:
            Temperature correction factor [n_patches]
        """
        # Convert to Celsius for calculation
        temp_c = temperature - 273.15
        return self.q10 ** ((temp_c - self.reference_temp) / 10.0)
    
    def get_acclimation_factor(
        self,
        t_10day: jnp.ndarray,
        reference_temp: float = 25.0  # °C
    ) -> jnp.ndarray:
        """Calculate acclimation adjustment factor.
        
        Args:
            t_10day: 10-day running mean temperature [K]
            reference_temp: Reference temperature for acclimation [°C]
            
        Returns:
            Acclimation factor to multiply base rates [n_patches]
        """
        # Use where instead of if for JIT compatibility
        temp_c = t_10day - 273.15
        acclim_factor = 10.0 ** (-self.acclimation_coef * (temp_c - reference_temp))
        return jnp.where(self.use_acclimation, acclim_factor, jnp.ones_like(t_10day))


# Default parameter sets for different scenarios
DEFAULT_PARAMS = RespirationParams()

RYAN_1991_PARAMS = RespirationParams(
    br=2.525e-6,
    br_root=2.525e-6,
    q10=2.0,  # Original Q10 before tuning
    use_acclimation=False,
)

ACCLIMATED_PARAMS = RespirationParams(
    br=2.525e-6,
    br_root=2.525e-6,
    q10=1.5,
    use_acclimation=True,
)

# Parameter bounds for optimization
PARAM_BOUNDS = {
    "br": (1.0e-6, 5.0e-6),
    "br_root": (1.0e-6, 5.0e-6),
    "q10": (1.2, 2.5),
    "acclimation_coef": (0.0, 0.02),
}
