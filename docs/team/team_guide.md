# JAX-CTSM Team Guide
## Training, Best Practices, AI Agents, and Contribution Guidelines

**Version**: 1.0  
**Last Updated**: October 2025 

---

## Table of Contents

1. [Quick Start for New Team Members](#1-quick-start-for-new-team-members)
2. [Development Workflow](#2-development-workflow)
3. [Code Standards and Best Practices](#3-code-standards-and-best-practices)
4. [Documentation Guidelines](#4-documentation-guidelines)
5. [AI Agent Best Practices](#5-ai-agent-best-practices)
6. [Testing Requirements](#6-testing-requirements)
7. [Pull Request Template](#7-pull-request-template)
8. [Review Checklist](#8-review-checklist)
9. [Common Pitfalls and Solutions](#9-common-pitfalls-and-solutions)
10. [Resources and References](#10-resources-and-references)

---

# 1. Quick Start for New Team Members

## 1.1 Welcome to JAX-CTSM!

You're joining a project to convert the Community Terrestrial Systems Model (CTSM) from Fortran to JAX, bringing modern machine learning capabilities to Earth system modeling.

### Prerequisites
- Python 3.8+
- Basic understanding of Earth system modeling
- Familiarity with NumPy (JAX is similar)
- Git/GitHub experience

### Your First Day

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_ORG/jax-ctsm.git
cd jax-ctsm

# 2. Set up your environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip setuptools
pip install -e ".[dev]"

# 4. Run tests to verify setup
pytest tests/ -v

# 5. Try the demo
python examples/driver_demo.py

# 6. Read the key docs
cat QUICKSTART.md
cat docs/IMPLEMENTATION_SUMMARY.md
cat docs/spatial_hierarchy.md
```

### Your First Week

**Day 1-2**: Understand the architecture
- Read `docs/IMPLEMENTATION_SUMMARY.md`
- Review `docs/spatial_hierarchy.md`
- Study `src/jax_ctsm/core/hierarchy.py`

**Day 3-4**: Understand a physics module
- Read `src/jax_ctsm/physics/maintenance_respiration.py`
- Compare with Fortran: `CTSM/src/biogeochem/CNMRespMod.F90`
- Run `validation/validate_against_fortran.py`

**Day 5**: Make your first contribution
- Fix a documentation typo
- Add a test case
- Improve a code comment

---

# 2. Development Workflow

## 2.1 Standard Git Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description

# 2. Make your changes
# ... edit files ...

# 3. Run tests locally
pytest tests/ -v
make format  # Auto-format code
make lint    # Check code quality

# 4. Commit with clear messages
git add src/jax_ctsm/physics/new_module.py
git add tests/physics/test_new_module.py
git commit -m "Add GPP calculation module

- Implement gross primary production physics
- Add unit tests for C3/C4 pathways
- Validate against Fortran CNNDynamicsMod.F90
- Closes #42"

# 5. Push and create PR
git push origin feature/your-feature-name
# Then create PR on GitHub
```

## 2.2 Branch Naming Conventions

| Type | Format | Example |
|------|--------|---------|
| New feature | `feature/description` | `feature/phenology-module` |
| Bug fix | `fix/description` | `fix/temperature-correction` |
| Documentation | `docs/description` | `docs/api-reference` |
| Performance | `perf/description` | `perf/vmap-optimization` |
| Validation | `validate/description` | `validate/allocation-fortran` |

## 2.3 Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `validate`

**Example**:
```
feat: Add leaf allocation module

- Implement leaf carbon/nitrogen allocation
- Add PFT-specific allocation coefficients
- Include priority-based allocation logic
- Validate against Fortran CNAllocationMod.F90

Tests:
- test_leaf_allocation_woody_plants
- test_leaf_allocation_crops
- test_allocation_nitrogen_limitation

Closes #45
```

---

# 3. Code Standards and Best Practices

## 3.1 JAX-Specific Principles

### ✅ DO: Pure Functions

```python
# GOOD: Pure function, no side effects
def calculate_respiration(
    patch_state: PatchState,
    params: RespirationParams,
) -> CarbonFlux:
    """Calculate maintenance respiration."""
    temp_correction = params.get_temp_correction(patch_state.temperature.t_ref2m)
    leaf_mr = patch_state.nitrogen.leafn * params.br * temp_correction
    return CarbonFlux(leaf_mr=leaf_mr, ...)
```

```python
# BAD: Mutation, side effects
def calculate_respiration(patch_state, params):
    patch_state.carbon.leaf_mr = ...  # MUTATION!
    return patch_state
```

### ✅ DO: Immutable Data Structures

```python
# GOOD: NamedTuple (immutable)
from typing import NamedTuple

class CarbonState(NamedTuple):
    leafc: jnp.ndarray
    frootc: jnp.ndarray
    
# Update using ._replace()
new_carbon = carbon_state._replace(leafc=new_leafc)
```

```python
# BAD: Mutable class
class CarbonState:
    def __init__(self):
        self.leafc = 0.0  # Mutable!
```

### ✅ DO: JIT-Compatible Code

```python
# GOOD: Use jnp.where for conditionals
result = jnp.where(is_woody, stem_calculation(), zero_array)
```

```python
# BAD: Python if statements with JAX arrays
if is_woody:  # ERROR: TracerBoolConversionError
    result = stem_calculation()
```

### ✅ DO: Vectorized Operations

```python
# GOOD: Vectorized
mr_total = jnp.sum(mr_components, axis=-1)
```

```python
# BAD: Python loops (slow)
for i in range(n_patches):
    mr_total[i] = sum(mr_components[i])
```

## 3.2 Code Style

We follow **PEP 8** with these additions:

```python
# Import order
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional

from jax_ctsm.core.hierarchy import PatchState
from jax_ctsm.params.respiration import RespirationParams

# Type hints ALWAYS
def ecosystem_timestep(
    patch_state: PatchState,
    params: EcosystemParams,
    dt: float = 1800.0,
) -> Tuple[PatchState, EcosystemFluxes]:
    """Execute one timestep.
    
    Args:
        patch_state: Current state
        params: Model parameters
        dt: Timestep in seconds
        
    Returns:
        Updated state and fluxes
    """
    pass

# Descriptive variable names
temp_correction = params.get_temp_correction(temperature)  # GOOD
tc = get_tc(t)  # BAD (not clear)

# Constants in UPPER_CASE
SECONDS_PER_DAY = 86400
N_REPR_POOLS = 3
```

## 3.3 Array Shapes and Documentation

**ALWAYS document array shapes!**

```python
def calculate_soil_respiration(
    soil_carbon: jnp.ndarray,  # [n_patches, n_layers]
    soil_temp: jnp.ndarray,    # [n_patches, n_layers]
    depth_weights: jnp.ndarray # [n_patches, n_layers]
) -> jnp.ndarray:              # [n_patches]
    """Calculate soil heterotrophic respiration.
    
    Args:
        soil_carbon: Carbon content by layer [gC/m2]
        soil_temp: Soil temperature by layer [K]
        depth_weights: Relative weights by layer (sum to 1)
        
    Returns:
        Total soil respiration [gC/m2/s]
    """
    pass
```

## 3.4 Parameter Management

```python
# GOOD: Parameters in NamedTuple
class AllocationParams(NamedTuple):
    """Allocation parameters."""
    leaf_priority: float = 1.0
    root_priority: float = 0.8
    stem_priority: float = 0.5
    
    def get_allocation_fraction(self, available_c: float) -> Dict[str, float]:
        """Calculate allocation fractions."""
        pass

params = AllocationParams()  # Use defaults
custom_params = AllocationParams(leaf_priority=1.5)  # Override
```

```python
# BAD: Hard-coded magic numbers
allocation = available_c * 0.35  # What is 0.35?
```

## 3.5 Fortran Translation Checklist

When translating Fortran to JAX:

- [ ] **Document source**: Add Fortran file reference in docstring
- [ ] **Preserve physics**: Keep scientific formulas identical
- [ ] **Convert arrays**: Fortran 1-indexed → Python 0-indexed
- [ ] **Handle filters**: Fortran filters → JAX boolean masks/where
- [ ] **Convert loops**: `do j=1,n` → vectorized operations or `vmap`
- [ ] **Fix conditionals**: `if` → `jnp.where` for JIT compatibility
- [ ] **Match parameters**: Use exact Fortran default values
- [ ] **Add validation**: Test against Fortran output

**Example**:
```python
def calculate_froot_mr(
    frootn: jnp.ndarray,        # [n_patches]
    t_soisno: jnp.ndarray,      # [n_patches, n_layers]
    crootfr: jnp.ndarray,       # [n_patches, n_layers]
    params: RespirationParams,
) -> jnp.ndarray:               # [n_patches]
    """Calculate fine root maintenance respiration.
    
    Translated from: CTSM/src/biogeochem/CNMRespMod.F90 (lines 278-295)
    
    Fortran implementation:
        do j = 1, nlevgrnd
            do fp = 1, num_soilp
                p = filter_soilp(fp)
                c = patch%column(p)
                tcsoi(c,j) = Q10**((t_soisno(c,j) - TFRZ - 20.0) / 10.0)
                froot_mr(p) = froot_mr(p) + frootn(p)*br_root*tcsoi(c,j)*crootfr(p,j)
            end do
        end do
    
    JAX implementation uses vectorized operations instead of loops.
    """
    # Temperature correction per layer
    tcsoi = params.q10 ** ((t_soisno - 273.15 - 20.0) / 10.0)  # [n_patches, n_layers]
    
    # Vectorized sum over layers
    froot_mr = jnp.sum(
        frootn[:, None] * params.br_root * tcsoi * crootfr,
        axis=-1
    )  # [n_patches]
    
    return froot_mr
```

---

# 4. Documentation Guidelines

## 4.1 Module-Level Documentation

Every module should have:

```python
"""
Module for calculating gross primary production (GPP).

This module implements the Farquhar-von Caemmerer-Berry (FvCB) model for
C3 photosynthesis and the Collatz model for C4 photosynthesis.

Fortran Reference:
    CTSM/src/biogeophys/PhotosynthesisMod.F90

Key Functions:
    - calculate_gpp: Main entry point for GPP calculation
    - farquhar_c3: C3 photosynthesis model
    - collatz_c4: C4 photosynthesis model

Scientific References:
    - Farquhar et al. (1980) Planta 149: 78-90
    - Collatz et al. (1992) Agric. For. Meteorol. 59: 1-25
    - Bonan et al. (2011) J. Geophys. Res. 116: G02019

Example:
    >>> from jax_ctsm import calculate_gpp, GPPParams
    >>> params = GPPParams()
    >>> gpp_flux = calculate_gpp(patch_state, params)
"""
```

## 4.2 Function Documentation

```python
def calculate_phenology(
    patch_state: PatchState,
    params: PhenologyParams,
    day_of_year: int,
) -> Tuple[PatchState, PhenologyFlux]:
    """Calculate phenology phase and associated carbon/nitrogen fluxes.
    
    Determines phenological state (dormant, leaf-on, leaf-off) and calculates
    litterfall, bud burst, and leaf senescence fluxes based on growing degree
    days (GDD) and chilling degree days (CDD).
    
    Fortran Reference:
        CTSM/src/biogeochem/CNPhenologyMod.F90
        - CNPhenology subroutine (lines 156-423)
    
    Args:
        patch_state: Current patch state containing:
            - carbon pools (leafc, etc.)
            - phenology state (gdd, cdd, phenology_phase)
            - temperature history (t_10day, t_ref2m)
        params: Phenology parameters including:
            - gdd_threshold: GDD required for bud burst [degree-days]
            - cdd_threshold: CDD required for leaf senescence [degree-days]
            - litterfall_rate: Background litterfall rate [1/s]
        day_of_year: Current day of year (1-365)
        
    Returns:
        new_state: Updated patch state with:
            - New phenology phase
            - Updated GDD/CDD accumulators
            - Modified carbon/nitrogen pools
        fluxes: Phenology fluxes including:
            - litterfall_c: Litterfall carbon flux [gC/m2/s]
            - litterfall_n: Litterfall nitrogen flux [gN/m2/s]
            - leafon_c: Leaf-on carbon flux [gC/m2/s]
    
    Notes:
        - Uses PFT-specific thresholds from `params`
        - Handles deciduous and evergreen separately
        - Crops use different phenology (not implemented here)
        
    Raises:
        ValueError: If day_of_year is outside [1, 365]
        
    Example:
        >>> params = PhenologyParams(gdd_threshold=100.0)
        >>> new_state, fluxes = calculate_phenology(patch_state, params, 120)
        >>> print(f"Litterfall: {fluxes.litterfall_c:.6f} gC/m2/s")
    
    References:
        White et al. (1997) Clim. Change 37: 347-375
        Richardson et al. (2013) Agric. For. Meteorol. 180: 249-263
    """
    # Implementation here
    pass
```

## 4.3 Inline Comments

```python
# GOOD: Explain WHY, not WHAT
# Use Q10 = 1.5 (tuned value from Richardson et al. 2018)
# Original value of 2.0 overestimated tropical respiration
q10 = 1.5

# BAD: Obvious comments
# Set q10 to 1.5
q10 = 1.5
```

## 4.4 Validation Documentation

Every physics module should have a validation section:

```python
# In tests/physics/test_module.py
"""
Validation against Fortran CTSM
================================

Test Case 1: Temperate Deciduous Forest
---------------------------------------
Input:
    - PFT: Temperate deciduous broadleaf
    - LAI: 5.0 m2/m2
    - Temperature: 25°C
    
Expected Output (from Fortran):
    - GPP: 1.234e-5 gC/m2/s
    - Respiration: 5.678e-6 gC/m2/s
    
JAX Output:
    - GPP: 1.234e-5 gC/m2/s (✓ matches)
    - Respiration: 5.678e-6 gC/m2/s (✓ matches)
    
Difference: < 0.001%

Test Case 2: Tropical Rainforest
---------------------------------
[...]
"""
```

---

# 5. AI Agent Best Practices

## 5.1 Working with AI Assistants (Cursor, GitHub Copilot, etc.)

### Effective Prompting

**DO**:
```
"Implement the temperature acclimation factor for root respiration 
based on the Atkin et al. 2015 model. The function should:
1. Take 10-day running mean temperature as input
2. Use the acclimation coefficient from params
3. Return adjustment factor as JAX array
4. Be JIT-compatible (no Python if statements)
5. Match the Fortran implementation in CNMRespMod.F90 lines 249-252"
```

**DON'T**:
```
"Add acclimation"  # Too vague
```

### Providing Context

When asking AI for help, provide:

```python
"""
CONTEXT for AI:
- Project: JAX-CTSM (Fortran to JAX conversion)
- Module: Maintenance respiration
- Goal: Add root acclimation feature
- Fortran reference: CTSM/src/biogeochem/CNMRespMod.F90 (lines 249-252)
- Constraints: Must be pure function, immutable, JIT-compatible
- Related code: src/jax_ctsm/params/respiration.py

FORTRAN CODE:
if(rootstem_acc)then
    br_root = br_root * 10._r8**(-0.00794_r8*((t10(p)-tfrz)-25._r8))
end if

TASK:
Convert this to JAX, making it JIT-compatible using jnp.where
"""
```

### Reviewing AI-Generated Code

**ALWAYS check**:
1. ✅ Type hints present
2. ✅ Docstring complete
3. ✅ Array shapes documented
4. ✅ Pure function (no mutations)
5. ✅ JIT-compatible (no Python if with JAX arrays)
6. ✅ Matches Fortran physics
7. ✅ Has unit tests
8. ✅ Validated against Fortran

**Example review**:
```python
# AI-generated code
def calculate_something(x, y):
    return x * y

# YOUR REVIEW COMMENTS:
# ❌ Missing type hints
# ❌ Missing docstring
# ❌ No array shape documentation
# ❌ No Fortran reference
# ❌ What are x and y physically?

# IMPROVED VERSION:
def calculate_leaf_respiration(
    leaf_nitrogen: jnp.ndarray,  # [n_patches] gN/m2
    temp_correction: jnp.ndarray, # [n_patches] dimensionless
    base_rate: float = 2.525e-6,  # gC/gN/s
) -> jnp.ndarray:                 # [n_patches] gC/m2/s
    """Calculate leaf maintenance respiration.
    
    Fortran: CNMRespMod.F90, line 256
    Formula: leaf_mr = leafn * br * tc
    
    Args:
        leaf_nitrogen: Leaf N content [gN/m2]
        temp_correction: Q10 temperature correction factor
        base_rate: Base respiration rate [gC/gN/s]
        
    Returns:
        Leaf maintenance respiration [gC/m2/s]
    """
    return leaf_nitrogen * base_rate * temp_correction
```

### AI-Assisted Debugging

When debugging with AI:

```python
"""
BUG REPORT:
-----------
Error: TracerBoolConversionError

Code:
    if use_acclimation:
        factor = 10.0 ** (-0.00794 * (temp - 25.0))
    else:
        factor = 1.0

Environment:
- JAX version: 0.4.20
- Inside JIT-compiled function: YES
- Variable types: use_acclimation is bool, temp is jnp.ndarray

Expected: JIT-compatible code
Need: Convert to use jnp.where instead of if statement
"""

# AI will help convert to:
factor = jnp.where(
    use_acclimation,
    10.0 ** (-0.00794 * (temp - 25.0)),
    1.0
)
```

## 5.2 AI Pair Programming Workflow

1. **Initial implementation**: Let AI generate boilerplate
2. **Review and refine**: Check against standards
3. **Add physics**: Ensure scientific accuracy
4. **Validate**: Compare with Fortran
5. **Document**: Add comprehensive docs
6. **Test**: Write unit tests
7. **Optimize**: Use AI for performance suggestions

## 5.3 What AI is Good At

✅ **Use AI for**:
- Boilerplate code generation
- Converting Fortran loops to vectorized JAX
- Writing unit tests
- Documentation generation
- Type hint suggestions
- Code formatting
- Finding similar code patterns

❌ **Don't rely solely on AI for**:
- Scientific correctness (always validate!)
- Architecture decisions (discuss with team)
- Parameter values (use literature/Fortran)
- Performance optimization (profile first)

## 5.4 AI Code Review Template

```markdown
## AI-Generated Code Review

**File**: src/jax_ctsm/physics/new_module.py
**Generated by**: Cursor AI
**Date**: 2025-10-04

### Checks Performed
- [ ] Type hints complete
- [ ] Docstrings present and detailed
- [ ] Array shapes documented
- [ ] Pure function (no mutations)
- [ ] JIT-compatible
- [ ] Matches Fortran reference
- [ ] Unit tests added
- [ ] Validation against Fortran performed
- [ ] Scientific references cited

### Changes Made
1. Added missing type hints for `temperature` parameter
2. Fixed JIT incompatibility: converted `if` to `jnp.where`
3. Added Fortran reference in docstring
4. Corrected parameter value (0.00794 → 0.00794 confirmed)

### Remaining Issues
- [ ] Need to validate against actual Fortran output
- [ ] Missing test case for C4 crops

### Reviewer Notes
AI did well on structure but didn't catch parameter units mismatch.
Manual validation caught this. Physics is correct after fix.
```

---

# 6. Testing Requirements

## 6.1 Test Coverage Requirements

**Minimum**: 80% code coverage  
**Target**: 90%+ for physics modules

```bash
# Run tests with coverage
pytest tests/ --cov=src/jax_ctsm --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 6.2 Test Structure

Every physics module needs:

```python
# tests/physics/test_module.py

import pytest
import jax.numpy as jnp
from jax_ctsm import calculate_module, ModuleParams

class TestModuleBasic:
    """Basic functionality tests."""
    
    def test_output_shape(self):
        """Test output has correct shape."""
        pass
    
    def test_zero_input(self):
        """Test behavior with zero inputs."""
        pass
    
    def test_positive_definite(self):
        """Test outputs are positive where expected."""
        pass

class TestModuleFortranValidation:
    """Validation against Fortran CTSM."""
    
    def test_temperate_forest(self):
        """Test temperate forest case against Fortran."""
        pass
    
    def test_tropical_forest(self):
        """Test tropical forest case."""
        pass
    
    def test_boreal_forest(self):
        """Test boreal forest case."""
        pass

class TestModuleEdgeCases:
    """Edge cases and error handling."""
    
    def test_extreme_temperature(self):
        """Test behavior at extreme temperatures."""
        pass
    
    def test_zero_nitrogen(self):
        """Test with zero nitrogen."""
        pass

class TestModuleJAXCompatibility:
    """JAX-specific tests."""
    
    def test_jit_compilation(self):
        """Test function can be JIT-compiled."""
        pass
    
    def test_vmap_batching(self):
        """Test function can be vmapped."""
        pass
    
    def test_grad_differentiability(self):
        """Test function is differentiable."""
        pass
```

## 6.3 Required Test Cases

For each physics module, test:

1. **Basic functionality**
   - Correct output shape
   - Correct output type
   - Basic input/output relationship

2. **Physical realism**
   - Positive definite (where expected)
   - Conservation laws (mass, energy)
   - Reasonable magnitudes

3. **Fortran validation**
   - Match Fortran output (< 0.01% difference)
   - Test multiple PFTs
   - Test seasonal variation

4. **Edge cases**
   - Zero inputs
   - Extreme values
   - Missing data handling

5. **JAX compatibility**
   - JIT compilation
   - vmap batching
   - Gradient computation

6. **Performance**
   - Benchmark execution time
   - Memory usage

## 6.4 Validation Tests

```python
def test_validate_against_fortran():
    """Validate JAX output matches Fortran CTSM.
    
    Fortran output from:
        CTSM simulation: tests/data/fortran_output.nc
        Configuration: temperate_forest_default
        Timestep: 2020-07-15 12:00:00
    """
    # Load Fortran output
    fortran_output = load_fortran_reference("temperate_forest_gpp.nc")
    
    # Create equivalent JAX state
    patch_state = create_patch_from_fortran(fortran_output)
    params = GPPParams()
    
    # Run JAX calculation
    jax_output = calculate_gpp(patch_state, params)
    
    # Compare
    assert jnp.allclose(
        jax_output.gpp,
        fortran_output["GPP"],
        rtol=1e-4  # 0.01% tolerance
    ), f"JAX GPP differs from Fortran by {relative_diff:.4%}"
```

---

# 7. Pull Request Template

When creating a PR, use this template:

```markdown
## Description

Brief description of what this PR does.

**Type**: [Feature / Bug Fix / Documentation / Performance / Validation]

**Related Issue**: Closes #XX

---

## Changes Made

### Physics/Code Changes
- [ ] Implemented X module based on Fortran Y
- [ ] Added parameter Z from literature
- [ ] Fixed bug in temperature correction

### Tests
- [ ] Added unit tests for module
- [ ] Added validation against Fortran
- [ ] All tests passing locally

### Documentation
- [ ] Updated docstrings
- [ ] Added example usage
- [ ] Updated relevant docs/

---

## Fortran Reference

**File**: `CTSM/src/biogeochem/ModuleName.F90`  
**Lines**: 123-456  
**Subroutine**: `SubroutineName`

**Key formulas**:
```fortran
! Fortran code here
gpp = vcmax * ci / (ci + kc)
```

**JAX equivalent**:
```python
gpp = vcmax * ci / (ci + kc)
```

---

## Validation Results

### Test Case 1: Temperate Deciduous Forest
- **Input**: LAI=5.0, T=25°C, PAR=1500 μmol/m2/s
- **Fortran output**: GPP = 1.234e-5 gC/m2/s
- **JAX output**: GPP = 1.234e-5 gC/m2/s
- **Difference**: < 0.001% ✅

### Test Case 2: Tropical Rainforest
- **Input**: LAI=7.0, T=30°C, PAR=2000 μmol/m2/s
- **Fortran output**: GPP = 2.345e-5 gC/m2/s
- **JAX output**: GPP = 2.345e-5 gC/m2/s
- **Difference**: < 0.001% ✅

---

## Performance

### Benchmarks
- **Single timestep**: 2.3 ms (CPU), 0.5 ms (GPU)
- **100 patches**: 5.1 ms (CPU), 0.8 ms (GPU)
- **Memory**: 45 MB peak

### Comparison with Fortran
- **Speedup**: 15x on CPU, 60x on GPU
- **Memory**: Similar to Fortran

---

## Checklist

### Code Quality
- [ ] Code follows style guide (PEP 8)
- [ ] Type hints for all functions
- [ ] Docstrings for all public functions
- [ ] Array shapes documented
- [ ] No hard-coded magic numbers

### JAX Compatibility
- [ ] Pure functions (no mutations)
- [ ] Immutable data structures
- [ ] JIT-compatible (no Python `if` with JAX arrays)
- [ ] Successfully JIT compiles
- [ ] Successfully vmaps
- [ ] Successfully differentiates (if applicable)

### Testing
- [ ] Unit tests added
- [ ] Edge cases tested
- [ ] Validation against Fortran
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Coverage > 80% for new code

### Documentation
- [ ] Module docstring complete
- [ ] Function docstrings with Args/Returns
- [ ] Fortran reference cited
- [ ] Example usage provided
- [ ] Scientific references cited

### Scientific Accuracy
- [ ] Physics matches Fortran
- [ ] Parameters from literature/Fortran
- [ ] Units documented and correct
- [ ] Conservation laws preserved
- [ ] Magnitudes reasonable

### AI Usage (if applicable)
- [ ] AI-generated code reviewed
- [ ] Type hints added/verified
- [ ] Physics validated by human
- [ ] Tests written (not just generated)

---

## Screenshots/Plots (if applicable)

[Add validation plots, performance graphs, etc.]

---

## Additional Notes

Any special considerations, known limitations, or future work.

---

## Reviewer Focus Areas

Please pay special attention to:
1. Temperature acclimation implementation (lines 45-67)
2. Layer-weighted respiration sum (lines 123-145)
3. Edge case: zero nitrogen handling (lines 200-210)

---

## Breaking Changes

- [ ] This PR introduces breaking changes
- [ ] Breaking changes documented
- [ ] Migration guide provided

---

## Post-Merge Tasks

- [ ] Update user documentation
- [ ] Add to release notes
- [ ] Notify team of new feature

---

**Ready for review**: @reviewer1 @reviewer2
```

---

# 8. Review Checklist

## 8.1 Reviewer Checklist

When reviewing a PR, check:

### Scientific Accuracy (CRITICAL)
- [ ] Physics matches Fortran reference
- [ ] Parameters match literature/Fortran values
- [ ] Units are correct and documented
- [ ] Formulas are mathematically correct
- [ ] Conservation laws preserved (mass, energy)
- [ ] Validated against Fortran output

### Code Quality
- [ ] Follows style guide
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] No code duplication
- [ ] Descriptive variable names
- [ ] Array shapes documented

### JAX Compatibility
- [ ] Pure functions (no side effects)
- [ ] Immutable data structures
- [ ] JIT-compatible code
- [ ] No Python `if` with JAX arrays
- [ ] Vectorized (not Python loops)

### Testing
- [ ] Tests added for new functionality
- [ ] Edge cases covered
- [ ] Fortran validation included
- [ ] All tests passing
- [ ] Coverage sufficient (>80%)

### Documentation
- [ ] Module docstring
- [ ] Function docstrings
- [ ] Inline comments for complex logic
- [ ] Example usage
- [ ] Updated relevant docs/

### Performance
- [ ] No obvious performance issues
- [ ] Appropriate use of JIT/vmap
- [ ] Benchmarks provided (if performance-critical)

## 8.2 Common Review Comments

**Physics Issues**:
```
⚠️ The temperature correction here differs from Fortran.
Fortran uses: Q10**((T - 20) / 10)
Your code has: Q10**((T - 25) / 10)
Please verify reference temperature.
```

**JAX Compatibility**:
```
❌ This code will fail JIT compilation.
Line 45: if is_woody:
Should be: result = jnp.where(is_woody, woody_calc(), non_woody_calc())
```

**Documentation**:
```
📝 Missing array shape documentation.
Please add shapes for all arrays:
    leaf_nitrogen: jnp.ndarray,  # [n_patches] gN/m2
```

**Testing**:
```
🧪 Please add test for edge case: what happens when LAI = 0?
```

## 8.3 Approval Criteria

**Requirements for approval**:
1. ✅ All checklist items complete
2. ✅ CI/CD passing
3. ✅ At least one approval from:
   - Science reviewer (physics/validation)
   - Code reviewer (implementation/JAX)
4. ✅ No unresolved comments
5. ✅ Passes Fortran validation

---

# 9. Common Pitfalls and Solutions

## 9.1 JAX-Specific Pitfalls

### Pitfall 1: Python `if` with JAX arrays

```python
# ❌ WRONG
if is_woody:  # TracerBoolConversionError!
    result = calculate_stem()

# ✅ CORRECT
result = jnp.where(is_woody, calculate_stem(), jnp.zeros_like(default))
```

### Pitfall 2: Array mutation

```python
# ❌ WRONG
state.carbon.leafc[0] = new_value  # Can't mutate NamedTuple!

# ✅ CORRECT
new_carbon = state.carbon._replace(leafc=new_leafc)
new_state = state._replace(carbon=new_carbon)
```

### Pitfall 3: Python loops instead of vectorization

```python
# ❌ SLOW
for i in range(n_layers):
    total += respiration[i] * weights[i]

# ✅ FAST
total = jnp.sum(respiration * weights, axis=-1)
```

### Pitfall 4: Shape mismatches

```python
# ❌ ERROR: Shapes don't match
leafn = jnp.array([0.1, 0.2, 0.3])  # [3]
temp_correction = jnp.array([1.2])   # [1]
result = leafn * temp_correction      # Broadcasting error!

# ✅ CORRECT: Ensure compatible shapes
result = leafn * temp_correction[:, None]  # Broadcast correctly
# Or ensure both have shape [n_patches]
```

## 9.2 Scientific Pitfalls

### Pitfall 5: Wrong parameter values

```python
# ❌ WRONG: Made up value
base_rate = 2.5e-6  # Where did this come from?

# ✅ CORRECT: Documented source
# From Ryan (1991): 0.0106 molC/(molN·h)
# Converted to gC/gN/s: 0.0106 * 12.011/14.007 / 3600
base_rate = 2.525e-6  # gC/gN/s
```

### Pitfall 6: Unit errors

```python
# ❌ WRONG: Mixed units
temperature_c = 25.0  # Celsius
q10_correction = 1.5 ** ((temperature_c - 273.15 - 20.0) / 10.0)  # Wrong!

# ✅ CORRECT: Consistent units
temperature_k = 298.15  # Kelvin
temperature_c = temperature_k - 273.15  # Convert to Celsius
q10_correction = 1.5 ** ((temperature_c - 20.0) / 10.0)
```

### Pitfall 7: Missing validation

```python
# ❌ WRONG: No validation
def new_module(...):
    # ... implementation ...
    return result
# How do we know this is correct?

# ✅ CORRECT: Validate against Fortran
def test_validate_new_module():
    fortran_output = 1.234e-5  # From CTSM simulation
    jax_output = new_module(...)
    assert jnp.allclose(jax_output, fortran_output, rtol=1e-4)
```

## 9.3 Performance Pitfalls

### Pitfall 8: Unnecessary JIT recompilation

```python
# ❌ SLOW: JIT recompiles every call
@jax.jit
def process(data, flag=True):  # Python bool causes recompilation
    return jnp.where(flag, data * 2, data)

# ✅ FAST: Use static_argnums
@jax.jit(static_argnums=(1,))
def process(data, flag: bool = True):
    return jnp.where(flag, data * 2, data)
```

### Pitfall 9: Large array copies

```python
# ❌ SLOW: Creates many intermediate copies
result = state
for i in range(100):
    result = result._replace(carbon=new_carbon)

# ✅ FAST: Minimize replacements
# Compute all updates first, then apply once
final_carbon = compute_all_updates(state.carbon)
result = state._replace(carbon=final_carbon)
```

---

# 10. Resources and References

## 10.1 Internal Documentation

- **Quick Start**: `QUICKSTART.md`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Spatial Hierarchy**: `docs/spatial_hierarchy.md`
- **Driver Design**: `docs/DRIVER_DESIGN.md`
- **Validation Results**: `VALIDATION_RESULTS.md`

## 10.2 JAX Resources

- **JAX Documentation**: https://jax.readthedocs.io/
- **JAX Tutorials**: https://jax.readthedocs.io/en/latest/tutorials.html
- **JAX 101**: https://jax.readthedocs.io/en/latest/jax-101/index.html
- **Common Gotchas**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

## 10.3 CTSM Resources

- **CTSM Documentation**: https://escomp.github.io/ctsm-docs/
- **CTSM Source Code**: https://github.com/ESCOMP/CTSM
- **Technical Note**: https://escomp.github.io/ctsm-docs/tech_note/index.html

## 10.4 Scientific References

### Respiration
- Ryan, M. G. (1991). Effects of climate change on plant respiration. *Ecological Applications*, 1(2), 157-167.
- Atkin, O. K., et al. (2015). Global variability in leaf respiration. *New Phytologist*, 206(2), 614-636.

### Photosynthesis
- Farquhar, G. D., et al. (1980). A biochemical model of photosynthetic CO2 assimilation. *Planta*, 149(1), 78-90.
- Collatz, G. J., et al. (1992). Coupled photosynthesis-stomatal conductance model. *Agricultural and Forest Meteorology*, 54(2-4), 107-136.

### Phenology
- White, M. A., et al. (1997). Parameterization and sensitivity analysis of the BIOME-BGC model. *Climatic Change*, 37(2), 347-375.

## 10.5 Team Communication

TBD

## 10.6 Getting Help

TBD
---

# Appendix A: Example Workflow

## Complete Feature Implementation

Here's a complete workflow for adding a new physics module:

```bash
# 1. Create branch
git checkout -b feature/gpp-calculation

# 2. Create module structure
touch src/jax_ctsm/physics/gpp.py
touch src/jax_ctsm/params/gpp.py
touch tests/physics/test_gpp.py
touch validation/validate_gpp_fortran.py

# 3. Implement (with AI assistance)
# ... edit files ...

# 4. Add tests
pytest tests/physics/test_gpp.py -v

# 5. Validate against Fortran
python validation/validate_gpp_fortran.py

# 6. Format and lint
make format
make lint

# 7. Run full test suite
pytest tests/ -v --cov=src/jax_ctsm

# 8. Commit
git add src/jax_ctsm/physics/gpp.py
git add src/jax_ctsm/params/gpp.py
git add tests/physics/test_gpp.py
git add validation/validate_gpp_fortran.py
git commit -m "feat: Add GPP calculation module

- Implement Farquhar C3 photosynthesis model
- Implement Collatz C4 photosynthesis model
- Add PFT-specific Vcmax parameters
- Validate against Fortran PhotosynthesisMod.F90

Tests:
- test_gpp_c3_temperate
- test_gpp_c4_tropical
- test_gpp_validation_fortran

Validation:
- C3 forest: <0.01% difference from Fortran
- C4 crops: <0.01% difference from Fortran

Closes #75"

# 9. Push and create PR
git push origin feature/gpp-calculation
# Create PR on GitHub using template

# 10. Address review comments
# ... make changes ...
git add .
git commit -m "Address review comments

- Fix temperature unit mismatch
- Add edge case test for LAI=0
- Clarify Vcmax parameter source"
git push

# 11. After approval, squash and merge
```

---

# Appendix B: Quick Reference

## Common Commands

```bash
# Setup
pip install -e ".[dev]"

# Testing
pytest tests/ -v                      # Run all tests
pytest tests/physics/test_module.py   # Run specific test
pytest tests/ --cov=src/jax_ctsm     # With coverage

# Code quality
make format                           # Auto-format
make lint                            # Check style
make type-check                      # Type checking

# Documentation
make docs                            # Build docs
make docs-serve                      # Serve docs locally

# Validation
python validation/validate_against_fortran.py

# Examples
python examples/driver_demo.py
python examples/basic_maintenance_respiration.py
```

## JAX Quick Reference

```python
# Arrays
x = jnp.array([1, 2, 3])
y = jnp.zeros((10, 5))
z = jnp.ones_like(x)

# Conditionals
result = jnp.where(condition, if_true, if_false)

# Aggregation
total = jnp.sum(arr, axis=-1)
mean = jnp.mean(arr, axis=0)

# JIT
@jax.jit
def fast_function(x):
    return x ** 2

# vmap (vectorize)
batch_fn = jax.vmap(single_fn)

# grad (differentiate)
grad_fn = jax.grad(loss_fn)
```

---

**Version History**:
- v1.0 (2025-10-04): Initial version

**Contributors**: TBD

**Questions?** Contact: TBD

---

**END OF TEAM GUIDE**

Remember: When in doubt, ask!