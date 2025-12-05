---
mode: agent
---

Your primary responsibilities:
1. Convert Fortran code to idiomatic JAX/Python
2. Apply JAX best practices (pure functions, immutable state, JIT-compatible)
3. Preserve exact scientific formulas and physics
4. Follow established patterns from the jax-ctsm codebase
5. Generate comprehensive type hints and docstrings
6. Convert loops to vectorized operations
7. Keep all the names exactly the same as in the Fortran code and keep all the dependencies as well
8. Map Fortran spatial hierarchy to JAX index-based arrays

Core JAX Principles to Follow:
- Pure functions (no side effects, no mutations)
- Immutable data structures (NamedTuples, not classes with __init__)
- JIT-compatible (no Python if with JAX arrays, use jnp.where instead)
- Vectorized operations (no Python loops, use jnp operations or vmap)
- Type hints for all functions
- Google-style docstrings with Args, Returns, and examples

Using static analysis output of the module dependencies and metadata, and the corresponding fortran code lines, generate a JAX/Python translation of the code following the best coding practices. 
Rename the py. file with the same name as the fortran file.