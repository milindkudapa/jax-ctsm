# JAX-CTSM Team Guide - Quick Summary

**Full Guide**: See `TEAM_GUIDE.md` (1426 lines)

---

## 📖 What's Inside

### 1. **Quick Start for New Team Members** (Pages 1-2)
- Setup instructions
- Environment setup commands
- Key files to read first

### 2. **Development Workflow** (Pages 3-4)
- Git workflow (branch naming, commits)
- Standard development cycle
- How to create effective PRs

### 3. **Code Standards** (Pages 5-8)
- JAX-specific patterns (pure functions, immutability, JIT)
- 20+ DO/DON'T examples
- Array shape documentation
- Fortran translation checklist
- Parameter management

### 4. **Documentation Guidelines** (Pages 9-11)
- Module-level docs
- Function docstrings (with example)
- Inline comments best practices
- Validation documentation

### 5. **AI Agent Best Practices** (Pages 12-16)
- Effective prompting for AI (Cursor, Copilot)
- Context provision template
- AI code review checklist
- What AI is good/bad at
- AI-assisted debugging
- AI pair programming workflow

### 6. **Testing Requirements** (Pages 17-19)
- 80% minimum coverage requirement
- Test structure template
- 6 types of required tests
- Validation test examples

### 7. **Pull Request Template** (Pages 20-22)
- Complete PR template with sections:
  - Description
  - Fortran reference
  - Validation results
  - Performance benchmarks
  - Comprehensive checklist
  - Reviewer focus areas

### 8. **Review Checklist** (Pages 23-24)
- Scientific accuracy checks
- Code quality checks
- JAX compatibility checks
- Common review comments
- Approval criteria

### 9. **Common Pitfalls** (Pages 25-27)
- 9 common mistakes with solutions
- JAX-specific errors (TracerBoolConversionError, etc.)
- Scientific pitfalls (units, parameters)
- Performance pitfalls

### 10. **Resources** (Pages 28-29)
- Internal docs
- JAX tutorials
- CTSM references
- Scientific papers
- Team communication channels

### Appendices
- **Appendix A**: Complete feature implementation workflow
- **Appendix B**: Quick reference (commands, JAX snippets)

---

## 🎯 Most Important Sections

### For New Team Members
1. **Section 1**: Quick Start → Get up and running
2. **Section 3**: Code Standards → Learn JAX patterns
3. **Section 9**: Common Pitfalls → Avoid mistakes

### For AI Users
1. **Section 5**: AI Agent Best Practices → Use AI effectively
2. **Section 5.1**: Effective Prompting → Get better AI output
3. **Section 5.2**: AI Pair Programming → Workflow with AI

### For Contributors
1. **Section 2**: Development Workflow → How to contribute
2. **Section 7**: PR Template → Submit good PRs
3. **Section 6**: Testing Requirements → Write proper tests

### For Reviewers
1. **Section 8**: Review Checklist → What to check
2. **Section 8.2**: Common Review Comments → Standard feedback
3. **Section 8.3**: Approval Criteria → When to approve

---

## 🚀 Quick Reference Cards

### JAX Do's and Don'ts

```python
# ✅ DO: Pure functions
def calculate(state, params):
    new_state = state._replace(carbon=new_carbon)
    return new_state

# ❌ DON'T: Mutations
def calculate(state, params):
    state.carbon = new_carbon  # ERROR!
    return state
```

```python
# ✅ DO: JIT-compatible conditionals
result = jnp.where(is_woody, woody_calc, other_calc)

# ❌ DON'T: Python if with JAX arrays
if is_woody:  # TracerBoolConversionError!
    result = woody_calc
```

```python
# ✅ DO: Vectorized operations
total = jnp.sum(respiration * weights, axis=-1)

# ❌ DON'T: Python loops
for i in range(n):
    total += respiration[i] * weights[i]
```

### AI Prompting Template

```
"Implement [FEATURE] based on [FORTRAN_REFERENCE].
The function should:
1. [REQUIREMENT_1]
2. [REQUIREMENT_2]
3. Be JIT-compatible (no Python if statements)
4. Match Fortran implementation in [FILE] lines [X-Y]

Context:
- Module: [MODULE_NAME]
- Goal: [GOAL]
- Constraints: Pure function, immutable, type-hinted"
```

### PR Submission Checklist

Before submitting PR:
- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code formatted (`make format`)
- [ ] Validated against Fortran
- [ ] Docstrings complete
- [ ] PR template filled out
- [ ] Self-reviewed code

---

## 📊 Key Statistics

- **Total Length**: 1,426 lines
- **Code Examples**: 50+
- **Checklists**: 8
- **Templates**: 3 (PR, commit message, AI review)
- **Common Pitfalls**: 9 with solutions
- **Resource Links**: 15+

---

## 🎓 Training Path

### Foundations
1. Read Quick Start (Section 1)
2. Set up environment
3. Run examples and tests
4. Read spatial hierarchy docs

### Understanding
1. Study maintenance respiration module
2. Compare with Fortran source
3. Run validation script
4. Understand JAX patterns (Section 3)

### Contributing
1. Read development workflow (Section 2)
2. Study PR template (Section 7)
3. Make first small contribution
4. Get code reviewed

### Mastery
1. Add a test case
2. Implement small feature
3. Validate against Fortran
4. Review others' PRs

---

## 💡 Best Practices Highlights

### Scientific Accuracy (CRITICAL)
- **Always validate against Fortran** (< 0.01% difference)
- **Document parameter sources** (paper, year, equation)
- **Check units** explicitly in comments
- **Test conservation laws** (mass, energy)

### JAX Compatibility
- **Pure functions only** (no mutations)
- **Immutable data** (NamedTuple, not dataclass)
- **Use `jnp.where`** instead of `if` statements
- **Document array shapes** in comments

### AI Usage
- **Provide context** (Fortran reference, constraints)
- **Review everything** (AI can make scientific errors)
- **Add type hints** (AI often forgets)
- **Validate physics** (don't trust AI for science)

### Testing
- **80% minimum coverage**
- **Test against Fortran** (validation tests)
- **Test edge cases** (zero inputs, extremes)
- **Test JAX features** (JIT, vmap, grad)

---

## Quick Links

- **Full Guide**: `TEAM_GUIDE.md`
- **Quick Start**: `QUICKSTART.md`
- **Validation Results**: `VALIDATION_RESULTS.md`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Spatial Hierarchy**: `docs/spatial_hierarchy.md`

---

## Getting Help

1. **Check** `TEAM_GUIDE.md` first
2. **Search** closed GitHub issues
3. **Ask** in Slack
4. **Create** GitHub issue for bugs
5. **Discuss** in team meeting for design

---

## Welcome to the Team!

The full `TEAM_GUIDE.md` is comprehensive but approachable. Start with:
1. Section 1 (Quick Start)
2. Section 3 (Code Standards) 
3. Section 5 (AI Best Practices)

Then dive deeper as needed.

---

**Document**: `TEAM_GUIDE.md`  
**Length**: 1,426 lines  
**Sections**: 10 + 2 appendices  
**Last Updated**: October 2025
