# Documentation Reorganization - October 2025

## Summary

Reorganized all markdown documentation into a clear, structured hierarchy to improve discoverability and maintainability.

## What Changed

### Before (Disorganized)
```
jax-ctsm/
├── SANITY_CHECK_REPORT.md      # ❌ Report in root
├── FIXES_APPLIED.md            # ❌ Report in root
├── VALIDATION_RESULTS.md       # ❌ Report in root
├── DRIVER_IMPLEMENTATION.md    # ❌ Report in root
├── README.md                   # ✓ Keep
├── QUICKSTART.md               # ✓ Keep
├── CONTRIBUTING.md             # ✓ Keep
└── docs/
    ├── DRIVER_DESIGN.md        # Mixed case naming
    ├── DRIVER_POC.md           # Mixed case naming
    ├── IMPLEMENTATION_SUMMARY.md
    ├── spatial_hierarchy.md
    └── team-guide/
        ├── TEAM_GUIDE.md
        └── TEAM_GUIDE_SUMMARY.md
```

### After (Structured)
```
jax-ctsm/
├── README.md                   # Main entry point
├── QUICKSTART.md               # Quick start guide
├── CONTRIBUTING.md             # Contributing guidelines
└── docs/
    ├── README.md               # 📚 Documentation index
    ├── guides/                 # 📖 User-facing guides
    │   └── spatial_hierarchy.md
    ├── design/                 # 🏗️ Design documents
    │   ├── driver_design.md
    │   ├── driver_poc.md
    │   └── implementation_summary.md
    ├── reports/                # 📊 Development reports
    │   ├── sanity_check_report.md
    │   ├── fixes_applied.md
    │   ├── validation_results.md
    │   └── driver_implementation.md
    └── team/                   # 👥 Team documentation
        ├── team_guide.md
        └── team_guide_summary.md
```

## File Movements

### Root → docs/reports/
- `SANITY_CHECK_REPORT.md` → `docs/reports/sanity_check_report.md`
- `FIXES_APPLIED.md` → `docs/reports/fixes_applied.md`
- `VALIDATION_RESULTS.md` → `docs/reports/validation_results.md`
- `DRIVER_IMPLEMENTATION.md` → `docs/reports/driver_implementation.md`

### docs/ → docs/design/
- `DRIVER_DESIGN.md` → `docs/design/driver_design.md`
- `DRIVER_POC.md` → `docs/design/driver_poc.md`
- `IMPLEMENTATION_SUMMARY.md` → `docs/design/implementation_summary.md`

### docs/ → docs/guides/
- `spatial_hierarchy.md` → `docs/guides/spatial_hierarchy.md`

### docs/team-guide/ → docs/team/
- `team-guide/TEAM_GUIDE.md` → `docs/team/team_guide.md`
- `team-guide/TEAM_GUIDE_SUMMARY.md` → `docs/team/team_guide_summary.md`

## Naming Convention Changes

All files now follow consistent lowercase naming with underscores:
- ❌ `SANITY_CHECK_REPORT.md`
- ✅ `sanity_check_report.md`

## New Documentation Structure

### 📖 docs/guides/
**Purpose**: User-facing documentation for understanding and using JAX-CTSM

**Content**: 
- How-to guides
- Conceptual explanations
- Tutorials

**Current files**:
- `spatial_hierarchy.md` - CTSM's spatial hierarchy explained

### 🏗️ docs/design/
**Purpose**: Technical design documentation for developers

**Content**:
- Architecture decisions
- Implementation patterns
- Design rationale
- Technical comparisons

**Current files**:
- `driver_design.md` - Driver architecture and Fortran vs JAX
- `driver_poc.md` - Proof of concept details
- `implementation_summary.md` - Complete implementation overview

### 📊 docs/reports/
**Purpose**: Development reports and historical records

**Content**:
- Validation reports
- Bug fix logs
- Assessment reports
- Implementation milestones

**Current files**:
- `sanity_check_report.md` - Comprehensive code review (Oct 2025)
- `fixes_applied.md` - Log of bug fixes
- `validation_results.md` - Validation against Fortran CTSM
- `driver_implementation.md` - POC completion report

### 👥 docs/team/
**Purpose**: Team-specific documentation

**Content**:
- Onboarding guides
- Team processes
- Internal documentation

**Current files**:
- `team_guide.md` - Comprehensive team guide
- `team_guide_summary.md` - Quick reference

## Benefits

### Improved Organization
- ✅ Clear separation of concerns (guides vs design vs reports)
- ✅ Consistent naming conventions
- ✅ Intuitive directory structure
- ✅ Easier to find relevant documentation

### Better Discoverability
- ✅ Central documentation index at `docs/README.md`
- ✅ Updated main README with links to key documents
- ✅ Clear categorization by purpose

### Enhanced Maintainability
- ✅ Documented standards for future additions
- ✅ Clear guidelines for where new docs should go
- ✅ Consistent structure across all directories

### Professional Appearance
- ✅ Clean root directory (only essential files)
- ✅ Organized subdirectories by type
- ✅ Follows common documentation patterns

## Documentation Standards

Going forward, new documentation should follow these guidelines:

### Directory Selection
1. **User guides** → `docs/guides/` - "How to use this feature"
2. **Design docs** → `docs/design/` - "How this works and why"
3. **Reports** → `docs/reports/` - Historical records, assessments
4. **Team docs** → `docs/team/` - Team-specific processes

### Naming Guidelines
- Use lowercase with underscores: `my_document.md`
- Be descriptive: `spatial_hierarchy_guide.md` not `hierarchy.md`
- Use present tense for ongoing docs: `contributing.md`
- Use past tense for reports: `validation_results.md`

### Format Guidelines
- Include clear title (H1) at top
- Add table of contents for long documents
- Use code blocks with language specification
- Include examples where appropriate
- Link to related documents
- Use consistent emoji for visual organization (optional)

## Migration Notes

### Broken Links
If any internal links were broken by this reorganization, update them to:

**Old**:
```markdown
[Sanity Check](../SANITY_CHECK_REPORT.md)
```

**New**:
```markdown
[Sanity Check](docs/reports/sanity_check_report.md)
```

### External References
If external tools reference these files (CI/CD, scripts, etc.), update:
- All uppercase filenames → lowercase
- Root level paths → docs/ subdirectory paths

## Files Unchanged

The following files remain in their original locations:
- `README.md` - Main project README (root)
- `QUICKSTART.md` - Quick start guide (root)
- `CONTRIBUTING.md` - Contributing guide (root)

These are kept in root for:
- Maximum visibility
- GitHub/GitLab conventions
- Common expectations

## Next Steps

### For Users
- Bookmark `docs/README.md` as documentation entry point
- Use the table in main README for quick links
- Follow the "I want to..." guide in docs index

### For Developers
- Add new docs to appropriate subdirectory
- Follow naming conventions
- Update docs/README.md when adding major documents
- Reference docs in code comments and commit messages

### For Maintainers
- Review documentation quarterly
- Archive outdated reports
- Update links if moving files
- Keep docs/README.md synchronized

## Impact

### Immediate
- ✅ Cleaner project structure
- ✅ Easier navigation
- ✅ Professional appearance
- ✅ Better onboarding experience

### Long-term
- ✅ Scalable documentation system
- ✅ Reduced maintenance burden
- ✅ Consistent contributions
- ✅ Improved discoverability

## Verification

To verify the reorganization is complete:

```bash
# Check root directory (should only have 3 .md files)
ls -1 *.md
# Expected: README.md, QUICKSTART.md, CONTRIBUTING.md

# Check docs structure
tree docs/
# Should show organized subdirectories

# Find all markdown files
find docs -name "*.md" | sort
# Should show properly organized files
```

## Rollback (if needed)

To rollback this reorganization:

```bash
# Move reports back to root
mv docs/reports/sanity_check_report.md SANITY_CHECK_REPORT.md
mv docs/reports/fixes_applied.md FIXES_APPLIED.md
mv docs/reports/validation_results.md VALIDATION_RESULTS.md
mv docs/reports/driver_implementation.md DRIVER_IMPLEMENTATION.md

# Move design docs back to docs/
mv docs/design/*.md docs/

# Move guides back
mv docs/guides/*.md docs/

# Recreate team-guide subdirectory
mkdir docs/team-guide
mv docs/team/*.md docs/team-guide/

# Restore uppercase names as needed
# (use individual mv commands)
```

---

**Date**: October 9, 2025  
**Status**: ✅ Complete  
**Impact**: Low risk - only file movements, no code changes  
**Verification**: ✅ All files accounted for, no broken internal references

