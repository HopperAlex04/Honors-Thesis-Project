# Versioning Guide

This project uses Git tags to mark stable versions. This allows users to easily access and return to specific versions of the codebase.

## Current Version

**v1.0.0** - Initial stable release with manual play functionality

## For Project Maintainers

### Creating a New Version Tag

When you're ready to create a new version (e.g., v2.0.0):

1. Ensure all changes are committed:
   ```bash
   git status
   git add .
   git commit -m "Your commit message"
   ```

2. Create an annotated tag:
   ```bash
   git tag -a v2.0.0 -m "Version 2.0.0 - Description of changes"
   ```

3. Push the tag to remote:
   ```bash
   git push origin v2.0.0
   # Or push all tags at once:
   git push origin --tags
   ```

4. Update the VERSION file:
   ```bash
   echo "2.0.0" > VERSION
   git add VERSION
   git commit -m "Update version to 2.0.0"
   git push
   ```

5. Update the README.md with the new version in the version history section.

### Version Numbering

Follow semantic versioning (semver):
- **MAJOR** (v1.0.0 → v2.0.0): Breaking changes
- **MINOR** (v1.0.0 → v1.1.0): New features, backward compatible
- **PATCH** (v1.0.0 → v1.0.1): Bug fixes, backward compatible

## For Users

### Viewing Available Versions

```bash
# List all version tags
git tag -l

# List tags with descriptions
git tag -l -n1

# View detailed information about a specific version
git show v1.0.0
```

### Accessing a Specific Version

```bash
# Checkout version 1.0.0
git checkout v1.0.0

# This puts you in "detached HEAD" state - safe for viewing/testing
# To make changes, create a new branch:
git checkout -b my-branch-name v1.0.0
```

### Returning to Latest Version

```bash
# Return to the main branch (latest version)
git checkout main

# Or if you're on a different branch:
git checkout <branch-name>
```

### Cloning a Specific Version

```bash
# Clone the repository
git clone https://github.com/HopperAlex04/Honors-Thesis-Project.git

# Checkout a specific version
cd Honors-Thesis-Project
git checkout v1.0.0
```

### Comparing Versions

```bash
# See what changed between versions
git diff v1.0.0 v2.0.0

# See commit history between versions
git log v1.0.0..v2.0.0
```

## Best Practices

1. **Always create annotated tags** (with `-a` flag) - they include metadata like author and date
2. **Write clear tag messages** - describe what's in this version
3. **Push tags to remote** - so others can access them
4. **Update VERSION file** - keeps version info accessible without Git
5. **Document changes** - update README version history when creating new tags

## Version History

- **v1.0.0** (2026-01-16) - Initial stable release with manual play functionality

