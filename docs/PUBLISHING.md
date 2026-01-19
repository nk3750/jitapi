# Publishing Guide

This guide is for maintainers who need to publish JitAPI to PyPI.

## Prerequisites (One-time Setup)

1. Create accounts:
   - PyPI: https://pypi.org/account/register/
   - TestPyPI: https://test.pypi.org/account/register/

2. Generate API tokens:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/

3. Store tokens in `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

4. Secure the file:

```bash
chmod 600 ~/.pypirc
```

## Pre-publish Checklist

Before publishing, verify these files are updated:

| File | Check |
|------|-------|
| `pyproject.toml` | Version number is correct |
| `src/jitapi/__init__.py` | `__version__` matches pyproject.toml |
| `README.md` | Installation instructions are accurate |

## Build Commands

```bash
# Install build tools
pip install build twine

# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# This creates:
#   dist/jitapi-X.Y.Z-py3-none-any.whl
#   dist/jitapi-X.Y.Z.tar.gz

# Verify the build
twine check dist/*
```

## Test on TestPyPI First

Always test on TestPyPI before publishing to production:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation in a fresh environment
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    jitapi

# Verify it works
jitapi --help
```

## Publish to PyPI

Once TestPyPI testing passes:

```bash
# Upload to production PyPI
twine upload dist/*

# Verify installation
pip install jitapi
uvx jitapi --help
```

## Version Bumping

For future releases:

1. Update version in both files:
   - `pyproject.toml`: `version = "X.Y.Z"`
   - `src/jitapi/__init__.py`: `__version__ = "X.Y.Z"`

2. Clean, build, and publish:

```bash
rm -rf dist/ build/ *.egg-info
python -m build
twine upload dist/*
```

## Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes, backwards compatible

## Verification Steps

After publishing, verify the release:

1. **Test fresh pip install:**
   ```bash
   pip install jitapi
   jitapi --help
   ```

2. **Test uvx:**
   ```bash
   uvx jitapi --help
   ```

3. **Test with Claude Desktop:**
   - Configure MCP server
   - Ask: "Register the Petstore API"
   - Verify it works without errors

4. **Test with Claude Code:**
   - Configure MCP server
   - Ask: "List JitAPI tools"
   - Should show all 9 tools

## Troubleshooting

### Upload fails with "File already exists"

You cannot overwrite an existing version on PyPI. Bump the version number and try again.

### Package not found after upload

PyPI may take a few minutes to index new packages. Wait and try again.

### Dependencies not installing

Ensure all dependencies are listed in `pyproject.toml` under `dependencies`.
