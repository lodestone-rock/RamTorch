#!/usr/bin/env bash
set -e

# Publish RamTorch to PyPI
# Usage:
#   ./publish.sh           - publish to PyPI (requires TWINE_PASSWORD or prompts for token)
#   ./publish.sh --test    - publish to TestPyPI first

TARGET="pypi"
TWINE_REPO="https://upload.pypi.org/legacy/"

if [[ "$1" == "--test" ]]; then
    TARGET="testpypi"
    TWINE_REPO="https://test.pypi.org/legacy/"
    echo "Publishing to TestPyPI..."
else
    echo "Publishing to PyPI..."
fi

# Ensure build tools are available
pip install --quiet --upgrade build "twine>=5.0"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
echo "Building distributions..."
python -m build

# Check the distributions
echo "Checking distributions..."
twine check dist/*

# Upload
echo "Uploading to $TARGET..."
twine upload --repository-url "$TWINE_REPO" dist/*  --verbose

echo "Done! Package published to $TARGET."
