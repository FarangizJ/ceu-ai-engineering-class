#!/usr/bin/env bash
# Clears outputs from all notebooks in the repo, excluding .venv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find "$REPO_ROOT" -name "*.ipynb" \
  -not -path "*/.venv/*" \
  -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace {} \;

echo "All notebook outputs cleared."
