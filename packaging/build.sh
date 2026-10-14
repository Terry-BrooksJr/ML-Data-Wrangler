#!/usr/bin/env bash
# Builds the desktop app into dist/ with PyInstaller.
#
#   packaging/build.sh
#
# Uses its own virtualenv (build/pkg-venv), synced from uv.lock, on the latest
# Python 3.10.x: PyInstaller refuses 3.10.0, which .python-version pins for
# development. PyInstaller doesn't cross-compile, so run this on each OS you
# ship for (macOS -> dist/Lasso.app, Windows/Linux -> dist/Lasso/).
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_VERSION="${PYTHON_VERSION:-3.10.21}"
export UV_PROJECT_ENVIRONMENT="$PWD/build/pkg-venv"

uv sync --frozen --no-dev --group package --python "$PYTHON_VERSION"

# NLTK's stopwords are downloaded at first use when running from source; bundle
# them so the packaged app works offline. PyInstaller's nltk hook collects
# every directory on nltk.data.path, NLTK_DATA included.
export NLTK_DATA="$PWD/build/nltk_data"
uv run --no-sync python -m nltk.downloader -q -d "$NLTK_DATA" stopwords

uv run --no-sync pyinstaller --noconfirm --clean \
    --workpath build/pyinstaller --distpath dist \
    packaging/lasso.spec

echo "Built: $(ls -d dist/*)"
