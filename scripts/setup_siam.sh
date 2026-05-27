#!/usr/bin/env bash
# Install siam-pred (pulls SIAM's nnU-Net wrapper) and download the v0.3
# pretrained weights to $SIAM_MODEL_DIR (default ~/siam_params).
#
# Usage:
#   scripts/setup_siam.sh
#
# After this runs, point the asparagus bridge at the result folder:
#   export SIAM_MODEL_DIR="$HOME/siam_params/v0.3/pred_DS108_LcsfP_Ano"
#
# Add SIAM_MODEL_DIR to .env so scripts/setup_asparagus_env.sh propagates it.

set -euo pipefail

repo="$(git rev-parse --show-toplevel)"

# siam-pred isn't a smri-fm dep; install on-demand so we don't pin nnunetv2's
# torch range. We rely on uv to share the smri-fm venv.
uv pip install --no-build-isolation 'siam-pred @ git+https://github.com/romainVala/SIAM.git@main'

# SIAM's downloader unzips to $SIAM_MODEL_DIR (default ~/siam_params).
SIAM_MODEL_DIR="${SIAM_MODEL_DIR:-$HOME/siam_params}"
export SIAM_MODEL_DIR
mkdir -p "$SIAM_MODEL_DIR"
echo ">>> downloading SIAM v0.3 weights to $SIAM_MODEL_DIR (this is ~5 GB)"
uv run python -m SIAMpred.download_model_weights

echo
echo ">>> done. Set in your .env or shell:"
echo "    SIAM_MODEL_DIR=$SIAM_MODEL_DIR/v0.3/pred_DS108_LcsfP_Ano"
