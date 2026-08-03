#!/bin/bash
# Setup script for Day 3 bootcamp pod
# This runs inside the RunPod pod after creation
set -e

echo "=== Day 3: LLM Inference Security - Pod Setup ==="
# Cell 1: Install dependencies and check GPU

# Install Python dependencies.
#
# Install from the repo's pinned requirements file rather than a floating list.
# Unpinned installs pull transformers 5.x, which imports DTensor from
# torch.distributed.tensor and fails against the torch 2.4 in the pod image.
which python
pip install --upgrade pip
pip install -r /workspace/aisb/3.1-tokenization/requirements.txt

# Notebook/test tooling that the exercises need but the requirements file
# does not cover (the pod image ships most of it already).
pip install jupyterlab ipywidgets ipykernel ipytest

echo ""
echo "=== Pre-downloading models (this takes a few minutes) ==="

# download_models.py owns the model list. Keeping a second inline copy here let
# the two drift apart, which is how an unused model id survived long enough to
# break setup.
python /workspace/aisb/3.1-tokenization/setup/download_models.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Models cached in /workspace/model-cache"
echo ""
echo "To start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
