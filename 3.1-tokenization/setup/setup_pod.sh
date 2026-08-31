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
echo "=== Verifying the GPU is actually usable ==="

# nvidia-smi succeeding is not enough: a host with a faulty driver/UVM state
# still reports a healthy GPU and a device count of 1, while every CUDA context
# creation fails. download_models.py only prints "CUDA available: False" and
# carries on downloading to CPU, so without this check a GPU-less pod finishes
# setup looking perfectly provisioned and the fault surfaces mid-exercise.
# Allocate on the device to force real context creation.
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False -- this pod has no usable GPU.")
try:
    x = torch.randn(1024, 1024, device="cuda")
    (x @ x).sum().item()
except Exception as e:
    sys.exit(f"FATAL: CUDA is present but unusable ({type(e).__name__}: {e}).")
props = torch.cuda.get_device_properties(0)
print(f"GPU OK: {props.name} ({props.total_memory / 1e9:.1f} GB)")
PY

# Pre-seed the VS Code server extensions (Python/Pylance/Jupyter) so a
# participant connecting over Remote-SSH gets a working `# %%` cell runner
# immediately. Best effort: the script exits 0 even when it fails, but guard it
# anyway so `set -e` above can never abort provisioning over a missing click.
bash "$(dirname "$0")/install_vscode_extensions.sh" || true

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
