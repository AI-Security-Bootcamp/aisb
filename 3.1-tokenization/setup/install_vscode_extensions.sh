#!/bin/bash
# Pre-install the VS Code extensions the labs need, on the pod side.
#
# Under Remote-SSH an extension runs in one of two places:
#
#   * "UI" extensions -- including whatever the participant uses to open the
#     remote connection -- run in the VS Code window on their own laptop. They
#     cannot be installed from here; the pod is the thing being connected *to*.
#     deploy_runpod.py prints the client-side install commands for those.
#   * "Workspace" extensions -- Python, Pylance, Jupyter -- run in the VS Code
#     *server* that Remote-SSH starts on the pod, under ~/.vscode-server.
#     Those are what this script pre-seeds, so a participant who connects is
#     not left waiting on (or dismissing) the "Install in SSH" prompt, and the
#     `# %%` cell runner works on first open.
#
# Best effort by design: a failure here costs a click, not a lab, so every
# error path warns and the script still exits 0.

set -uo pipefail

# Remote-SSH always reads this directory, whatever server version the client
# happens to spawn, so extensions installed here survive a client upgrade.
EXT_DIR="$HOME/.vscode-server/extensions"

# The standalone VS Code CLI: a single static binary that can install
# extensions headlessly. We keep it out of ~/.vscode-cli, which is where the
# CLI itself caches the headless server build it downloads on first use.
CLI_BIN_DIR="$HOME/.local/bin"
CLI_BIN="$CLI_BIN_DIR/code"
CLI_URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64"

# Workspace-side extensions only. Pylance and debugpy ship platform-specific
# builds; the CLI resolves the right linux-x64 one, which is why we do not
# fetch .vsix files by hand.
EXTENSIONS=(
    ms-python.python
    ms-python.vscode-pylance
    ms-python.debugpy
    ms-toolsai.jupyter
    ms-toolsai.jupyter-renderers
)

warn() { echo "  WARNING: $*" >&2; }

echo ""
echo "=== Pre-installing VS Code server extensions ==="

mkdir -p "$EXT_DIR" "$CLI_BIN_DIR"

if [ ! -x "$CLI_BIN" ]; then
    echo "  Downloading the VS Code CLI..."
    if ! curl -fsSL "$CLI_URL" | tar -xz -C "$CLI_BIN_DIR"; then
        warn "could not download the VS Code CLI."
        warn "Participants can still accept VS Code's \"Install in SSH\" prompt."
        exit 0
    fi
    chmod +x "$CLI_BIN"
fi

# The first `ext install` pulls a headless server build (~100 MB, under a
# minute); the rest reuse it. Installing an extension that is already present
# is a no-op, so re-running setup on a provisioned pod is cheap.
failed=()
for ext in "${EXTENSIONS[@]}"; do
    echo "  - $ext"
    if ! "$CLI_BIN" ext install "$ext" --extensions-dir "$EXT_DIR"; then
        failed+=("$ext")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    warn "failed to install: ${failed[*]}"
    warn "Participants can install these from VS Code once connected."
else
    echo ""
    echo "VS Code server extensions ready in $EXT_DIR"
fi

echo ""
echo "Participants still install Python (ms-python.python) and Jupyter"
echo "(ms-toolsai.jupyter) in VS Code on their own machine: client-side"
echo "extensions cannot be installed from the pod."

exit 0
