#!/bin/bash
# Quick runner for Day 3 bootcamp exercises on RunPod
# Usage: ./run_exercise.sh <command>
#
# Examples:
#   ./run_exercise.sh setup       # Install deps + download models (run first!)
#   ./run_exercise.sh run         # Run the full day3_solution.py
#   ./run_exercise.sh test        # Run exercise tests

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="/workspace/model-cache"
export TRANSFORMERS_CACHE="/workspace/model-cache"

case "${1:-help}" in
    setup)
        echo "=== Day 3: Installing dependencies and downloading models ==="
        bash "$SCRIPT_DIR/setup_pod.sh"
        ;;
    run)
        echo "=== Running day3_solution.py ==="
        python3 "$SCRIPT_DIR/day3_solution.py"
        ;;
    test)
        echo "=== Running exercise tests ==="
        python3 -m pytest "$SCRIPT_DIR/day3_test.py" -v 2>/dev/null \
            || python3 "$SCRIPT_DIR/day3_test.py"
        ;;
    help|*)
        echo "Day 3: LLM Inference Security — Exercise Runner"
        echo ""
        echo "Usage: ./run_exercise.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup    Install dependencies + download models (run first!)"
        echo "  run      Run the full day3_solution.py end-to-end"
        echo "  test     Run the exercise test suite"
        echo ""
        echo "Or run directly:"
        echo "  python3 day3_solution.py"
        ;;
esac
