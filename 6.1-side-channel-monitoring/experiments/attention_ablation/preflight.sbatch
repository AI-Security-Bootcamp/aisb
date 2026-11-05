#!/bin/bash
#SBATCH --partition=capture
#SBATCH --account=workshop
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --job-name=aisb-attn-preflight
#SBATCH --output=/home/tenant/experiments/attention-ablation-20261010/logs/preflight-%j.log
set -euo pipefail

experiment=/home/tenant/experiments/attention-ablation-20261010
python=/home/tenant/envs/power-inference/bin/python
export LD_LIBRARY_PATH="$experiment/nvml-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$experiment/src"
"$python" -m unittest test_workload -v
for variant in flash eager no-attention no-mlp no-blocks; do
    AISB_ATTENTION_VARIANT="$variant" "$python" preflight.py \
        --model /home/tenant/experiments/pico-access/model/Qwen3-0.6B \
        --output "$experiment/preflight-$variant-v2"
done
