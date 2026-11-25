#!/bin/bash
#SBATCH --partition=capture
#SBATCH --account=workshop
#SBATCH --gres=gpu:1,scope:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --array=0-5%1
#SBATCH --job-name=aisb-attn-capture
#SBATCH --output=/home/tenant/experiments/attention-ablation-20261010/logs/capture-%A-%a.log
set -euo pipefail

experiment=/home/tenant/experiments/attention-ablation-20261010
variants=(efficient flash eager no-attention no-mlp no-blocks)
export AISB_ATTENTION_VARIANT="${variants[$SLURM_ARRAY_TASK_ID]}"
export LD_LIBRARY_PATH="$experiment/nvml-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2
export PYTHONPATH=/home/tenant/experiments/pico-access/workshop/capture
/home/tenant/envs/power-inference/bin/python \
    "$experiment/src/capture.py" \
    --model /home/tenant/experiments/pico-access/model/Qwen3-0.6B \
    --workload "$experiment/src/workload.py" \
    --output "$experiment/runs/$AISB_ATTENTION_VARIANT" \
    --batch 8 --sequence-length 512 --duration 0.42 --seed 20260922
