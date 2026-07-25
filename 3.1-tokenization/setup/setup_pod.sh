#!/bin/bash
# Setup script for Day 3 bootcamp pod
# This runs inside the RunPod pod after creation
set -e

echo "=== Day 3: LLM Inference Security - Pod Setup ==="
# Cell 1: Install dependencies and check GPU

# Install Python dependencies
which python
pip install --upgrade pip
pip install \
    torch \
    transformers \
    accelerate \
    scikit-learn \
    tqdm \
    numpy \
    matplotlib \
    jupyterlab \
    ipywidgets \
    termcolor~=3.1.0 \
    ipykernel \
    pytest \
    ipytest

echo ""
echo "=== Pre-downloading models (this takes a few minutes) ==="

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch

CACHE = '/workspace/model-cache'

# Tokenizers only (for Exercises 3.1.1–3.1.3)
print('Downloading tokenizers...')
for name in [
    'NousResearch/Meta-Llama-3-8B-Instruct',
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen2.5-0.5B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'unsloth/gemma-2-2b-it',
]:
    print(f'  {name} (tokenizer)')
    AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)

# Full models needed for exercises
print('Downloading google/gemma-4-E4B-it (guardrails)...')
AutoModelForCausalLM.from_pretrained('google/gemma-4-E4B-it', torch_dtype=torch.bfloat16, cache_dir=CACHE, trust_remote_code=True)

print('Downloading Qwen/Qwen3-0.6B (Exercises 3.1.1–3.1.3)...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float16, cache_dir=CACHE, trust_remote_code=True)

print('Downloading Qwen/Qwen2.5-0.5B (Exercise 3.1.3)...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', torch_dtype=torch.float16, cache_dir=CACHE, trust_remote_code=True)

print('Downloading GPT-2 (Exercises 3.4.1–3.4.3, distillation)...')
GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=CACHE)
GPT2LMHeadModel.from_pretrained('openai-community/gpt2', cache_dir=CACHE)

print('All models downloaded!')
"

# Download models
python /workspace/aisb-sg/3.1-tokenization/setup/download_models.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Models cached in /workspace/model-cache"
echo ""
echo "To start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
