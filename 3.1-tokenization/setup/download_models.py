import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


os.environ['HF_HOME'] = '/workspace/model-cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/model-cache'
CACHE = os.getenv('TRANSFORMERS_CACHE')

# Tokenizers only
for name in ['NousResearch/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B',
             'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'unsloth/gemma-2-2b-it']:
    print(f'Downloading tokenizer: {name}')
    AutoTokenizer.from_pretrained(name, cache_dir=CACHE, trust_remote_code=True)

# Full models
for name in ['google/gemma-4-E4B-it', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B']:
    print(f'Downloading model: {name}')
    AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, cache_dir=CACHE, trust_remote_code=True)

print('Downloading GPT-2...')
GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=CACHE)
GPT2LMHeadModel.from_pretrained('openai-community/gpt2', cache_dir=CACHE)

print('All models downloaded!')