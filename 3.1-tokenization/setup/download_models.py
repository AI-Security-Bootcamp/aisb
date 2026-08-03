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

# Full models. Keep this list in sync with the model ids the 3.1-3.5 exercises
# actually load: anything newer than the pinned transformers (4.57.6) supports
# will fail here with an unrecognized `model_type`.
for name in ['Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen3-4B']:
    print(f'Downloading model: {name}')
    AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, cache_dir=CACHE, trust_remote_code=True)

# GPT-2 for the distillation exercises, which use both the small and XL sizes.
for name in ['openai-community/gpt2', 'openai-community/gpt2-xl']:
    print(f'Downloading GPT-2: {name}')
    GPT2Tokenizer.from_pretrained(name, cache_dir=CACHE)
    GPT2LMHeadModel.from_pretrained(name, cache_dir=CACHE)

print('All models downloaded!')