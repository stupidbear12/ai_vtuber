import subprocess, sys
code = """
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
print('Setting up BnB config...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print('Loading model in 4-bit...')
model = AutoModelForCausalLM.from_pretrained(
    'unsloth/Meta-Llama-3.1-8B-Instruct',
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
print('Model loaded!')
print('GPU mem:', torch.cuda.memory_allocated() / 1024**3, 'GB')
"""
result = subprocess.run([sys.executable, '-u', '-c', code], capture_output=True, text=True, timeout=120)
print('STDOUT:', result.stdout[:2000])
print('STDERR:', result.stderr[:2000])
print('RC:', result.returncode)
