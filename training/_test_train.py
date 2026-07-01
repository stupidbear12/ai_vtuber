"""Minimal training test to find the crash point."""
import subprocess, sys

code = r"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (4-bit)...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

model = prepare_model_for_kbit_training(model)
print("Prepared for kbit training", flush=True)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA applied", flush=True)

# Minimal dataset
train_data = [{"messages": [
    {"role": "system", "content": "You are sion."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "[happy] hi!"},
]} for _ in range(20)]
train_ds = Dataset.from_list(train_data)

print("Setting up trainer...", flush=True)
training_args = SFTConfig(
    output_dir="./training/_test_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_length=256,
    logging_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
print("Trainer created. Starting training...", flush=True)
trainer.train()
print("TRAINING COMPLETED!", flush=True)
"""

result = subprocess.run([sys.executable, '-u', '-c', code], capture_output=True, text=True, timeout=180)
print("STDOUT:", result.stdout[-3000:])
print("STDERR:", result.stderr[-2000:])
print("RC:", result.returncode)
