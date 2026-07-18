"""
Test training with REAL data using the same subprocess pattern that proved working.
Uses SFTTrainer with messages format (same as the 20-sample test that succeeded).
"""
import subprocess, sys, os

code = r"""
import json, os, time
os.environ["PYTHONUNBUFFERED"] = "1"

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Load data
print("Loading data...", flush=True)
train_data = []
with open(r"C:\Users\thtgg\workspace2\ai_vtuber\training\sion_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            train_data.append(json.loads(line))

eval_data = []
with open(r"C:\Users\thtgg\workspace2\ai_vtuber\training\sion_eval.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            eval_data.append(json.loads(line))

print(f"Train: {len(train_data)}, Eval: {len(eval_data)}", flush=True)
train_ds = Dataset.from_list(train_data)
eval_ds = Dataset.from_list(eval_data)

# Tokenizer
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model
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
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training config
OUTPUT = r"C:\Users\thtgg\workspace2\ai_vtuber\training\sion-finetuned"
training_args = SFTConfig(
    output_dir=OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    max_length=512,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    dataloader_pin_memory=False,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print(f"Starting training... Total steps: ~{len(train_ds)//16*3}", flush=True)
start = time.time()
trainer.train()
elapsed = (time.time() - start) / 60
print(f"Training done in {elapsed:.1f} min", flush=True)

trainer.save_model(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print(f"Model saved to {OUTPUT}", flush=True)

eval_result = trainer.evaluate()
print(f"Final eval loss: {eval_result.get('eval_loss', 'N/A')}", flush=True)
"""

# Run with no timeout (will take hours)
log_path = os.path.join(os.path.dirname(__file__), "finetune_capture.log")

proc = subprocess.Popen(
    [sys.executable, '-u', '-c', code],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

# Read and write output line by line
with open(log_path, "w", encoding="utf-8") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()
        # Also print last few chars for monitoring

proc.wait()
with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"\n=== EXIT CODE: {proc.returncode} ===\n")
