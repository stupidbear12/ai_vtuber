"""Diagnostic: find the exact point where training hangs."""
import json, os, time, sys
os.environ["PYTHONUNBUFFERED"] = "1"

# datasets BEFORE torch
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load data
log("Loading data...")
train_data = []
with open(r"C:\Users\thtgg\workspace2\ai_vtuber\training\sion_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            train_data.append(json.loads(line))
log(f"Loaded {len(train_data)} samples")

# Use only first N samples for diagnosis
N = 100
train_data = train_data[:N]
log(f"Using {N} samples for diagnosis")

# Tokenizer
log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model
log("Loading model (4-bit)...")
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
log(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === TEST 1: SFTTrainer with messages (same as working 20-sample test) ===
log("=== TEST: SFTTrainer with 100 real samples ===")
train_ds = Dataset.from_list(train_data)

training_args = SFTConfig(
    output_dir="./training/_diag_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # keep simple
    max_length=512,
    logging_steps=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    packing=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    max_steps=20,  # only 20 steps
    torch_compile=False,
)

log("Creating SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

log("Calling trainer.train()...")
t0 = time.time()
trainer.train()
log(f"Training done in {time.time()-t0:.1f}s")
log("SUCCESS!")
