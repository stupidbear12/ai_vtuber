# -*- coding: utf-8 -*-
"""
시온(sion) QLoRA 파인튜닝 스크립트

베이스 모델: unsloth/Meta-Llama-3.1-8B-Instruct
기본 학습 데이터: sion_combined_train.jsonl (기존 + AI Hub 병합)
기본 평가 데이터: sion_combined_eval.jsonl
출력: training/sion-finetuned/

사전 준비:
  python preprocess_aihub_emotion.py <AI_Hub.xls>
  python merge_training_data.py
  python finetune_sion.py
"""
import argparse
import json
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

from datasets import Dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(TRAINING_DIR, "finetune_combined.log")

DEFAULT_TRAIN = os.path.join(TRAINING_DIR, "sion_combined_v2.jsonl")
DEFAULT_EVAL = os.path.join(TRAINING_DIR, "sion_aihub_emotion_eval.jsonl")
FALLBACK_TRAIN = os.path.join(TRAINING_DIR, "sion_combined_clean.jsonl")
FALLBACK_EVAL = os.path.join(TRAINING_DIR, "sion_eval.jsonl")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def resolve_data_files(train_arg, eval_arg):
    train_file = train_arg or DEFAULT_TRAIN
    eval_file = eval_arg or DEFAULT_EVAL

    if not os.path.exists(train_file) and os.path.exists(FALLBACK_TRAIN):
        print(f"[warn] {os.path.basename(train_file)} 없음 -> {os.path.basename(FALLBACK_TRAIN)} 사용")
        train_file = FALLBACK_TRAIN

    if not os.path.exists(eval_file) and os.path.exists(FALLBACK_EVAL):
        print(f"[warn] {os.path.basename(eval_file)} 없음 -> {os.path.basename(FALLBACK_EVAL)} 사용")
        eval_file = FALLBACK_EVAL

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"학습 파일 없음: {train_file}\n"
            "  python preprocess_aihub_emotion.py <xls>\n"
            "  python merge_training_data.py"
        )
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"평가 파일 없음: {eval_file}")

    return train_file, eval_file


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="시온 QLoRA 파인튜닝")
    parser.add_argument("--train-file", default=None, help="학습 JSONL 경로")
    parser.add_argument("--eval-file", default=None, help="평가 JSONL 경로")
    args = parser.parse_args()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 60)
    log("시온(sion) QLoRA 파인튜닝 시작")
    log("=" * 60)
    start_time = time.time()

    BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
    OUTPUT_DIR = os.path.join(TRAINING_DIR, "sion-finetuned")
    TRAIN_FILE, EVAL_FILE = resolve_data_files(args.train_file, args.eval_file)

    log(f"학습 데이터: {TRAIN_FILE}")
    log(f"평가 데이터: {EVAL_FILE}")

    log("데이터 로드 중...")
    train_data = load_jsonl(TRAIN_FILE)
    eval_data = load_jsonl(EVAL_FILE)
    log(f"학습: {len(train_data)}건, 평가: {len(eval_data)}건")

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    log("토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("모델 로드 (4-bit 양자화)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    log(f"모델 로드 완료. GPU: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_steps = (len(train_ds) // 16) * 3
    log(f"예상 총 학습 스텝: {total_steps}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_length=512,
        logging_steps=20,
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
        packing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        torch_compile=False,
    )

    log("SFTTrainer 시작...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log("학습 시작...")
    trainer.train()

    log(f"모델 저장: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    eval_result = trainer.evaluate()
    log(f"최종 eval loss: {eval_result.get('eval_loss', 'N/A')}")

    elapsed = (time.time() - start_time) / 60
    log("=" * 60)
    log(f"파인튜닝 완료! (소요: {elapsed:.1f}분)")
    log(f"모델 저장 위치: {OUTPUT_DIR}")
    log("다음: python export_to_ollama.py")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"파인튜닝 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
