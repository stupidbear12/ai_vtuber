# -*- coding: utf-8 -*-
"""
train_lora.py - PEFT QLoRA로 시온 모델 파인튜닝

베이스 모델: Meta-Llama-3.1-8B-Instruct (4-bit 양자화)
학습 방식: QLoRA (r=16, alpha=32) via PEFT + TRL
GPU: RTX 4060 Ti 16GB 최적화

사용법:
    python train_lora.py
    python train_lora.py --epochs 3 --batch_size 2 --lr 2e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ── 기본 설정 ────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# LoRA 적용 대상 레이어
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_training_data(data_path: str) -> Dataset:
    """학습 데이터를 로드한다."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[데이터] {len(data):,}개 샘플 로드: {data_path}")
    return Dataset.from_list(data)


def formatting_prompts_func(examples, tokenizer):
    """ShareGPT 형식의 대화를 토크나이저 chat template으로 변환한다."""
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        messages = []
        for turn in convo:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="시온 LoRA 파인튜닝")
    parser.add_argument(
        "--base_model", default=DEFAULT_BASE_MODEL,
        help=f"베이스 모델 (기본: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument("--data_dir", default="./data", help="학습 데이터 디렉토리")
    parser.add_argument("--output_dir", default="./output", help="체크포인트 저장 디렉토리")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 (기본: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="배치 사이즈 (기본: 2)")
    parser.add_argument("--grad_accum", type=int, default=8, help="그래디언트 누적 스텝 (기본: 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="학습률 (기본: 2e-4)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="워밍업 비율")
    parser.add_argument("--save_steps", type=int, default=100, help="체크포인트 저장 간격")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH,
                        help=f"최대 시퀀스 길이 (기본: {MAX_SEQ_LENGTH})")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="체크포인트에서 재개 (경로)")
    parser.add_argument("--use_eval", action="store_true",
                        help="eval.json 사용 (기본: 미사용)")
    args = parser.parse_args()

    print("=" * 60)
    print("  시온(sion) LoRA 파인튜닝 - PEFT QLoRA")
    print("=" * 60)
    print(f"  베이스 모델: {args.base_model}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  학습: epochs={args.epochs}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}")
    print(f"  학습률: {args.lr}, 워밍업: {args.warmup_ratio}")
    print(f"  시퀀스 길이: {args.max_seq_length}")
    if args.resume_from:
        print(f"  체크포인트 재개: {args.resume_from}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # ── 1. 4-bit 양자화 모델 로드 ────────────────────────────────
    print("\n[1/4] 모델 로드 중 (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. LoRA 어댑터 추가 ──────────────────────────────────────
    print("[2/4] LoRA 어댑터 추가 중...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  학습 파라미터: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── 3. 데이터 로드 및 변환 ───────────────────────────────────
    print("[3/4] 학습 데이터 로드 중...")
    train_path = os.path.join(args.data_dir, "train.json")
    eval_path = os.path.join(args.data_dir, "eval.json")

    if not os.path.exists(train_path):
        print(f"[오류] {train_path} 파일을 찾을 수 없습니다.")
        print("먼저 prepare_data.py를 실행하세요.")
        sys.exit(1)

    train_dataset = load_training_data(train_path)
    eval_dataset = None
    if args.use_eval and os.path.exists(eval_path):
        eval_dataset = load_training_data(eval_path)

    # 포맷 변환
    train_dataset = train_dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: formatting_prompts_func(x, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    # ── 4. 학습 ──────────────────────────────────────────────────
    print("[4/4] 학습 시작...")
    os.makedirs(args.output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT-specific
        dataset_text_field="text",
        max_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    stats = trainer.train(resume_from_checkpoint=args.resume_from)

    print("\n" + "=" * 60)
    print("  학습 완료!")
    print("=" * 60)
    print(f"  총 스텝: {stats.global_step}")
    print(f"  학습 손실: {stats.training_loss:.4f}")
    print(f"  학습 시간: {stats.metrics['train_runtime']:.0f}초")

    # LoRA 어댑터 저장
    lora_path = os.path.join(args.output_dir, "sion_lora")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\n  LoRA 저장: {lora_path}")

    # 학습 통계 저장
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_seq_length": args.max_seq_length,
            "total_steps": stats.global_step,
            "final_loss": stats.training_loss,
            "train_runtime_sec": stats.metrics["train_runtime"],
            "trainable_params": trainable,
            "total_params": total,
        }, f, indent=2)

    print(f"\n다음 단계: python export_ollama.py --lora_path {lora_path}")


if __name__ == "__main__":
    main()
