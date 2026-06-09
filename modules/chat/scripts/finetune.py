#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시온(sion) 캐릭터 QLoRA 파인튜닝
  베이스 모델 : meta-llama/Meta-Llama-3.1-8B-Instruct
  데이터셋    : emeth_dataset.jsonl  (460개 ShareGPT 대화쌍)
  GPU 타깃    : RTX 4060 Ti  8 GB VRAM

사전 준비:
  1) conda activate sion_finetune
  2) huggingface-cli login  (Llama 3.1 라이선스 동의 후 토큰 입력)
     https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

실행:
  py -m modules.chat.scripts.finetune
  또는
  cd workspace2/ai_vtuber && py modules/chat/scripts/finetune.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

# ── 경로 ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
DATA_PATH   = SCRIPT_DIR.parent / "data" / "emeth_dataset.jsonl"
OUTPUT_DIR  = SCRIPT_DIR.parent / "output" / "sion-llama31-qlora"

# ── 모델 ─────────────────────────────────────────────────────────────
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ── QLoRA 하이퍼파라미터 ─────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
# Llama 3.1 8B attention + MLP 레이어 전체 타깃
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── 학습 하이퍼파라미터  (8 GB VRAM 최적화) ──────────────────────────
MAX_SEQ_LENGTH = 512    # 긴 시퀀스가 VRAM을 많이 잡음 → 512로 제한
NUM_EPOCHS     = 3
BATCH_SIZE     = 1      # per-device; VRAM 안전 마진 확보
GRAD_ACCUM     = 8      # 실효 배치 크기 = 8
LEARNING_RATE  = 2e-4
WARMUP_RATIO   = 0.05

# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── 유틸 ─────────────────────────────────────────────────────────────

def check_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU가 감지되지 않았습니다.\n"
            "CUDA 드라이버와 PyTorch(+CUDA) 설치를 확인하세요."
        )
    name   = torch.cuda.get_device_name(0)
    vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {name}  |  VRAM: {vram:.1f} GB")
    if vram < 6.5:
        raise RuntimeError(f"VRAM {vram:.1f} GB 부족 — 최소 8 GB 필요")
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        log.warning(
            "HF_TOKEN 환경변수가 없습니다.\n"
            "Llama 3.1은 게이티드(gated) 모델입니다. 접근 권한이 없으면 다운로드가 실패합니다.\n"
            "  → https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct 에서 라이선스 동의 후\n"
            "  → huggingface-cli login  또는  set HF_TOKEN=<your_token>"
        )


# ── 데이터 ───────────────────────────────────────────────────────────

def load_raw_dataset():
    from datasets import Dataset

    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"데이터셋 로드 완료: {len(records)} 개")
    return Dataset.from_list(records)


def preprocess(dataset, tokenizer):
    """
    ShareGPT conversations → Llama 3.1 채팅 템플릿 text

    Llama 3.1 Instruct는 system role을 네이티브로 지원하므로
    system 메시지를 별도 role로 유지합니다.
    """

    def to_messages(ex):
        messages    = []
        system_text = None
        for turn in ex["conversations"]:
            role, content = turn["from"], turn["value"]
            if role == "system":
                system_text = content
            elif role == "human":
                if system_text and not messages:
                    messages.append({"role": "system", "content": system_text})
                    system_text = None
                messages.append({"role": "user", "content": content})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": content})
        return {"messages": messages}

    def apply_template(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(to_messages, remove_columns=dataset.column_names)
    dataset = dataset.map(apply_template, remove_columns=["messages"])
    log.info(f"전처리 완료: {len(dataset)} 개")

    # 길이 분포 확인
    sample_lengths = [len(tokenizer(ex["text"])["input_ids"]) for ex in dataset.select(range(min(50, len(dataset))))]
    avg_len = sum(sample_lengths) / len(sample_lengths)
    max_len = max(sample_lengths)
    log.info(f"토큰 길이 (샘플 50개 기준) — 평균: {avg_len:.0f}  최대: {max_len}")
    if max_len > MAX_SEQ_LENGTH:
        log.warning(
            f"일부 샘플이 MAX_SEQ_LENGTH({MAX_SEQ_LENGTH})를 초과합니다. "
            "초과 부분은 잘립니다."
        )

    return dataset


# ── 모델 로드 ─────────────────────────────────────────────────────────

def load_model_4bit(model_id: str):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — QLoRA 논문 권장
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # 이중 양자화로 추가 메모리 절약
    )

    log.info(f"토크나이저 로드: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # causal LM은 right-padding이 안전

    log.info(f"모델 로드 중: {model_id}  (최초 실행 시 ~16 GB 다운로드)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",    # flash_attention_2는 Windows에서 불안정
        trust_remote_code=True,
    )
    log.info("모델 로드 완료")
    return model, tokenizer


# ── QLoRA 적용 ────────────────────────────────────────────────────────

def apply_qlora(model):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # gradient checkpointing 활성화 후 k-bit 학습 준비
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── 학습 ─────────────────────────────────────────────────────────────

def run_training(model, tokenizer, dataset) -> Path:
    from transformers import TrainingArguments

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        # 메모리 최적화
        bf16=True,                             # RTX 4060 Ti bfloat16 지원
        tf32=True,                             # Ada Lovelace TF32 최적화
        optim="paged_adamw_8bit",              # paged optimizer로 VRAM 절약
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        # 체크포인트 & 로깅
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
    )

    # TRL 버전별 호환성 처리
    try:
        from trl import SFTTrainer, SFTConfig
        # TRL >= 0.12: SFTConfig 사용 가능
        sft_config = SFTConfig(
            **vars(training_args),
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    except (ImportError, TypeError):
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=False,
        )

    total_steps = len(trainer.get_train_dataloader()) * NUM_EPOCHS
    log.info("=" * 60)
    log.info(f"파인튜닝 시작")
    log.info(f"  샘플 수     : {len(dataset)}")
    log.info(f"  에폭        : {NUM_EPOCHS}")
    log.info(f"  실효 배치   : {BATCH_SIZE * GRAD_ACCUM}")
    log.info(f"  총 스텝     : {total_steps}")
    log.info(f"  예상 시간   : {total_steps * 25 // 60}~{total_steps * 35 // 60} 분 (RTX 4060 Ti 기준)")
    log.info("=" * 60)

    trainer.train()

    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"LoRA 어댑터 저장 완료: {final_dir}")
    return final_dir


# ── 메인 ─────────────────────────────────────────────────────────────

def main():
    check_environment()

    log.info("[1/4] 데이터셋 로드")
    dataset = load_raw_dataset()

    log.info("[2/4] 모델 로드 (4-bit NF4 양자화)")
    model, tokenizer = load_model_4bit(BASE_MODEL_ID)

    log.info("[3/4] 데이터 전처리 (ShareGPT → Llama 3.1 채팅 템플릿)")
    dataset = preprocess(dataset, tokenizer)

    log.info("[4/4] QLoRA 파인튜닝 시작")
    model    = apply_qlora(model)
    final_dir = run_training(model, tokenizer, dataset)

    log.info("\n" + "=" * 60)
    log.info("파인튜닝 완료!")
    log.info(f"  저장 위치: {final_dir}")
    log.info("\n[다음 단계]")
    log.info("  1) LoRA 병합 + 내보내기:")
    log.info("       py modules/chat/scripts/merge_and_export.py")
    log.info("  2) GGUF 변환 (llama.cpp 필요):")
    log.info("       merge_and_export.py 실행 후 출력된 명령어 참고")
    log.info("  3) Ollama 등록:")
    log.info("       ollama create sion -f modules/chat/scripts/Modelfile")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
