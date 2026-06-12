#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 어댑터 → 베이스 모델 병합 → GGUF 변환 준비

실행:
  py modules/chat/scripts/merge_and_export.py

요구사항:
  - finetune.py 완료 (modules/chat/output/sion-llama31-qlora/final/ 존재)
  - RAM 32 GB 이상 권장 (CPU에서 병합, Llama 3.1 8B는 ~16 GB 필요)
"""

import logging
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

SCRIPT_DIR   = Path(__file__).parent.resolve()
LORA_DIR     = SCRIPT_DIR.parent / "output" / "sion-llama31-qlora" / "final"
MERGED_DIR   = SCRIPT_DIR.parent / "output" / "sion-llama31-merged"
GGUF_OUT     = SCRIPT_DIR.parent / "output" / "sion-llama31-q4_k_m.gguf"
BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"


def merge_lora() -> None:
    if not LORA_DIR.exists():
        raise FileNotFoundError(
            f"LoRA 어댑터를 찾을 수 없습니다: {LORA_DIR}\n"
            "finetune.py를 먼저 실행하세요."
        )

    log.info("베이스 모델 CPU 로드 중 (bf16) — RAM 약 16~20 GB 필요...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",            # VRAM 부족 방지: CPU에서 병합
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    log.info(f"LoRA 어댑터 적용: {LORA_DIR}")
    model = PeftModel.from_pretrained(model, str(LORA_DIR), torch_dtype=torch.bfloat16)

    log.info("LoRA 가중치 병합 중...")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"병합된 모델 저장: {MERGED_DIR}")
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    log.info("병합 완료!")


def print_gguf_guide() -> None:
    guide = f"""
{'=' * 65}
  GGUF 변환 가이드
{'=' * 65}

[1단계] llama.cpp 준비 (미설치 시)
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  pip install -r requirements.txt

[2단계] HuggingFace → GGUF 변환
  방법 A) 직접 Q4_K_M 변환 (빠름, 권장):
    python llama.cpp/convert_hf_to_gguf.py \\
      {MERGED_DIR} \\
      --outtype q4_k_m \\
      --outfile {GGUF_OUT}

  방법 B) f16 변환 후 별도 양자화 (더 세밀한 제어):
    python llama.cpp/convert_hf_to_gguf.py \\
      {MERGED_DIR} \\
      --outtype f16 \\
      --outfile {SCRIPT_DIR.parent}/output/emeth-gemma3-f16.gguf

    llama.cpp/llama-quantize.exe \\
      {SCRIPT_DIR.parent}/output/emeth-gemma3-f16.gguf \\
      {GGUF_OUT} \\
      Q4_K_M

[3단계] Modelfile에서 경로 확인 후 Ollama 등록
  Modelfile 수정:
    FROM ./sion-llama31-q4_k_m.gguf
    →  실제 GGUF 파일의 절대 경로 또는 상대 경로로 수정

  Ollama 등록:
    cd {SCRIPT_DIR}
    ollama create sion -f Modelfile

  테스트:
    ollama run emeth "안녕! 오늘 기분 어때?"

{'=' * 65}
  llama.cpp 없이 간편하게: gguf-py 패키지 사용
{'=' * 65}
  pip install gguf
  python -c "
from gguf import GGUFWriter
# gguf 패키지는 변환 유틸리티가 아닌 쓰기 API입니다.
# 변환에는 llama.cpp 스크립트를 사용하세요.
print('llama.cpp를 사용해 변환하세요.')
"
{'=' * 65}
"""
    print(guide)


def main() -> None:
    merge_lora()
    print_gguf_guide()


if __name__ == "__main__":
    main()
