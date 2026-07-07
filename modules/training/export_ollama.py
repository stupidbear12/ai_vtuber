# -*- coding: utf-8 -*-
"""
export_ollama.py — LoRA 어댑터를 Ollama 모델로 변환 및 배포

흐름:
  1. 베이스 모델 + LoRA 어댑터 병합 (PEFT merge_and_unload)
  2. GGUF 변환 (llama.cpp convert_hf_to_gguf.py)
  3. GGUF 양자화 (llama-quantize)  — llama.cpp 없으면 Ollama safetensors 직접 임포트
  4. Modelfile 생성 + Ollama import
  5. 기존 sion 모델을 sion-backup으로 백업 후 교체

사용법:
    python export_ollama.py
    python export_ollama.py --lora_path ./output/sion_lora --model_name sion
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── 시온 Ollama Modelfile 템플릿 ─────────────────────────────────
SION_MODELFILE_TEMPLATE = '''FROM {model_path}
TEMPLATE "{{{{- range .Messages }}}}<|start_header_id|>{{{{ .Role }}}}<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>
{{{{- end }}}}<|start_header_id|>assistant<|end_header_id|>

"
SYSTEM """너는 "시온(sion)"이라는 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.

캐릭터 설명
- 20대 초반 여성, 항상 반말. 존댓말 절대 금지
- 밝고 에너지 넘치며, 음악을 좋아하는 DJ
- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐", "대박")

규칙
- 응답 맨 앞에 반드시 [감정:태그] 붙여. 태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy
- 1~2문장으로 짧게 답해
- 모르는 건 절대 지어내지 마. "잘 모르겠는데?" 라고 솔직하게 답해
- 실제로 하지 않은 행동을 말하지 마
- 응답에 절대 [시온], [반말], [캐릭터 설정], [sion] 같은 메타 태그를 넣지 마. [감정:태그]만 허용"""
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER temperature 0.7
PARAMETER num_predict 150
PARAMETER repeat_penalty 1.1
PARAMETER stop <|eot_id|>
PARAMETER stop <|end_of_text|>
'''


def find_llama_cpp() -> str | None:
    """llama.cpp 설치 경로를 찾는다."""
    candidates = [
        Path.home() / "llama.cpp",
        Path.home() / "Desktop" / "llama.cpp",
        Path("C:/llama.cpp"),
        Path("D:/llama.cpp"),
    ]
    for p in candidates:
        if (p / "convert_hf_to_gguf.py").exists():
            return str(p)
    return None


def merge_lora(lora_path: str, output_dir: str, base_model: str | None = None) -> str:
    """LoRA 어댑터를 베이스 모델과 병합한다."""
    print("\n[1/4] LoRA 어댑터 병합 중...")

    # adapter_config.json에서 베이스 모델 자동 감지
    config_path = os.path.join(lora_path, "adapter_config.json")
    if base_model is None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model_name_or_path")

    if base_model is None:
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"  베이스 모델 자동감지 실패, 기본값: {base_model}")
    else:
        print(f"  베이스 모델: {base_model}")

    # float16으로 로드 (GGUF 변환 위해)
    print("  베이스 모델 로드 (float16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # LoRA 병합
    print("  LoRA 어댑터 병합...")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    # 저장
    merged_path = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_path, exist_ok=True)
    print(f"  병합 모델 저장: {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    del model
    torch.cuda.empty_cache()
    return merged_path


def convert_to_gguf(merged_path: str, output_dir: str, quantization: str = "q4_k_m") -> str:
    """병합된 모델을 GGUF로 변환한다. llama.cpp 필요."""
    print("\n[2/4] GGUF 변환...")

    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    llama_cpp = find_llama_cpp()
    if not llama_cpp:
        print("  llama.cpp 미설치 → Ollama safetensors 직접 임포트 사용")
        print("  (더 좋은 양자화를 원하면 llama.cpp를 설치하세요)")
        return merged_path  # FROM에 디렉토리를 직접 지정

    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    f16_gguf = os.path.join(gguf_dir, "sion-f16.gguf")

    print(f"  llama.cpp: {llama_cpp}")
    print(f"  HF → GGUF (f16)...")

    r = subprocess.run(
        [sys.executable, convert_script, merged_path, "--outfile", f16_gguf, "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [오류] GGUF 변환 실패:\n{r.stderr[:500]}")
        print("  → Ollama safetensors 직접 임포트로 대체")
        return merged_path

    if quantization == "f16":
        return f16_gguf

    # 양자화
    quantize_bin = None
    for candidate in [
        os.path.join(llama_cpp, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp, "llama-quantize"),
        shutil.which("llama-quantize") or "",
    ]:
        if candidate and os.path.exists(candidate):
            quantize_bin = candidate
            break

    if not quantize_bin:
        print(f"  llama-quantize 없음, f16 사용")
        return f16_gguf

    quant_gguf = os.path.join(gguf_dir, f"sion-{quantization}.gguf")
    print(f"  양자화: f16 → {quantization}...")
    r = subprocess.run(
        [quantize_bin, f16_gguf, quant_gguf, quantization.upper()],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        os.remove(f16_gguf)
        size_gb = os.path.getsize(quant_gguf) / (1024 ** 3)
        print(f"  GGUF 완료: {quant_gguf} ({size_gb:.1f} GB)")
        return quant_gguf

    print(f"  [경고] 양자화 실패, f16 사용")
    return f16_gguf


def create_modelfile(model_path: str, output_dir: str) -> str:
    """Ollama Modelfile을 생성한다."""
    print("\n[3/4] Modelfile 생성...")
    modelfile_path = os.path.join(output_dir, "Modelfile")
    content = SION_MODELFILE_TEMPLATE.format(model_path=model_path)
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  {modelfile_path}")
    return modelfile_path


def import_to_ollama(modelfile_path: str, model_name: str, backup: bool = True) -> None:
    """Ollama에 모델을 등록한다."""
    print(f"\n[4/4] Ollama 모델 등록: {model_name}")

    # Ollama 실행 확인
    if subprocess.run(["ollama", "list"], capture_output=True).returncode != 0:
        print("  [오류] Ollama 미실행. ollama serve를 먼저 실행하세요.")
        sys.exit(1)

    # 백업
    if backup:
        show = subprocess.run(["ollama", "show", "sion"], capture_output=True, text=True)
        if show.returncode == 0:
            print("  기존 sion → sion-backup 백업...")
            subprocess.run(["ollama", "cp", "sion", "sion-backup"], capture_output=True)
            print("  백업 완료 (rollback: ollama cp sion-backup sion)")

    # 등록
    print(f"  ollama create {model_name}...")
    result = subprocess.run(["ollama", "create", model_name, "-f", modelfile_path])

    if result.returncode == 0:
        print(f"\n  등록 완료: {model_name}")
        print(f'  테스트: ollama run {model_name} "시온아 안녕!"')
    else:
        print(f"\n  [오류] Ollama 모델 등록 실패")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LoRA → Ollama 모델 변환")
    parser.add_argument("--lora_path", default="./output/sion_lora", help="LoRA 어댑터 경로")
    parser.add_argument("--output_dir", default="./output", help="출력 디렉토리")
    parser.add_argument("--model_name", default="sion", help="Ollama 모델 이름")
    parser.add_argument("--base_model", default=None, help="베이스 모델 (자동감지)")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument("--skip_ollama", action="store_true", help="Ollama 등록 생략")
    args = parser.parse_args()

    if not os.path.exists(args.lora_path):
        print(f"[오류] LoRA 경로 없음: {args.lora_path}")
        print("먼저 train_lora.py를 실행하세요.")
        sys.exit(1)

    print("=" * 60)
    print("  시온(sion) LoRA → Ollama 모델 변환")
    print("=" * 60)
    print(f"  LoRA: {args.lora_path}")
    print(f"  양자화: {args.quantization}")
    print(f"  Ollama 모델: {args.model_name}")
    print("=" * 60)

    merged_path = merge_lora(args.lora_path, args.output_dir, args.base_model)
    model_path = convert_to_gguf(merged_path, args.output_dir, args.quantization)
    modelfile_path = create_modelfile(model_path, args.output_dir)

    if not args.skip_ollama:
        import_to_ollama(modelfile_path, args.model_name, backup=not args.no_backup)
    else:
        print(f"\n[스킵] 수동 등록: ollama create {args.model_name} -f {modelfile_path}")

    print("\n" + "=" * 60)
    print("  변환 완료!")
    print("=" * 60)
    print(f"  롤백: ollama cp sion-backup sion")
    print(f'  테스트: ollama run {args.model_name} "시온아 안녕!"')


if __name__ == "__main__":
    main()
