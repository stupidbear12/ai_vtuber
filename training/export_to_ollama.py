# -*- coding: utf-8 -*-
"""
시온(sion) LoRA → GGUF 변환 + Ollama 등록 스크립트

1. LoRA 어댑터를 베이스 모델에 병합 (CPU, float16)
2. llama.cpp convert_hf_to_gguf.py로 GGUF 변환
3. (선택) llama-quantize로 Q4_K_M 양자화
4. Ollama Modelfile 생성 + 등록
"""
import os, sys, subprocess, shutil, time, glob

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(TRAINING_DIR, "sion-finetuned")
MERGED_DIR = os.path.join(TRAINING_DIR, "sion-merged")
GGUF_DIR = os.path.join(TRAINING_DIR, "sion-gguf")
LLAMA_CPP_DIR = os.path.join(TRAINING_DIR, "llama.cpp")

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = "sion"  # will overwrite existing sion in Ollama

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_llama_cpp():
    """llama.cpp repo clone (convert 스크립트만 필요)"""
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if os.path.exists(convert_script):
        log("  llama.cpp 이미 존재")
        return convert_script

    log("  llama.cpp 클론 중 (shallow)...")
    subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/ggerganov/llama.cpp.git",
        LLAMA_CPP_DIR
    ], check=True)

    # Install requirements for convert script
    req_file = os.path.join(LLAMA_CPP_DIR, "requirements", "requirements-convert_hf_to_gguf.txt")
    if os.path.exists(req_file):
        log("  convert 스크립트 의존성 설치...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file],
                       check=True, capture_output=True)

    return convert_script


def step1_merge_lora():
    """LoRA 어댑터를 베이스 모델에 병합"""
    if os.path.exists(MERGED_DIR) and os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        log("Step 1: 이미 병합된 모델 존재, 건너뜀")
        return True

    log("Step 1: LoRA 어댑터 병합 중...")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    log("  베이스 모델 로드 (float16, CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    log("  LoRA 어댑터 로드...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)

    log("  병합 중...")
    merged = model.merge_and_unload()

    log(f"  저장: {MERGED_DIR}")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(MERGED_DIR)

    # Free memory
    del merged, model, base_model
    import gc; gc.collect()

    log("  병합 완료!")
    return True


def step2_convert_gguf():
    """병합된 모델을 GGUF F16으로 변환"""
    log("Step 2: GGUF 변환 중...")
    os.makedirs(GGUF_DIR, exist_ok=True)

    gguf_f16 = os.path.join(GGUF_DIR, "sion-f16.gguf")
    gguf_q4 = os.path.join(GGUF_DIR, "sion-q4_k_m.gguf")

    # 이미 양자화된 파일이 있으면 건너뜀
    if os.path.exists(gguf_q4):
        log(f"  이미 존재: {gguf_q4}")
        return gguf_q4

    # Get convert script
    convert_script = ensure_llama_cpp()

    # Step 2a: HF → GGUF F16
    if not os.path.exists(gguf_f16):
        log("  HF → GGUF (F16) 변환...")
        cmd = [
            sys.executable, convert_script,
            MERGED_DIR,
            "--outfile", gguf_f16,
            "--outtype", "f16",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  변환 실패: {result.stderr[-500:]}")
            return None
        log(f"  F16 GGUF 생성: {gguf_f16}")
    else:
        log(f"  F16 이미 존재: {gguf_f16}")

    # Step 2b: F16 → Q4_K_M 양자화
    # llama-quantize 바이너리 필요. 없으면 F16 그대로 사용
    quantize_bin = None
    for name in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]:
        p = shutil.which(name)
        if p:
            quantize_bin = p
            break
        # Also check llama.cpp build dir
        for subdir in ["", "Release", "Debug"]:
            bp = os.path.join(LLAMA_CPP_DIR, "build", "bin", subdir, name)
            if os.path.exists(bp):
                quantize_bin = bp
                break
        if quantize_bin:
            break

    if quantize_bin:
        log(f"  Q4_K_M 양자화 중 ({quantize_bin})...")
        cmd = [quantize_bin, gguf_f16, gguf_q4, "q4_k_m"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"  양자화 완료: {gguf_q4}")
            return gguf_q4
        else:
            log(f"  양자화 실패, F16 사용: {result.stderr[-300:]}")
            return gguf_f16
    else:
        log("  llama-quantize 없음. F16 GGUF를 그대로 사용합니다.")
        log("  (약 16GB. Q4_K_M 양자화하면 ~5GB로 줄어듭니다)")
        return gguf_f16


def step3_register_ollama(gguf_path):
    """Ollama Modelfile 생성 + 등록"""
    log("Step 3: Ollama 등록...")

    modelfile_path = os.path.join(GGUF_DIR, "Modelfile")

    # Read system prompt from existing chat_engine or use default
    system_prompt = """너는 "시온(sion)"이라는 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.

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

    # Try to load system prompt from chat_engine.py
    chat_engine_path = os.path.join(TRAINING_DIR, "..", "chat_engine.py")
    if os.path.exists(chat_engine_path):
        try:
            with open(chat_engine_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract SYSTEM_PROMPT if defined
            import re
            match = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', content, re.DOTALL)
            if match:
                system_prompt = match.group(1).strip()
                log("  chat_engine.py에서 시스템 프롬프트 로드")
        except Exception:
            pass

    modelfile_content = f'''FROM {gguf_path}

TEMPLATE """{{{{- range .Messages }}}}<|start_header_id|>{{{{ .Role }}}}<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>
{{{{- end }}}}<|start_header_id|>assistant<|end_header_id|>

"""

SYSTEM """{system_prompt}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 150
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
'''

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    log(f"  Modelfile 생성: {modelfile_path}")

    # Register (overwrite existing sion)
    log(f"  ollama create {MODEL_NAME}...")
    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", modelfile_path],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        log(f"  Ollama 등록 완료! 모델: {MODEL_NAME}")
        return True
    else:
        log(f"  Ollama 등록 실패: {result.stderr}")
        return False


def main():
    log("=" * 60)
    log("시온 파인튜닝 모델 → Ollama 배포")
    log("=" * 60)

    # Check finetuned model exists
    adapter_cfg = os.path.join(FINETUNED_DIR, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        log(f"오류: {FINETUNED_DIR}에 파인튜닝된 모델이 없습니다.")
        log("학습이 완료된 후 실행하세요.")
        sys.exit(1)

    t0 = time.time()

    # Step 1: Merge LoRA → full model
    step1_merge_lora()

    # Step 2: Convert to GGUF
    gguf_path = step2_convert_gguf()

    if gguf_path and os.path.exists(gguf_path):
        # Step 3: Register with Ollama
        step3_register_ollama(gguf_path)

        elapsed = (time.time() - t0) / 60
        log("=" * 60)
        log(f"배포 완료! (소요: {elapsed:.1f}분)")
        log(f"테스트: ollama run {MODEL_NAME}")
        log("=" * 60)
    else:
        log("GGUF 변환 실패. 수동 변환이 필요합니다.")
        log(f"병합된 모델: {MERGED_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
