# -*- coding: utf-8 -*-
"""
시온 QLoRA 전체 파이프라인
  1. (선택) AI Hub 전처리  preprocess_aihub_emotion.py <xls>
  2. 데이터 병합         merge_training_data.py
  3. QLoRA 학습          finetune_sion.py
  4. GGUF + Ollama       export_to_ollama.py
"""
import argparse
import os
import subprocess
import sys
import time

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(TRAINING_DIR, "pipeline.log")


def run_step(label, script, extra_args=None):
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] {label}")
    print(f"{'='*60}\n", flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === {label} ===\n")

    cmd = [sys.executable, script] + (extra_args or [])
    result = subprocess.run(cmd, cwd=TRAINING_DIR)

    if result.returncode != 0:
        print(f"\n[FAIL] {label} 실패 (exit {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n[OK] {label} 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="시온 파인튜닝 파이프라인")
    parser.add_argument("--aihub-xls", default=None,
                        help="AI Hub 엑셀 경로 (지정 시 전처리부터 실행)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="merge_training_data.py 건너뛰기")
    args = parser.parse_args()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"파이프라인 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    t0 = time.time()

    if args.aihub_xls:
        run_step("0단계: AI Hub 전처리", os.path.join(TRAINING_DIR, "preprocess_aihub_emotion.py"),
                 [args.aihub_xls])

    if not args.skip_merge:
        run_step("1단계: 데이터 병합", os.path.join(TRAINING_DIR, "merge_training_data.py"))

    run_step("2단계: QLoRA 파인튜닝", os.path.join(TRAINING_DIR, "finetune_sion.py"))
    run_step("3단계: GGUF 변환 + Ollama 등록", os.path.join(TRAINING_DIR, "export_to_ollama.py"))

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"전체 완료! (약 {elapsed:.1f}분)")
    print("테스트: ollama run sion")
    print(f"{'='*60}")
