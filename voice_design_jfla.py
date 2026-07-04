# -*- coding: utf-8 -*-
"""
voice_design_jfla.py — Chatterbox 보이스 클로닝 테스트 (JFla 스타일)

사용법:
  python voice_design_jfla.py                     # 기본 TTS 테스트 (클로닝 없음)
  python voice_design_jfla.py --ref path/to/jfla.wav  # JFla 레퍼런스로 클로닝

레퍼런스 오디오는 3~10초 분량의 고품질 WAV 파일을 권장합니다.
생성된 샘플은 voice_samples/ 폴더에 저장됩니다.
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

output_dir = Path("voice_samples")
output_dir.mkdir(exist_ok=True)

TEST_TEXTS = {
    "test_korean_happy.wav": (
        "안녕하세요 여러분! 오늘도 제 방송에 와주셔서 감사해요~ "
        "오늘은 정말 신나는 노래를 준비했어요. 같이 즐겨봐요!",
        "happy",
    ),
    "test_korean_calm.wav": (
        "오늘 하루도 정말 수고하셨어요. 잠깐 쉬어가면서 "
        "따뜻한 음악 들어보실래요? 제가 곁에 있을게요.",
        "calm",
    ),
    "test_korean_excited.wav": (
        "와 진짜?! 너무 좋아! 이 노래 제가 엄청 좋아하는 곡인데, "
        "커버하게 돼서 너무 설레요! 들어봐요!",
        "excited",
    ),
}

EMOTION_EXAGGERATION = {
    "happy": 0.60,
    "calm": 0.10,
    "excited": 0.90,
}


def run(ref_audio: str | None):
    print("Chatterbox TTS - JFla 스타일 보이스 테스트")
    print(f"레퍼런스: {ref_audio or '없음 (기본 음성)'}")
    print()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU 모드")

    from chatterbox.tts import ChatterboxTTS
    import torchaudio

    print("모델 로딩...")
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"모델 로드 완료 ({time.time()-t0:.1f}s)\n")

    for fname, (text, emotion) in TEST_TEXTS.items():
        exaggeration = EMOTION_EXAGGERATION.get(emotion, 0.5)
        print(f"[{emotion}] {fname}")
        print(f"  텍스트: {text[:40]}...")

        t0 = time.time()
        wav = model.generate(
            text=text,
            audio_prompt_path=ref_audio,
            exaggeration=exaggeration,
            cfg_weight=0.6 if ref_audio else 0.5,
        )
        elapsed = time.time() - t0
        duration = wav.shape[-1] / model.sr

        out_path = output_dir / fname
        torchaudio.save(str(out_path), wav.cpu(), model.sr)
        print(f"  저장: {out_path} | {duration:.1f}s 오디오, {elapsed:.1f}s 소요 (RTF: {elapsed/duration:.2f}x)")
        print()

    print("완료! 생성된 샘플:")
    for p in sorted(output_dir.glob("*.wav")):
        print(f"  {p}")
    print()
    if ref_audio:
        print(f"JFla 스타일 클로닝 적용됨: {ref_audio}")
        print("voice 서버에 등록하려면:")
        print(f"  1. 파일을 modules/voice/data/reference_audio/ 에 복사")
        print(f"  2. POST /voice/set-default {{\"reference_name\": \"jfla\"}}")
    else:
        print("레퍼런스 오디오 없이 기본 음성으로 생성됨.")
        print("JFla 스타일 클로닝: --ref path/to/jfla.wav 옵션 사용")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatterbox JFla 스타일 보이스 테스트")
    parser.add_argument("--ref", default=None, help="레퍼런스 오디오 WAV 파일 경로 (선택)")
    args = parser.parse_args()

    ref = args.ref
    if ref and not Path(ref).exists():
        print(f"ERROR: 레퍼런스 파일 없음: {ref}")
        sys.exit(1)

    run(ref)
