"""
tts_engine.py - emeth RVC 기반 TTS 엔진 (ffmpeg 불필요 버전)

파이프라인:
  텍스트 → [pause] → SSML 변환 → Edge TTS → 단일 mp3 → [RVC] → emeth 목소리

의존성:
  pip install edge-tts rvc-python
"""

import asyncio
import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path

from config_tts import (
    BASE_VOICE, BASE_VOICE_RATE, BASE_VOICE_PITCH,
    RVC_ENABLED, RVC_MODEL_DIR, RVC_MODEL_NAME,
    F0_UP_KEY, F0_METHOD, INDEX_RATE, FILTER_RADIUS,
    RMS_MIX_RATE, PROTECT,
    OUTPUT_DIR, TEMP_DIR, PAUSE_DURATION_MS,
)


# ─────────────────────────────────────────────────────────────
# 텍스트 전처리 — [pause] → SSML <break>
# ─────────────────────────────────────────────────────────────

def text_to_ssml(text: str) -> str:
    """[pause] 태그를 SSML <break> 태그로 변환하여 단일 SSML 문자열 반환"""
    # [pause] → <break time='800ms'/>
    converted = re.sub(
        r'\[pause\]',
        f"<break time='{PAUSE_DURATION_MS}ms'/>",
        text,
        flags=re.IGNORECASE
    )
    return f"<speak>{converted}</speak>"


# ─────────────────────────────────────────────────────────────
# Edge TTS 생성
# ─────────────────────────────────────────────────────────────

async def _edge_tts_generate(text: str, output_path: str) -> None:
    """Edge TTS로 텍스트(또는 SSML) → 음성 파일 생성"""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts 미설치: pip install edge-tts")

    # [pause] 태그가 있으면 SSML로 변환
    has_pause = bool(re.search(r'\[pause\]', text, re.IGNORECASE))
    if has_pause:
        ssml_text = text_to_ssml(text)
        communicate = edge_tts.Communicate(
            text=ssml_text,
            voice=BASE_VOICE,
            rate=BASE_VOICE_RATE,
            pitch=BASE_VOICE_PITCH,
        )
    else:
        communicate = edge_tts.Communicate(
            text=text,
            voice=BASE_VOICE,
            rate=BASE_VOICE_RATE,
            pitch=BASE_VOICE_PITCH,
        )
    await communicate.save(output_path)


def generate_base_audio(text: str, output_path: str) -> str:
    """텍스트를 Edge TTS로 변환하여 파일에 저장 (스레드 안전 동기 래퍼)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_edge_tts_generate(text, output_path))
    finally:
        loop.close()
    return output_path


# ─────────────────────────────────────────────────────────────
# RVC 변환
# ─────────────────────────────────────────────────────────────

def find_rvc_model():
    """voice_models/ 에서 .pth 모델과 .index 파일 탐색"""
    model_dir = Path(RVC_MODEL_DIR)
    if not model_dir.exists():
        return None, None
    pth_files = list(model_dir.glob(f"{RVC_MODEL_NAME}*.pth")) or list(model_dir.glob("*.pth"))
    if not pth_files:
        return None, None
    model_path = str(pth_files[0])
    index_files = list(model_dir.glob("*.index"))
    index_path = str(index_files[0]) if index_files else ""
    return model_path, index_path


def convert_with_rvc(input_path: str, output_path: str) -> bool:
    """RVC로 음성 파일을 emeth 목소리로 변환"""
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        print("[경고] rvc-python 미설치. Edge TTS 음성을 그대로 사용합니다.")
        return False

    model_path, index_path = find_rvc_model()
    if not model_path:
        print(f"[경고] voice_models/ 에 .pth 파일 없음. Edge TTS 음성을 그대로 사용합니다.")
        return False

    try:
        print(f"  [RVC] 모델 로드: {Path(model_path).name}")
        rvc = RVCInference(models_path=RVC_MODEL_DIR)
        rvc.load_model(Path(model_path).stem)
        rvc.infer(
            input_path=input_path,
            output_path=output_path,
            f0_up_key=F0_UP_KEY,
            f0_method=F0_METHOD,
            index_rate=INDEX_RATE,
            filter_radius=FILTER_RADIUS,
            rms_mix_rate=RMS_MIX_RATE,
            protect=PROTECT,
            index_path=index_path or "",
        )
        print(f"  [RVC] 변환 완료: {Path(output_path).name}")
        return True
    except Exception as e:
        print(f"[경고] RVC 변환 실패 ({e}). Edge TTS 음성을 그대로 사용합니다.")
        return False


# ─────────────────────────────────────────────────────────────
# TTSEngine
# ─────────────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self.use_rvc = RVC_ENABLED
        model_path, _ = find_rvc_model()
        if self.use_rvc and not model_path:
            print("[안내] RVC 모델 없음 → Edge TTS 모드로 실행합니다.")
            print(f"  모델 추가: {RVC_MODEL_DIR}")
            self.use_rvc = False

    def synthesize(self, text: str, output_filename: str = None) -> str:
        """
        텍스트 → 최종 음성 파일 생성
        [pause] 태그는 SSML로 처리 (ffmpeg 불필요)
        """
        if not output_filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"emeth_{ts}.mp3"

        # 출력 경로 (RVC 있으면 임시 → 변환 후 최종)
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if self.use_rvc:
            # 1. Edge TTS → 임시 파일
            os.makedirs(TEMP_DIR, exist_ok=True)
            tmp_path = os.path.join(TEMP_DIR, f"base_{datetime.now().strftime('%H%M%S%f')}.mp3")
            print(f"  [Edge TTS] 음성 생성 중...")
            generate_base_audio(text, tmp_path)
            # 2. RVC → 최종 파일
            success = convert_with_rvc(tmp_path, output_path)
            if not success:
                import shutil
                shutil.copy(tmp_path, output_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Edge TTS 직접 → 최종 파일
            print(f"  [Edge TTS] 음성 생성 중...")
            generate_base_audio(text, output_path)

        print(f"\n[완료] 음성 파일 저장: {output_path}")
        return output_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="emeth TTS 엔진")
    parser.add_argument("--text", "-t", type=str)
    parser.add_argument("--file", "-f", type=str)
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    engine = TTSEngine()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = input("변환할 텍스트: ").strip()

    engine.synthesize(text, args.output)


if __name__ == "__main__":
    main()