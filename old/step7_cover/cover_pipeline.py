"""
cover_pipeline.py - emeth AI 커버곡 생성 파이프라인

흐름:
  원곡 (풀 트랙 또는 보컬/MR 분리 파일)
      ↓ [선택] Demucs 보컬 분리
      ↓ RVC — 보컬을 emeth 목소리로 변환
      ↓ 믹싱 — 변환된 보컬 + MR 합치기
      ↓ 완성된 emeth 커버곡 저장

사용법:
  # 풀 트랙에서 자동으로 보컬 분리 후 커버
  python cover_pipeline.py --input song.mp3 --separate

  # 이미 분리된 보컬 + MR로 커버
  python cover_pipeline.py --vocal vocals.wav --instrumental mr.wav

  # 보컬만 변환 (믹싱 없이)
  python cover_pipeline.py --vocal vocals.wav --no-mix

의존성:
  pip install rvc-python soundfile numpy
  pip install demucs   (보컬 분리 시 필요)
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from config_cover import (
    INPUT_DIR, OUTPUT_DIR, SEPARATED_DIR, TEMP_DIR,
    RVC_MODEL_DIR, RVC_MODEL_NAME,
    F0_UP_KEY, F0_METHOD, INDEX_RATE, FILTER_RADIUS,
    RMS_MIX_RATE, PROTECT,
    VOCAL_VOLUME, INSTRUMENTAL_VOLUME,
    DEMUCS_MODEL,
)


# ─────────────────────────────────────────────────────────────
# STEP 1 : 보컬 분리 (Demucs)
# ─────────────────────────────────────────────────────────────

def separate_vocals(input_path: str) -> tuple[str, str]:
    """
    Demucs로 원곡에서 보컬(vocals)과 MR(no_vocals)을 분리합니다.
    반환: (vocal_path, instrumental_path)
    """
    try:
        import demucs.separate
    except ImportError:
        raise RuntimeError(
            "Demucs가 설치되지 않았습니다.\n"
            "설치: pip install demucs\n"
            "또는 --vocal / --instrumental 옵션으로 분리된 파일을 직접 입력하세요."
        )

    input_path = os.path.abspath(input_path)
    song_name = Path(input_path).stem

    print(f"[Step 1] 보컬 분리 중 — {Path(input_path).name}")
    print(f"  모델: {DEMUCS_MODEL} (처음 실행 시 모델 다운로드, 약 1~2분 소요)")

    result = subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", DEMUCS_MODEL,
            "-o", SEPARATED_DIR,
            input_path,
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Demucs 실패:\n{result.stderr}")

    # 출력 경로: separated/htdemucs/<song_name>/
    out_dir = os.path.join(SEPARATED_DIR, DEMUCS_MODEL, song_name)
    vocal_path        = os.path.join(out_dir, "vocals.wav")
    instrumental_path = os.path.join(out_dir, "no_vocals.wav")

    if not os.path.exists(vocal_path):
        raise FileNotFoundError(f"분리된 보컬 파일을 찾을 수 없습니다: {vocal_path}")

    print(f"  보컬   : {vocal_path}")
    print(f"  MR     : {instrumental_path}")
    return vocal_path, instrumental_path


# ─────────────────────────────────────────────────────────────
# STEP 2 : RVC 보컬 변환
# ─────────────────────────────────────────────────────────────

def find_rvc_model() -> tuple:
    """voice_models/ 에서 emeth .pth 모델 탐색"""
    model_dir = Path(RVC_MODEL_DIR)
    if not model_dir.exists():
        return None, None
    pth_files = list(model_dir.glob(f"{RVC_MODEL_NAME}*.pth")) or list(model_dir.glob("*.pth"))
    if not pth_files:
        return None, None
    index_files = list(model_dir.glob("*.index"))
    return str(pth_files[0]), str(index_files[0]) if index_files else ""


def convert_vocal_rvc(vocal_path: str, output_path: str) -> str:
    """
    RVC로 보컬을 emeth 목소리로 변환합니다.
    반환: 변환된 보컬 파일 경로
    """
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        raise RuntimeError("rvc-python 미설치: pip install rvc-python")

    model_path, index_path = find_rvc_model()
    if not model_path:
        raise FileNotFoundError(
            f"RVC 모델(.pth)이 없습니다.\n"
            f"경로: {RVC_MODEL_DIR}\n"
            f"weights.gg 또는 Hugging Face에서 모델을 다운로드하거나 직접 학습해주세요.\n"
            f"자세한 방법: step2_tts/voice_models/README.txt"
        )

    print(f"\n[Step 2] RVC 보컬 변환 중")
    print(f"  모델    : {Path(model_path).name}")
    print(f"  피치    : {F0_UP_KEY:+d} 반음 / 방식: {F0_METHOD}")
    print(f"  음색강도: {INDEX_RATE}")

    rvc = RVCInference(models_path=RVC_MODEL_DIR)
    rvc.load_model(Path(model_path).stem)
    rvc.infer(
        input_path=vocal_path,
        output_path=output_path,
        f0_up_key=F0_UP_KEY,
        f0_method=F0_METHOD,
        index_rate=INDEX_RATE,
        filter_radius=FILTER_RADIUS,
        rms_mix_rate=RMS_MIX_RATE,
        protect=PROTECT,
        index_path=index_path or "",
    )

    print(f"  완료    : {Path(output_path).name}")
    return output_path


# ─────────────────────────────────────────────────────────────
# STEP 3 : 믹싱 (변환 보컬 + MR)
# ─────────────────────────────────────────────────────────────

def mix_audio(vocal_path: str, instrumental_path: str, output_path: str) -> str:
    """
    변환된 보컬과 MR을 합쳐 최종 커버곡을 만듭니다.
    soundfile + numpy 사용 (ffmpeg 불필요)
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise RuntimeError("soundfile / numpy 미설치: pip install soundfile numpy")

    print(f"\n[Step 3] 믹싱 중")

    vocal_data,  vocal_sr  = sf.read(vocal_path)
    inst_data,   inst_sr   = sf.read(instrumental_path)

    # 샘플레이트 일치 확인
    if vocal_sr != inst_sr:
        print(f"  [경고] 샘플레이트 불일치 (보컬:{vocal_sr} / MR:{inst_sr}). 보컬 기준으로 진행.")

    # 모노 → 스테레오 변환
    if vocal_data.ndim == 1:
        vocal_data = np.column_stack([vocal_data, vocal_data])
    if inst_data.ndim == 1:
        inst_data = np.column_stack([inst_data, inst_data])

    # 길이 맞추기 (짧은 쪽을 긴 쪽에 맞게 zero-padding)
    max_len = max(len(vocal_data), len(inst_data))
    if len(vocal_data) < max_len:
        pad = max_len - len(vocal_data)
        vocal_data = np.vstack([vocal_data, np.zeros((pad, vocal_data.shape[1]))])
    if len(inst_data) < max_len:
        pad = max_len - len(inst_data)
        inst_data = np.vstack([inst_data, np.zeros((pad, inst_data.shape[1]))])

    # 볼륨 조정 후 믹싱
    mixed = (vocal_data * VOCAL_VOLUME) + (inst_data * INSTRUMENTAL_VOLUME)

    # 클리핑 방지 (노멀라이즈)
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    sf.write(output_path, mixed, vocal_sr, subtype="PCM_16")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  저장    : {Path(output_path).name} ({size_mb:.1f} MB)")
    return output_path


# ─────────────────────────────────────────────────────────────
# 전체 파이프라인
# ─────────────────────────────────────────────────────────────

class CoverPipeline:
    def __init__(self):
        pass

    def run(
        self,
        input_path: str = None,
        vocal_path: str = None,
        instrumental_path: str = None,
        separate: bool = False,
        no_mix: bool = False,
        output_name: str = None,
        f0_key: int = None,
    ) -> dict:
        """
        커버곡 생성 파이프라인 실행
        반환: {"vocal": 변환된_보컬_경로, "cover": 최종_커버곡_경로}
        """
        global F0_UP_KEY
        if f0_key is not None:
            F0_UP_KEY = f0_key

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        song_name = output_name or (Path(input_path or vocal_path).stem if (input_path or vocal_path) else "cover")

        print("=" * 55)
        print(f"  emeth AI 커버곡 생성 파이프라인")
        print(f"  곡명: {song_name}")
        print("=" * 55)

        # Step 1: 보컬 분리 (필요한 경우)
        if separate and input_path:
            vocal_path, instrumental_path = separate_vocals(input_path)
        elif input_path and not vocal_path:
            raise ValueError("--separate 옵션 없이 --input 사용 시 --vocal 과 --instrumental 도 필요합니다.")

        if not vocal_path:
            raise ValueError("보컬 파일이 지정되지 않았습니다.")

        # Step 2: RVC 변환
        converted_vocal = os.path.join(TEMP_DIR, f"converted_{ts}.wav")
        convert_vocal_rvc(vocal_path, converted_vocal)

        result = {"vocal": converted_vocal, "cover": None}

        # Step 3: 믹싱
        if not no_mix and instrumental_path:
            cover_path = os.path.join(OUTPUT_DIR, f"emeth_{song_name}_{ts}.wav")
            mix_audio(converted_vocal, instrumental_path, cover_path)
            result["cover"] = cover_path
            print(f"\n[완료] 커버곡 저장: {cover_path}")
        elif no_mix:
            final_vocal = os.path.join(OUTPUT_DIR, f"emeth_vocal_{song_name}_{ts}.wav")
            shutil.copy(converted_vocal, final_vocal)
            result["vocal"] = final_vocal
            print(f"\n[완료] 변환 보컬 저장: {final_vocal}")
        else:
            print("\n[안내] MR 파일이 없어 보컬만 변환됐습니다.")
            final_vocal = os.path.join(OUTPUT_DIR, f"emeth_vocal_{song_name}_{ts}.wav")
            shutil.copy(converted_vocal, final_vocal)
            result["vocal"] = final_vocal

        print("=" * 55)
        return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="emeth AI 커버곡 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 풀 트랙 자동 분리 후 커버
  python cover_pipeline.py --input "노래.mp3" --separate

  # 미리 분리된 보컬 + MR로 커버
  python cover_pipeline.py --vocal vocals.wav --instrumental mr.wav

  # 피치 조정 (여성 목소리: +12)
  python cover_pipeline.py --vocal vocals.wav --instrumental mr.wav --pitch 12

  # 보컬 변환만 (믹싱 없이)
  python cover_pipeline.py --vocal vocals.wav --no-mix
        """
    )
    parser.add_argument("--input",          "-i", help="원곡 파일 (--separate 와 함께 사용)")
    parser.add_argument("--vocal",          "-v", help="분리된 보컬 파일")
    parser.add_argument("--instrumental",   "-m", help="MR(반주) 파일")
    parser.add_argument("--separate",       "-s", action="store_true", help="Demucs로 보컬 자동 분리")
    parser.add_argument("--no-mix",               action="store_true", help="믹싱 없이 보컬만 변환")
    parser.add_argument("--output",         "-o", help="출력 파일명 (확장자 제외)")
    parser.add_argument("--pitch",          "-p", type=int, default=None, help="피치 조정 반음 (기본: 0)")
    args = parser.parse_args()

    if not args.input and not args.vocal:
        parser.print_help()
        print("\n[오류] --input 또는 --vocal 중 하나는 필요합니다.")
        sys.exit(1)

    pipeline = CoverPipeline()
    pipeline.run(
        input_path=args.input,
        vocal_path=args.vocal,
        instrumental_path=args.instrumental,
        separate=args.separate,
        no_mix=args.no_mix,
        output_name=args.output,
        f0_key=args.pitch,
    )


if __name__ == "__main__":
    main()