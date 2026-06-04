"""
test_rvc.py - RVC TTS 엔진 테스트

실행: python test_rvc.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("  emeth RVC TTS 엔진 테스트")
print("=" * 50)

# 1. 패키지 확인
print("\n[1] 패키지 설치 확인")
packages = {"edge-tts": "edge_tts", "rvc-python": "rvc_python", "pydub": "pydub"}
for pkg, mod in packages.items():
    try:
        __import__(mod)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} - 설치 필요: pip install {pkg}")

# 2. 모델 파일 확인
print("\n[2] RVC 모델 확인")
from config_tts import RVC_MODEL_DIR
model_files = [f for f in os.listdir(RVC_MODEL_DIR) if f.endswith(".pth")] if os.path.exists(RVC_MODEL_DIR) else []
if model_files:
    for m in model_files:
        print(f"  ✅ {m}")
else:
    print(f"  ❌ 모델 없음 - voice_models/ 폴더에 .pth 파일 추가 필요")
    print(f"     Edge TTS 모드로 테스트 진행합니다.")

# 3. TTS 테스트
print("\n[3] TTS 생성 테스트")
sample_text = "안녕하세요, 여러분. emeth입니다. [pause] 오늘도 함께해 주셔서 감사해요."
print(f"  텍스트: {sample_text[:40]}...")

try:
    from tts_engine import TTSEngine
    engine = TTSEngine()
    output = engine.synthesize(sample_text, "test_output.mp3")
    print(f"  ✅ 생성 완료: {output}")
except Exception as e:
    print(f"  ❌ 오류: {e}")

print("\n" + "=" * 50)
print("  테스트 완료!")
print("=" * 50)
