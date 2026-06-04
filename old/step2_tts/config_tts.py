# config_tts.py - emeth RVC 기반 TTS 설정

import os

# ─── 기본 음성 (Edge TTS - RVC 변환 전 베이스) ────────────────
BASE_VOICE = "ko-KR-SunHiNeural"   # 중립 베이스 음성
BASE_VOICE_RATE = "-10%"           # 말하기 속도 (느리게)
BASE_VOICE_PITCH = "-5Hz"          # 기본 피치

# ─── RVC 설정 ────────────────────────────────────────────────
RVC_ENABLED = True                 # False면 Edge TTS 음성 그대로 사용
RVC_MODEL_DIR = os.path.join(os.path.dirname(__file__), "voice_models")
RVC_MODEL_NAME = "emeth"           # voice_models/ 안의 모델명 (확장자 제외)

# RVC 변환 파라미터
F0_UP_KEY = 0          # 피치 조정 (여성 목소리: +12, 남성→여성: +6~12)
F0_METHOD = "rmvpe"    # 피치 추출 방식: rmvpe (추천), harvest, crepe
INDEX_RATE = 0.75      # 음색 적용 강도 (0.0~1.0, 높을수록 모델 음색 강함)
FILTER_RADIUS = 3      # 피치 필터 반경 (홀수, 높을수록 부드러움)
RESAMPLE_SR = 0        # 출력 샘플레이트 (0=자동)
RMS_MIX_RATE = 0.25    # 원본/변환 볼륨 믹스 비율
PROTECT = 0.33         # 자음 보호 강도 (0.0~0.5)

# ─── 저장 설정 ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_audio")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ─── [pause] 태그 처리 ──────────────────────────────────────
PAUSE_DURATION_MS = 800   # [pause] 태그 → 무음 길이 (밀리초)
