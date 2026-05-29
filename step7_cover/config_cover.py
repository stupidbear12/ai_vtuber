# config_cover.py - emeth 커버곡 파이프라인 설정

import os

# ─── 경로 설정 ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR      = os.path.join(BASE_DIR, "input")        # 원본 파일 넣는 곳
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")       # 완성 커버곡 저장
SEPARATED_DIR  = os.path.join(BASE_DIR, "separated")    # 분리된 보컬/MR 저장
TEMP_DIR       = os.path.join(BASE_DIR, "temp")

for d in [INPUT_DIR, OUTPUT_DIR, SEPARATED_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── RVC 모델 경로 ───────────────────────────────────────────
RVC_MODEL_DIR  = os.path.join(os.path.dirname(BASE_DIR), "step2_tts", "voice_models")
RVC_MODEL_NAME = "emeth"

# ─── RVC 변환 파라미터 ──────────────────────────────────────
F0_UP_KEY      = 0       # 피치 조정 (여성 목소리 맞추기: +12 시도)
F0_METHOD      = "rmvpe" # rmvpe 추천 (노래에 가장 적합)
INDEX_RATE     = 0.75    # 음색 강도 (0.0~1.0)
FILTER_RADIUS  = 3       # 피치 필터 (홀수, 높을수록 부드러움)
RMS_MIX_RATE   = 0.25    # 볼륨 믹스
PROTECT        = 0.33    # 자음 보호

# ─── 믹싱 설정 ──────────────────────────────────────────────
VOCAL_VOLUME   = 1.0     # 변환된 보컬 볼륨 (1.0 = 원본)
INSTRUMENTAL_VOLUME = 0.85  # MR 볼륨

# ─── 보컬 분리 설정 (Demucs) ────────────────────────────────
DEMUCS_MODEL   = "htdemucs"   # 최고 품질 모델
DEMUCS_STEMS   = ["vocals", "no_vocals"]  # vocals + no_vocals(MR)