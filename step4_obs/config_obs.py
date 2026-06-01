# ============================================================
# config_obs.py - OBS WebSocket 연결 설정
# VTuber 자동화 파이프라인 Step 4
# ============================================================

# OBS WebSocket 서버 주소
OBS_HOST = "localhost"

# OBS WebSocket 포트 (기본값: 4455)
OBS_PORT = 4455

# OBS WebSocket 비밀번호 (OBS 설정에서 확인)
# 비어있으면 인증 없이 연결 시도
OBS_PASSWORD = ""

# 녹화 저장 경로 (비어있으면 OBS 기본 설정 경로 사용)
RECORDING_PATH = ""

# 씬 이름 설정
SCENE_MAIN = "emeth-main"    # 방송용 메인 씬
SCENE_WAITING = "emeth-wait" # 대기 화면 씬

# 파이프라인 시작 시 씬 자동 전환 여부
AUTO_SWITCH_SCENE = True

# 연결 타임아웃 (초)
CONNECTION_TIMEOUT = 10

# 재연결 시도 횟수
MAX_RETRY = 3
