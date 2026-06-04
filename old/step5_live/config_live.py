# ============================================================
# config_live.py - YouTube 라이브 방송 설정
# VTuber 자동화 파이프라인 Step 5 (Live)
# ============================================================

# OAuth 인증 파일 경로
CLIENT_SECRETS_FILE = "./client_secrets.json"   # Google Cloud에서 다운로드
TOKEN_FILE = "./youtube_token.json"             # 자동 저장 (최초 인증 후 재사용)

# 방송 기본 설정
DEFAULT_TITLE = "emeth lofi 방송"
DEFAULT_DESCRIPTION = "emeth와 함께하는 잔잔한 lofi 음악"
DEFAULT_PRIVACY = "public"        # public / unlisted / private
DEFAULT_CATEGORY_ID = "10"        # 10 = 음악 카테고리
LATENCY_PREFERENCE = "ultraLow"   # ultraLow / low / normal

# 스트림 인코더 설정
STREAM_RESOLUTION = "1080p"
STREAM_FRAME_RATE = "30fps"

# YouTube API 권한 범위
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube"]

# 방송 상태 전환 대기 (초)
TRANSITION_WAIT_SEC = 5
