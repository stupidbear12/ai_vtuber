# config_chat.py - emeth 유튜브 라이브 채팅 시스템 설정

# ─── 유튜브 API 설정 ─────────────────────────────────────────
YOUTUBE_API_KEY = ""          # Google API 키 (직접 입력)
LIVE_CHAT_POLL_INTERVAL = 5   # 채팅 폴링 간격 (초)
MAX_CHAT_QUEUE = 20           # 동시 처리 최대 채팅 수

# ─── emeth 채팅 응답 설정 ────────────────────────────────────
MAX_RESPONSE_LENGTH = 80      # 응답 최대 글자수 (짧고 자연스럽게)
RESPONSE_COOLDOWN = 10        # 같은 유저 재응답 대기 시간 (초)
SKIP_KEYWORDS = ["http", "www", "spam"]  # 무시할 채팅 키워드

# ─── TTS 연동 ────────────────────────────────────────────────
AUTO_TTS = True               # 응답을 자동으로 TTS 변환
AUTO_VTUBE = False            # VTube Studio 립싱크 연동 (VTS 실행 시)

# ─── Ollama 설정 ─────────────────────────────────────────────
OLLAMA_MODEL = "emeth"        # 사용할 Ollama 모델명
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama API 주소

# ─── 로그 설정 ───────────────────────────────────────────────
import os
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RESPONSE_LOG_PATH = os.path.join(LOG_DIR, "responses.jsonl")
