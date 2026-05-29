# config.py - emeth 버튜버 대본 생성 설정

# ─── 모델 설정 ───────────────────────────────────────────────
MODEL_NAME = "emeth"          # ollama create 후 사용할 커스텀 모델명
BASE_MODEL = "gemma3:4b"      # 기반 모델 (대안: llama3.2:3b)
MODELFILE_PATH = "./Modelfile"

# ─── 캐릭터 설정 ────────────────────────────────────────────
CHARACTER_NAME = "emeth"
CHARACTER_PERSONALITY = "차분하고 따뜻함"
CONTENT_TYPE = "lofi 음악 소개 및 일상 힐링"
VIEWER_PRONOUN = "여러분"

# ─── 대본 설정 ──────────────────────────────────────────────
SCRIPT_DURATION_MIN = 3       # 최소 분량 (분)
SCRIPT_DURATION_MAX = 5       # 최대 분량 (분)
WORDS_PER_MINUTE = 150        # 한국어 평균 말하기 속도 (분당 글자 수 기준)
TARGET_LENGTH = WORDS_PER_MINUTE * SCRIPT_DURATION_MIN  # 최소 글자 수

# ─── 저장 설정 ──────────────────────────────────────────────
import os
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "output_scripts")
os.makedirs(SCRIPT_DIR, exist_ok=True)

# ─── 프롬프트 템플릿 ────────────────────────────────────────
PROMPT_TEMPLATE = """다음 주제로 3~5분 분량의 버튜버 방송 대본을 작성해주세요.

주제: {topic}

대본 구성:
1. [오프닝] - 여러분을 따뜻하게 맞이하며 오늘의 주제 소개 (30초 분량)
2. [본론] - 음악의 분위기, 배경, 감성적인 이야기나 일상적인 단상 연결 (2~3분 분량)
3. [클로징] - 위로의 말과 감사 인사, 다음 방송 예고 (30초 분량)

주의사항:
- emeth의 말투(차분하고 따뜻한 존댓말)를 유지해주세요.
- 자연스러운 pause가 필요한 곳에 [pause] 표시를 넣어주세요.
- 잔잔하고 힐링되는 분위기를 유지해주세요.
- 전체 글자 수는 약 {target_length}자 이상으로 작성해주세요.
"""
