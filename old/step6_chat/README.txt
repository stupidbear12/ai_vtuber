========================================
 emeth 유튜브 라이브 채팅 시스템
 step6_chat / README
========================================

【 개요 】
유튜브 라이브 방송 중 시청자 채팅을 실시간으로 수신하고,
emeth (Ollama 모델)가 따뜻하게 응답을 생성하는 시스템입니다.


========================================
 1. 사전 준비 — Google API 키 발급
========================================

1. https://console.cloud.google.com 접속
2. 새 프로젝트 생성 (또는 기존 프로젝트 선택)
3. 상단 메뉴 → [API 및 서비스] → [라이브러리]
4. 검색창에 "YouTube Data API v3" 입력 후 선택
5. [사용 설정] 클릭

6. [API 및 서비스] → [사용자 인증 정보]
7. [+ 사용자 인증 정보 만들기] → [API 키] 클릭
8. 생성된 API 키를 복사

9. config_chat.py 파일을 열어 아래 줄에 붙여넣기:
   YOUTUBE_API_KEY = "여기에_API_키_입력"


========================================
 2. YouTube Data API v3 활성화 확인
========================================

Google Cloud Console에서:
- [API 및 서비스] → [사용 설정된 API 및 서비스]
- 목록에 "YouTube Data API v3" 가 있으면 활성화 완료


========================================
 3. video_id 찾는 방법
========================================

유튜브 라이브 방송 URL 예시:
  https://www.youtube.com/watch?v=dQw4w9WgXcQ
                                  ^^^^^^^^^^^^
  → video_id = dQw4w9WgXcQ  (watch?v= 뒤의 값)

※ 반드시 현재 진행 중인 라이브 방송의 video_id여야 합니다.
   다시보기(VOD)는 사용 불가.


========================================
 4. 의존성 설치
========================================

프로젝트 루트에서 실행:

  pip install google-api-python-client

(나머지 의존성 — requests, fastapi 등은 이미 설치되어 있다고 가정)


========================================
 5. 실행 방법
========================================

[ 방법 A ] 직접 실행 (단독)
  cd vtuber-auto/step6_chat
  python chat_pipeline.py <video_id>

  예시:
  python chat_pipeline.py dQw4w9WgXcQ

[ 방법 B ] FastAPI 서버를 통한 실행
  1. 서버 시작:
     cd vtuber-auto
     uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

  2. 브라우저에서 http://localhost:8000 접속
     → 웹 UI에서 [채팅 모니터링] 섹션 사용

  3. 또는 API 직접 호출:
     POST http://localhost:8000/chat/start
     Body: {"video_id": "dQw4w9WgXcQ"}

     GET  http://localhost:8000/chat/status
     POST http://localhost:8000/chat/stop


========================================
 6. 설정 조정 (config_chat.py)
========================================

LIVE_CHAT_POLL_INTERVAL = 5   # 채팅 폴링 간격 (초) — 너무 낮으면 API 할당량 소모
MAX_CHAT_QUEUE = 20           # 큐 최대 크기
MAX_RESPONSE_LENGTH = 80      # emeth 응답 최대 글자수
RESPONSE_COOLDOWN = 10        # 같은 유저 재응답 대기 시간 (초)
AUTO_TTS = True               # 응답 자동 TTS 변환 여부
AUTO_VTUBE = False            # VTube Studio 립싱크 연동 여부

SKIP_KEYWORDS = ["http", "www", "spam"]  # 이 키워드가 포함된 채팅은 무시


========================================
 7. 로그 확인
========================================

응답 이력: step6_chat/logs/responses.jsonl
  - 각 줄이 JSON 형식으로 저장됨
  - {"timestamp": ..., "username": ..., "message": ..., "response": ...}


========================================
 8. 주의사항
========================================

- YouTube Data API v3 무료 할당량: 하루 10,000 유닛
  채팅 조회 1회 = 약 5 유닛 소모
  → LIVE_CHAT_POLL_INTERVAL을 5초 이상으로 유지 권장

- Ollama(emeth 모델)가 실행 중이어야 응답이 생성됩니다.
  ollama serve 명령으로 확인하세요.

========================================
