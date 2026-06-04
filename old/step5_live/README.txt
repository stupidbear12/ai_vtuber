============================================================
Step 5 (Live) — YouTube 라이브 방송 자동화
VTuber 자동화 파이프라인
============================================================

■ 개요
  YouTube Data API v3를 사용하여 라이브 방송 생성, RTMP 스트림 키 획득,
  OBS 자동 연결, 방송 상태 전환을 자동화하는 모듈입니다.

------------------------------------------------------------
■ 1. Google Cloud Console — OAuth 2.0 클라이언트 ID 생성
------------------------------------------------------------

  (1) https://console.cloud.google.com 접속 후 프로젝트 선택/생성

  (2) 좌측 메뉴 → "API 및 서비스" → "라이브러리"
      검색창에 "YouTube Data API v3" 입력 → 선택 → "사용 설정"

  (3) 좌측 메뉴 → "API 및 서비스" → "OAuth 동의 화면"
      - 사용자 유형: "외부" 선택 후 만들기
      - 앱 이름, 지원 이메일 입력 후 저장

  (4) 좌측 메뉴 → "API 및 서비스" → "사용자 인증 정보"
      → "사용자 인증 정보 만들기" → "OAuth 2.0 클라이언트 ID"
      - 애플리케이션 유형: "데스크톱 앱" 선택
      - 이름 입력 (예: emeth-vtuber)
      - "만들기" 클릭

  (5) 생성된 클라이언트 ID 우측의 ⬇ (다운로드) 버튼 클릭
      → JSON 파일 저장

------------------------------------------------------------
■ 2. client_secrets.json 배치
------------------------------------------------------------

  다운로드한 JSON 파일 이름을 반드시 아래와 같이 변경하세요:

    client_secrets.json

  그리고 다음 경로에 배치합니다:

    vtuber-auto/step5_live/client_secrets.json

  ※ config_live.py의 CLIENT_SECRETS_FILE 값을 변경하면
    다른 경로도 사용 가능합니다.

------------------------------------------------------------
■ 3. 필요 패키지 설치
------------------------------------------------------------

  pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client

------------------------------------------------------------
■ 4. 최초 인증 절차
------------------------------------------------------------

  처음 실행 시 브라우저가 자동으로 열리며 Google 계정 로그인을 요청합니다.

  (1) 브라우저에서 YouTube 채널 소유 계정으로 로그인
  (2) "계속" 클릭 (앱 권한 부여)
  (3) 인증 완료 후 브라우저 창이 닫힘

  인증 성공 시 youtube_token.json 파일이 자동 저장됩니다.
  이후 실행 시 브라우저 팝업 없이 자동으로 인증됩니다.

  ⚠ 주의: youtube_token.json에는 개인 인증 정보가 담겨 있습니다.
    Git 등 공개 저장소에 올리지 마세요.
    .gitignore에 추가를 권장합니다:
      step5_live/client_secrets.json
      step5_live/youtube_token.json

------------------------------------------------------------
■ 5. 실행 방법
------------------------------------------------------------

  [방법 A] 직접 실행 (커맨드라인)
  ---------------------------------
  # vtuber-auto/step5_live/ 디렉토리에서 실행

  # 방송 시작
  python live_pipeline.py start --title "내 방송 제목" --privacy public

  # 방송 종료 (start 출력의 broadcast_id 사용)
  python live_pipeline.py end --broadcast-id <broadcast_id>

  # 방송 상태 확인
  python live_pipeline.py status --broadcast-id <broadcast_id>


  [방법 B] API 서버를 통한 실행
  ---------------------------------
  # vtuber-auto/ 루트에서 서버 실행
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

  # 방송 시작
  POST http://localhost:8000/live/start
  Body: {"title": "emeth lofi 방송", "description": "설명", "privacy": "public"}

  # 방송 종료
  POST http://localhost:8000/live/end
  Body: {"broadcast_id": "<broadcast_id>"}

  # 방송 상태
  GET http://localhost:8000/live/status

  웹 UI: http://localhost:8000 접속 → 🔴 라이브 방송 섹션 사용

------------------------------------------------------------
■ 6. 파일 구성
------------------------------------------------------------

  step5_live/
  ├── config_live.py      — 방송 기본 설정 (제목, 해상도, 공개 설정 등)
  ├── youtube_live.py     — YouTubeLiveManager (YouTube API 래퍼)
  ├── live_pipeline.py    — LivePipeline 오케스트레이터 (OBS 연동)
  ├── client_secrets.json — Google OAuth 클라이언트 정보 (직접 배치 필요)
  ├── youtube_token.json  — 자동 저장되는 인증 토큰
  └── README.txt          — 이 파일

------------------------------------------------------------
■ 7. 자주 발생하는 오류
------------------------------------------------------------

  오류: "client_secrets.json 파일이 없습니다"
  → 2번 항목을 참고하여 파일을 배치하세요.

  오류: "OBS 연결 실패"
  → OBS가 실행 중인지, step4_obs/config_obs.py의 포트/비밀번호를 확인하세요.

  오류: "방송 상태 전환 실패 (testing)"
  → 스트림이 아직 OBS에서 전송되지 않았습니다.
    OBS 스트리밍이 정상적으로 시작된 후 YouTube Studio에서 스트림 상태를 확인하세요.

  오류: "HttpError 403"
  → YouTube 채널에 라이브 스트리밍 기능이 활성화되어 있는지 확인하세요.
    YouTube Studio → 라이브 스트리밍 탭에서 활성화할 수 있습니다.

============================================================
