================================================================
  OBS WebSocket 설정 가이드
  VTuber 자동화 파이프라인 Step 4
================================================================

## 1. OBS WebSocket 서버 활성화

1. OBS Studio 실행
2. 상단 메뉴: [도구] → [WebSocket 서버 설정]
3. "WebSocket 서버 활성화" 체크박스 ON
4. 서버 포트: 4455 (기본값 유지 권장)
5. [적용] → [확인]

※ OBS 28.0 이상부터 WebSocket 서버가 내장되어 있습니다.


## 2. 비밀번호 설정 방법

[비밀번호 없이 사용할 경우 (로컬 전용 권장)]
- "인증 사용" 체크 해제
- config_obs.py: OBS_PASSWORD = ""

[비밀번호 사용할 경우]
1. [도구] → [WebSocket 서버 설정]
2. "인증 사용" 체크
3. 비밀번호 입력 및 저장
4. config_obs.py 수정:
   OBS_PASSWORD = "여기에_비밀번호_입력"


## 3. 씬 이름 설정 가이드

config_obs.py에서 아래 값을 실제 OBS 씬 이름과 일치하게 수정:

  SCENE_MAIN    = "emeth-main"   # 실제 방송용 씬 이름
  SCENE_WAITING = "emeth-wait"   # 실제 대기 화면 씬 이름

씬 이름 확인 방법:
- OBS Studio 우측 하단 [씬] 패널에서 이름 확인
- 대소문자, 공백 포함 정확히 일치해야 함
- test_obs.py 실행 시 현재 씬 목록이 출력됨


## 4. 녹화 경로 설정 (선택)

config_obs.py:
  RECORDING_PATH = ""             # 비어있으면 OBS 기본 경로 사용
  RECORDING_PATH = "D:\\vtuber"  # 특정 경로 지정 시


## 5. 연결 테스트

cd C:\Users\thtgg\mydream\vtuber-auto\step4_obs
python test_obs.py


## 6. API 서버 실행 (step4 포함)

cd C:\Users\thtgg\mydream\vtuber-auto
python -m uvicorn api.main:app --reload --port 8000

API 엔드포인트:
  GET  /obs/status            - OBS 현재 상태 조회
  POST /obs/start-recording   - 녹화 시작
  POST /obs/stop-recording    - 녹화 정지 (파일 경로 반환)
  POST /obs/start-stream      - 스트리밍 시작
  POST /obs/stop-stream       - 스트리밍 정지
  POST /obs/switch-scene      - 씬 전환 {"scene_name": "씬이름"}
  POST /run-pipeline          - 파이프라인 실행 {"use_obs": true}


## 7. 주의사항

- OBS Studio가 실행된 상태에서만 연결 가능
- 방화벽에서 4455 포트 차단 시 연결 불가 (로컬은 보통 OK)
- 하나의 클라이언트만 연결 권장 (동시 연결 시 충돌 가능)
- 녹화/스트리밍 중 OBS 강제 종료 시 파일이 손상될 수 있음


================================================================
  문의: VTuber 자동화 파이프라인 프로젝트
================================================================
