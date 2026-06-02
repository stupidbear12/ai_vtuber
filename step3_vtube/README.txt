=======================================================
step3_vtube - VTube Studio API 연동 모듈
emeth VTuber 방송 자동화 파이프라인 3주차
=======================================================

-------------------------------------------------------
1. VTube Studio 설치 방법
-------------------------------------------------------
1) Steam 실행
2) 검색창에 "VTube Studio" 입력
3) 무료 앱이므로 바로 설치 가능
   - Steam: https://store.steampowered.com/app/1325860/VTube_Studio/
4) 설치 후 VTube Studio 실행

-------------------------------------------------------
2. WebSocket API 활성화 방법
-------------------------------------------------------
1) VTube Studio 실행
2) 우측 하단의 [설정(기어 아이콘)] 클릭
3) 상단 탭에서 [연결] 또는 [Connections] 탭 클릭
4) "Start API (WebSocket server)" 를 ON으로 설정
5) 포트 번호 확인: 기본값 8001 (config_vtube.py의 VTS_PORT와 일치해야 함)
6) "Allow plugin token requests" 도 ON으로 설정

-------------------------------------------------------
3. 무료 샘플 모델 로드 방법 (Akari 기본 모델)
-------------------------------------------------------
1) VTube Studio 실행 후 메인 화면에서
   [모델 선택] 버튼 클릭 (화면 좌측 하단 사람 아이콘)
2) "Sample Models" 폴더에서 기본 제공 모델 선택
   - Akari (권장) 또는 다른 내장 샘플 모델
3) 모델 로드 완료 후 화면에 캐릭터가 표시되면 준비 완료

별도 다운로드 불필요 - VTube Studio 설치 시 포함되어 있습니다.

-------------------------------------------------------
4. 표정(Expression) 등록 방법
-------------------------------------------------------
표정을 사용하려면 VTube Studio에서 미리 등록해야 합니다.

1) 모델 로드 후 [설정] > [표정(Expressions)] 탭
2) [+] 버튼으로 새 표정 추가
3) config_vtube.py의 EXPRESSIONS 딕셔너리에 정의된 이름과 동일하게 등록:
   - default  (기본 표정)
   - happy    (밝은 표정)
   - calm     (차분한 표정)
   - speaking (말하는 표정)

표정이 등록되지 않아도 연결/립싱크 기능은 정상 작동합니다.

-------------------------------------------------------
5. 플러그인 인증 방법
-------------------------------------------------------
1) Python 스크립트 실행 (처음 실행 시)
2) VTube Studio 화면에 인증 팝업이 나타남:
   "emeth-controller 플러그인 연결 허용?"
3) [허용] 버튼 클릭
4) 인증 토큰이 vts_token.json에 자동 저장됨
5) 이후 실행 시에는 팝업 없이 자동 인증

-------------------------------------------------------
6. 필요 패키지 설치
-------------------------------------------------------
# 필수
pip install pyvts websockets

# 립싱크 분석 (둘 중 하나 설치)
pip install librosa          # 고품질 분석 (권장)
pip install pydub            # 가벼운 대안

# pydub 사용 시 ffmpeg 추가 필요
# https://ffmpeg.org/download.html 에서 다운로드 후 PATH 등록

-------------------------------------------------------
7. 테스트 실행 방법
-------------------------------------------------------
VTube Studio가 실행 중이고 API가 활성화된 상태에서:

  cd C:\Users\thtgg\mydream\vtuber-auto
  python step3_vtube/test_vtube.py

  ※ -m 방식(python -m step3_vtube.test_vtube)은 내부 임포트 경로 문제로
     직접 스크립트 실행 방식을 권장합니다.

-------------------------------------------------------
8. API 서버 실행 방법 (step1~3 통합)
-------------------------------------------------------
  cd C:\Users\thtgg\mydream\vtuber-auto
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

브라우저에서 http://localhost:8000 접속
VTube Studio 제어: POST /control-vtube
통합 파이프라인:   POST /run-pipeline (use_vtube: true 설정 시 VTS 연동)

v0.8.0 VTube Studio 제어 API 전체 목록:
  GET  /vtube/status           - 연결 상태 및 애니메이션 활성 여부 조회
  POST /vtube/connect          - VTube Studio 연결 및 플러그인 인증
  POST /vtube/disconnect       - 연결 해제 (애니메이션 자동 정지)
  POST /vtube/animation/start  - 전체 리깅 애니메이션 시작
  POST /vtube/animation/stop   - 애니메이션 정지
  POST /vtube/emotion          - 감정 변경 (calm/happy/sad/surprised/thinking)
  POST /vtube/reaction         - 즉각 반응 (chat_superchat/surprised/nod/shake)
  POST /vtube/speak            - 립싱크 + 감정 동시 처리

-------------------------------------------------------
9. 파일 구조
-------------------------------------------------------
step3_vtube/
  config_vtube.py          - API 설정값 (포트, 파라미터 이름 등)
  vtube_controller.py      - VTubeController 클래스 (메인 모듈)
  animations.py            - AnimationEngine (레거시 호환용)
  animation_controller.py  - AnimationController (전체 리깅 오케스트레이터)
  rigging_animation.py     - 개별 애니메이션 레이어 클래스들
  lipsync.py               - 오디오 분석 립싱크 값 생성
  test_vtube.py            - 연결/기능 테스트 스크립트
  README.txt               - 이 파일

-------------------------------------------------------
10. Windows 한글 인코딩 문제 해결 (중요)
-------------------------------------------------------
Windows 환경에서 cp949(EUC-KR) 인코딩으로 인해 한글이 깨질 수 있습니다.
반드시 아래 환경변수를 설정한 후 실행하세요:

  방법 A - 실행 시 직접 지정 (권장):
    set PYTHONUTF8=1 && python step3_vtube/test_vtube.py
    set PYTHONUTF8=1 && uvicorn api.main:app --reload

  방법 B - 시스템 환경변수 영구 등록:
    1) Windows 키 → "시스템 환경 변수 편집" 검색
    2) "환경 변수" 클릭 → 시스템 변수에서 "새로 만들기"
    3) 변수 이름: PYTHONUTF8   값: 1

  방법 C - PowerShell 세션 내 임시 적용:
    $env:PYTHONUTF8 = "1"
    python step3_vtube/test_vtube.py

모든 Python 파일 상단에 # -*- coding: utf-8 -*- 헤더가 포함되어 있으나,
Windows 터미널 출력(stdout)은 PYTHONUTF8=1 없이는 여전히 깨질 수 있습니다.

-------------------------------------------------------
주의사항
-------------------------------------------------------
- VTube Studio는 PC(Windows/Mac)에서만 실행 가능합니다
- WebSocket API는 로컬 전용입니다 (localhost:8001)
- 립싱크는 실제 오디오 재생과 별도로 파라미터만 제어합니다
  (오디오 재생은 별도 처리 필요)
- pyvts API 호환성: 이 코드는 현행 pyvts API 기준으로 작성되었습니다
    표정 전환: requestTriggerHotKey(hotkeyID)   ← VTS 핫키 이름과 일치해야 함
    파라미터:  requestSetMultiParameterValue()  ← InjectParameterDataRequest 대체
    모델 정보: BaseRequest("CurrentModelRequest")
- 표정 핫키(EXPRESSIONS)는 VTube Studio에서 사전 등록 필요
  (설정 > 표정 탭 > 핫키 이름을 config_vtube.py EXPRESSIONS 값과 동일하게)
