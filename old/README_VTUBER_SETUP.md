# AI Vtuber 빠른 세팅 가이드 (Windows)

이 문서는 현재 저장소(`ai_vtuber`)를 기준으로, 설치/실행/핵심 설정을 빠르게 맞추는 용도입니다.

## 1) 사전 준비

- Python 3.10+ 설치 (설치 시 PATH 추가 체크)
- ffmpeg 설치 및 PATH 등록
- VTube Studio 실행 (WebSocket API 포트 기본 8001)
- (선택) Ollama 설치/실행: 로컬 LLM을 쓸 경우

## 2) 의존성 설치

프로젝트 루트에서:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r api/requirements.txt
python -m pip install google-api-python-client requests
```

## 3) 핵심 설정 파일

### LLM/채팅
- 파일: `step6_chat/config_chat.py`
- 필수:
  - `YOUTUBE_API_KEY` 입력 (유튜브 라이브 채팅 연동 시)
  - `OLLAMA_MODEL`, `OLLAMA_BASE_URL` 확인

### TTS
- 파일: `step2_tts/config_tts.py`
- 기본 Edge TTS 보이스:
  - `BASE_VOICE = "ko-KR-SunHiNeural"`
- 필요하면 RVC 관련 값(`RVC_ENABLED`, `RVC_MODEL_NAME`) 조정

### VTube Studio
- 파일: `step3_vtube/config_vtube.py`
- 기본값:
  - `VTS_HOST = "localhost"`
  - `VTS_PORT = 8001`
  - `TOKEN_FILE = "./vts_token.json"`

## 4) 실행

### API 서버 실행

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- 브라우저: `http://localhost:8000`
- API 문서: `http://localhost:8000/docs`

### VTube 단독 아이들 테스트

```powershell
python run_idle.py
```

## 5) 자주 생기는 문제

- `pyvts` 관련 오류:
  - `python -m pip install pyvts websockets`
- ffmpeg 인식 안 됨:
  - 설치 후 새 터미널에서 `ffmpeg -version` 확인
- VTube 인증 실패:
  - VTube Studio 승인 팝업 허용 후 재실행
  - `step3_vtube/vts_token.json` 삭제 후 다시 인증
- Ollama 연결 실패:
  - `ollama serve` 상태 확인
  - `config_chat.py`의 주소/모델명 일치 확인

## 6) 권장 다음 단계

- `AUTO_VTUBE=True`로 두고 `step6_chat/chat_pipeline.py` 연동 테스트
- `/vtube/animation/start`, `/vtube/speak` 엔드포인트 조합으로 감정/립싱크 확인
- 안정화 후 실행용 PowerShell 스크립트 추가(원하면 바로 만들어드림)
