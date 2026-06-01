# vtuber-auto — emeth 방송 자동화 파이프라인

**캐릭터:** emeth  
**콘텐츠:** 성경과 복음 기반 lofi 음악 소개  
**말투:** 차분하고 따뜻한 존댓말, 시청자를 '여러분'이라 부름  
**방송:** 24시간 자동 스트리밍

---

## 파이프라인 구조

```
[Ollama + Gemma3:4b]
        ↓
[step1_script]  대본 생성
        ↓  .txt
[step2_tts]     Edge TTS → .mp3  (→ step7_cover RVC 예정)
        ↓  .mp3
[step3_vtube]   VTube Studio 립싱크 + 리깅 애니메이션
        ↓  WebSocket
[step4_obs]     OBS 장면 전환 / 오디오 라우팅
        ↓  WebSocket
[step5_live]    YouTube 라이브 스트림 생성/관리
        ↓  Data API v3
[step6_chat]    채팅 감지 → emeth 즉시 응답

[api/]          FastAPI v0.7.0 — 전체 파이프라인 통합 백엔드
```

---

## 디렉토리 구조

```
vtuber-auto/
├── api/
│   ├── main.py              # FastAPI 앱 (v0.7.0)
│   ├── requirements.txt
│   └── static/index.html    # 대시보드 UI
├── step1_script/
│   ├── Modelfile            # emeth Ollama 캐릭터 정의
│   ├── config.py            # 모델/경로 설정
│   ├── generate_script.py   # 대본 자동 생성
│   ├── test_gen.py
│   └── output_scripts/      # 생성된 대본 저장
├── step2_tts/
│   ├── tts_engine.py        # Edge TTS 합성
│   ├── config_tts.py
│   ├── test_rvc.py          # RVC 테스트
│   ├── output_audio/        # 합성 음성 .mp3
│   ├── temp/                # 임시 청크
│   └── voice_models/        # RVC 모델 경로 (예정)
├── step3_vtube/
│   ├── vtube_controller.py  # VTS WebSocket 연결
│   ├── lipsync.py           # 립싱크 파라미터 제어
│   ├── rigging_animation.py # 리깅 애니메이션
│   ├── animation_controller.py
│   ├── animations.py
│   ├── config_vtube.py
│   └── test_vtube.py
├── step4_obs/
│   ├── obs_controller.py    # OBS WebSocket 제어
│   ├── config_obs.py
│   └── test_obs.py
├── step5_live/
│   ├── youtube_live.py      # YouTube 라이브 생성/관리
│   ├── live_pipeline.py     # 라이브 전체 흐름
│   └── config_live.py
├── step6_chat/
│   ├── youtube_chat.py      # 채팅 폴링
│   ├── chat_handler.py      # 응답 로직
│   ├── chat_pipeline.py
│   └── config_chat.py
└── step7_cover/             # RVC 커버 생성 (개발 중)
    ├── cover_pipeline.py
    └── config_cover.py
```

---

## 설치

```bash
# Python 패키지 설치
pip install ollama edge-tts websockets fastapi uvicorn obs-websocket-py google-api-python-client

# Ollama 설치 후 emeth 모델 등록
ollama create emeth -f step1_script/Modelfile
```

**사전 요구 사항:**
- [Ollama](https://ollama.com) 설치 및 실행 중
- [VTube Studio](https://denchisoft.com/) 실행 + WebSocket API 활성화
- [OBS Studio](https://obsproject.com/) + obs-websocket 플러그인
- YouTube Data API v3 인증 정보 (`client_secrets.json`)

---

## 실행 순서

```bash
# 1. FastAPI 백엔드 시작
cd api
uvicorn main:app --reload

# 2. 대본 생성
cd step1_script
python generate_script.py --topic "시편 23편과 함께하는 새벽 lofi"

# 3. TTS 음성 합성
cd step2_tts
python tts_engine.py

# 4. VTube Studio 립싱크 + 애니메이션
cd step3_vtube
python vtube_controller.py

# 5. OBS 방송 시작
cd step4_obs
python obs_controller.py

# 6. YouTube 라이브 스트림
cd step5_live
python live_pipeline.py

# 7. 채팅 응답 시작
cd step6_chat
python chat_pipeline.py
```

---

## 기술 스택

| 구성요소 | 기술 |
|---|---|
| LLM | Ollama + Gemma3:4b |
| TTS | Edge TTS (현재) → RVC (예정) |
| 캐릭터 | VTube Studio WebSocket API |
| 방송 | OBS WebSocket |
| 라이브 | YouTube Data API v3 |
| 백엔드 | FastAPI v0.7.0 |

---

## 개발 현황

| 단계 | 상태 |
|---|---|
| step1_script — 대본 생성 | ✅ 완성 |
| step2_tts — Edge TTS | ✅ 완성 |
| step3_vtube — 립싱크 + 애니메이션 | ✅ 완성 |
| step4_obs — OBS 제어 | ✅ 완성 |
| step5_live — YouTube 라이브 | ✅ 완성 |
| step6_chat — 채팅 응답 | ✅ 완성 |
| step7_cover — RVC 커버 | 🔧 개발 중 |
| api — FastAPI 통합 | ✅ v0.7.0 |
| TTS → RVC 교체 | ⏳ 예정 |
