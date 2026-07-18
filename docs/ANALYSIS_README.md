# 코드 분석 읽기 가이드

`docs/CODEBASE_MAP.md`를 먼저 읽은 뒤, 아래 순서대로 파일을 열면 전체 흐름을 빠르게 파악할 수 있습니다.

---

## 추천 읽기 순서 (방송 기능 중심)

### 1단계 — 진입점·기동 (10분)

1. `start-all.ps1` — 어떤 모듈이 몇 번 포트로 뜨는지
2. `modules/broadcast/app/main.py` — 방송 API 표면
3. `modules/chat/app/main.py` — 채팅 API 표면
4. `modules/core/app/main.py` — 통합 API 표면

### 2단계 — 방송 핵심 파이프라인 (30분)

1. `modules/broadcast/app/chat_collector.py`
   - `ChatFilter.should_respond()` — 어떤 채팅에 반응하는지
   - `BroadcastChatManager._respond_to_chat()` — chat → voice → live2d
2. `modules/chat/app/chat_engine.py`
   - `generate_reply()` — 프롬프트·RAG·감정 파싱
3. `modules/chat/app/llm_provider.py` — Ollama 호출
4. `modules/live2d/app/router.py` — `POST /live2d/broadcast`
5. `modules/voice/app/voice_engine.py` — TTS

### 3단계 — 치지직 연동 (20분)

1. `modules/broadcast/app/chzzk_auth.py` — OAuth
2. `modules/broadcast/app/chzzk_session.py` — Session WebSocket
3. `modules/broadcast/app/chzzk_api.py` — Open API

### 4단계 — 프론트·표시 (15분)

1. `modules/live2d/static/js/app.js` — OBS 브라우저 소스
2. `modules/live2d/app/ws_manager.py` — WebSocket 명령 전달

### 5단계 — 선택

| 관심사 | 파일 |
|--------|------|
| RAG 메모리 | `modules/chat/app/memory.py`, `data/knowledge/*.md` |
| 통합 파이프라인 | `modules/core/app/orchestrator.py` |
| AI DJ (미완) | `modules/music/app/main.py`, `dj_controller.py` |

---

## 모듈별 “이 파일만 보면 됨”

| 모듈 | 최소 세트 |
|------|-----------|
| broadcast | `main.py` + `chat_collector.py` + `chzzk_*.py` |
| chat | `main.py` + `chat_engine.py` + `llm_provider.py` |
| live2d | `router.py` + `ws_manager.py` + `static/js/app.js` |
| voice | `main.py` + `voice_engine.py` |
| core | `main.py` + `orchestrator.py` |
| music | `main.py` + `music_engine.py` |

---

## Cursor / AI 분석 팁

1. **`modules/`만** `@` 멘션하거나 검색 범위로 제한
2. `old/`, `training/`은 레거시 — 현재 방송과 무관
3. Live2D `static/models/`는 JSON·바이너리만 있음 — `app/`과 `static/js/`만 보면 충분
4. 루트 `.cursorignore`가 있으면 인덱싱 노이즈가 줄어듦

---

## API 빠른 참조

```bash
# 방송 채팅 수집 시작
curl -X POST http://localhost:8003/broadcast/start \
  -H "Content-Type: application/json" \
  -d '{"platform":"chzzk","channel_id":"auto"}'

# 채팅 테스트
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"안녕 시온!","mode":"broadcast","viewer_name":"테스트"}'

# 통합 파이프라인
curl -X POST http://localhost:8000/pipeline/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"안녕","mode":"broadcast","with_voice":true}'
```

OBS URL: `http://localhost:8001/live2d/static/?transparent=1`
