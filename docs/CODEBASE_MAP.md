# AI VTuber — 코드베이스 맵 (분석용)

> 최종 갱신: 2026-06-20  
> **분석 범위: `modules/` 활성 서비스만.** `old/`, `training/`, `chatbot/`은 레거시·별도 프로젝트.

---

## 1. 한눈에 보는 구조

```
ai_vtuber/
├── modules/          ← ★ 분석·실행의 전부
│   ├── core/         8000  오케스트레이터
│   ├── live2d/       8001  Live2D 웹뷰어 + WebSocket
│   ├── chat/         8002  Ollama 채팅 + RAG
│   ├── broadcast/    8003  치지직/유튜브 채팅 수집
│   ├── voice/        8004  GPT-SoVITS TTS
│   └── music/        8005  AI DJ (개발중, start-all 미포함)
├── start-all.ps1     로컬 5모듈 기동
├── .env              런타임 설정
├── docs/             분석 문서 (여기)
│
├── old/              ✗ 분석 제외 — 구 파이프라인
├── training/         ✗ 분석 제외 — 파인튜닝·체크포인트
├── chatbot/          ✗ 분석 제외 — Vercel Next.js 앱
└── scripts/          ✗ 분석 제외 — 데이터 전처리
```

---

## 2. 모듈별 핵심 파일

### core (8000) — `modules/core/app/`

| 파일 | 역할 |
|------|------|
| `main.py` | FastAPI, `/health`, `/status`, `/pipeline/chat`, broadcast 프록시 |
| `orchestrator.py` | chat → live2d → voice 파이프라인, 헬스 집계 |

### chat (8002) — `modules/chat/app/`

| 파일 | 역할 |
|------|------|
| `main.py` | `POST /chat`, RAG lifespan |
| `chat_engine.py` | 시스템 프롬프트, RAG 컨텍스트, 감정 태그 파싱 |
| `llm_provider.py` | Ollama HTTP 호출 |
| `memory.py` | ChromaDB RAG (대화 기억 + 지식) |
| `data/knowledge/*.md` | 캐릭터 지식 베이스 |

### live2d (8001) — `modules/live2d/app/` + `static/js/`

| 파일 | 역할 |
|------|------|
| `main.py` | FastAPI, 정적 파일, WebSocket |
| `router.py` | `/live2d/emotion`, `/live2d/broadcast`, 모션 API |
| `ws_manager.py` | 브라우저 WebSocket 브로드캐스트 |
| `static/js/app.js` | PixiJS Live2D 뷰어 (OBS용) |

### broadcast (8003) — `modules/broadcast/app/`

| 파일 | 역할 |
|------|------|
| `main.py` | `/broadcast/start`, `/broadcast/stop`, Chzzk OAuth |
| `chat_collector.py` | 채팅 수집·선별·응답 파이프라인 (핵심) |
| `chzzk_auth.py` | OAuth 토큰 관리 |
| `chzzk_api.py` | 치지직 Open API |
| `chzzk_session.py` | Session API WebSocket |

### voice (8004) — `modules/voice/app/`

| 파일 | 역할 |
|------|------|
| `main.py` | `POST /voice/tts` |
| `voice_engine.py` | GPT-SoVITS API 서버(api_v2.py) HTTP 클라이언트 |

### music (8005) — `modules/music/app/` (개발중)

| 파일 | 역할 |
|------|------|
| `main.py` | DJ API, WebSocket 스트림 |
| `music_engine.py` | ACE-Step 연동 (스텁 모드 지원) |
| `dj_controller.py` | 자동 선곡·큐 |

---

## 3. 런타임 데이터 흐름 (방송)

```
치지직 채팅
    ↓ WebSocket (chzzk_session / 비공식 WS)
broadcast/chat_collector.py
    ↓ ChatFilter (키워드·후원·랜덤·간격)
    ↓ POST :8002/chat  {mode:"broadcast", viewer_name, is_donation}
chat/chat_engine.py → Ollama (:11434)
    ↓ {reply, emotion}
    ├→ POST :8004/voice/tts
    ├→ POST :8001/live2d/emotion + /live2d/broadcast
    └→ Chzzk send_chat (공식 API, 선택)
OBS ← live2d/static/?transparent=1
```

**수동 기동:** `POST :8003/broadcast/start` `{"platform":"chzzk","channel_id":"auto"}`

---

## 4. 모듈 간 HTTP 의존성

| 호출자 | 대상 | 엔드포인트 |
|--------|------|------------|
| core | chat | `POST /chat` |
| core | live2d | `POST /live2d/emotion` |
| core | voice | `POST /voice/tts` (선택) |
| core | broadcast | `POST /broadcast/start`, `/stop` |
| broadcast | chat | `POST /chat` |
| broadcast | live2d | `POST /live2d/emotion`, `/live2d/broadcast` |
| broadcast | voice | `POST /voice/tts` |
| chat | Ollama | `POST /api/chat` (:11434) |
| chat | ChromaDB | HTTP (:8010, 선택) |

환경변수: `.env.example` 참고 (`AI_CHAT_URL`, `OLLAMA_MODEL` 등)

---

## 5. 분석 시 무시해도 되는 것

| 경로 | 이유 |
|------|------|
| `old/` | 단계별 실험 코드, import 없음 |
| `training/` | 학습 스크립트·체크포인트 (수 GB) |
| `chatbot/` | Vercel 배포용, modules와 독립 |
| `modules/chat/scripts/` | 파인튜닝 도구 |
| `modules/live2d/static/models/` | Live2D 바이너리·모션 JSON |
| 루트 `Modelfile.*` | Ollama 모델 생성용 |

---

## 6. 알려진 이슈 (분석 시 참고)

| 우선순위 | 이슈 | 위치 |
|----------|------|------|
| 중 | 랜덤 응답률 60% 하드코딩 | `chat_collector.py` `RANDOM_RESPONSE_RATE` |
| 중 | `chzzk_tokens.json` gitignore 미포함 | `.gitignore` |
| 중 | OAuth state 서버 검증 없음 | `chzzk_auth.py` |
| 낮음 | `get_live_status()` 항상 false | `chzzk_api.py` |
| 낮음 | core 파이프라인 립싱크 미연동 | `orchestrator.py` |

---

## 7. 외부 의존

| 서비스 | 포트 | 필수 여부 |
|--------|------|-----------|
| Ollama | 11434 | 필수 |
| ChromaDB | 8010 | RAG 켤 때만 |
| GPT-SoVITS API 서버 | 9880 | TTS 켤 때 (별도 프로세스, `python api_v2.py`) |
| 치지직 Open API | — | OAuth·공식 채팅 전송 시 |

---

다음: [ANALYSIS_README.md](./ANALYSIS_README.md) — 모듈별 읽기 순서
