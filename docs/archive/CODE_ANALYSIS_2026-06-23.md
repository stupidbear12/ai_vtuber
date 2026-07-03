# AI VTuber "시온(sion)" — 코드 분석 보고서

> 분석 일자: 2026-06-23  
> 분석 범위: `modules/` 전체 (core·live2d·chat·broadcast·voice·music) + `chatbot/`

---

## 1. 전체 아키텍처 및 모듈 간 의존성

### 1.1 시스템 구성도

```
┌──────────────────────────────────────────────────────────────────┐
│  외부 입력                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ 데스크톱 펫  │  │ Vercel 챗봇  │  │  치지직/유튜브 방송     │ │
│  │ (Electron)   │  │  (Next.js)   │  │  (시청자 채팅)           │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬────────────┘ │
└─────────┼─────────────────┼──────────────────────-─┼─────────────┘
          │                 │  WebSocket/REST          │
          ▼                 ▼                          ▼
┌─────────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  ai_live2d :8001    │  │ ai_chat API      │  │ ai_broadcast     │
│  - Live2D 웹뷰어    │  │ (route.js가 직접 │  │ :8003            │
│  - WebSocket 관리   │  │  Ollama 호출)    │  │ - 치지직 OAuth   │
│  - 표정/모션 제어   │  └──────────────────┘  │ - 채팅 수집      │
│  - /live2d/chat     │                         │ - 자동 모니터링  │
│    (ai_chat 직접    │                         └────────┬─────────┘
│     호출)           │                                  │
└─────────┬───────────┘                                  │
          │              ┌───────────────────────────────┘
          │              │
          ▼              ▼
┌────────────────────────────────────┐
│  ai_vtuber_core :8000 (오케스트)  │
│  - /status  (헬스 집계)            │
│  - /pipeline/chat                  │
│  - /broadcast/start·stop           │
└──────────────┬─────────────────────┘
               │
    ┌──────────┼──────────────┐
    ▼          ▼              ▼
┌───────┐  ┌───────┐  ┌──────────────┐
│ai_chat│  │ai_live│  │  ai_voice    │
│ :8002 │  │2d:8001│  │  :8004       │
│Ollama │  │WebSock│  │  ElevenLabs  │
│Gemini │  │ et    │  │  TTS         │
│RAG    │  └───────┘  └──────────────┘
└───────┘
    ▲
    │ HTTP 직접 호출 (오케스트레이터 우회)
    │
┌────────────────────────────────────┐
│  ai_broadcast, ai_live2d(/chat)    │
└────────────────────────────────────┘
```

### 1.2 포트 및 역할 요약

| 모듈 | 포트 | 핵심 역할 | 외부 의존 |
|------|------|-----------|-----------|
| `ai_vtuber_core` | 8000 | 오케스트레이터, 헬스 집계 | ai_chat, ai_live2d, ai_broadcast, ai_voice |
| `ai_live2d` | 8001 | Live2D 렌더링, WebSocket 브로드캐스트 | ai_chat (직접) |
| `ai_chat` | 8002 | LLM 응답 생성, RAG, 감정 파싱 | Ollama, Gemini API, ChromaDB |
| `ai_broadcast` | 8003 | 치지직/유튜브 채팅 수집, OAuth | ai_chat (직접), ai_live2d, ai_voice |
| `ai_voice` | 8004 | ElevenLabs TTS | ElevenLabs API |
| `ai_music` | 8005 | ACE-Step AI 음악 생성/DJ 믹싱 | ACE-Step (로컬 GPU) |
| `chatbot` | Vercel | 웹 챗봇 UI | Ollama (tunnel URL) |

---

## 2. 모듈별 역할 및 주요 파일

### 2.1 `modules/core` — 오케스트레이터

**주요 파일:**
- `app/main.py` — FastAPI 앱, `/pipeline/chat`, `/broadcast/start·stop` 엔드포인트
- `app/orchestrator.py` — 모듈 URL 관리(`ModuleConfig`), 채팅 파이프라인 실행, 헬스 체크 병렬 실행

**설계 특징:** `asyncio.gather()`로 4개 모듈 헬스 체크 병렬화. `/pipeline/chat`에서 chat → live2d → voice 순차 처리.

### 2.2 `modules/chat` — LLM 채팅 엔진

**주요 파일:**
- `app/chat_engine.py` — 시온 캐릭터 응답 생성, 감정 태그 파싱, RAG 컨텍스트 주입
- `app/llm_provider.py` — Ollama/Gemini 하이브리드 라우팅 (후원 → Gemini, 일반 → Ollama)
- `app/memory.py` — ChromaDB HTTP 클라이언트, 과거 대화 저장/검색, 지식 베이스 관리

**설계 특징:** `is_donation=True`인 중요 채팅만 Gemini로 라우팅해 비용 절감. RAG는 `CHAT_DISABLE_RAG=1`로 비활성화 가능.

### 2.3 `modules/broadcast` — 방송 채팅 수집

**주요 파일:**
- `app/main.py` — 치지직 OAuth 엔드포인트 포함, 30초 자동 방송 감지 루프
- `app/chat_collector.py` — `YouTubeChatCollector`, `ChzzkChatCollector`(비공식 WebSocket), `BroadcastChatManager`
- `app/chzzk_auth.py` — OAuth2 토큰 관리, `chzzk_tokens.json` 영속화
- `app/chzzk_api.py` — 치지직 공식 REST/Session API 클라이언트

**설계 특징:** 치지직 비공식 WebSocket과 공식 Session API 두 경로를 `CHZZK_CHANNEL_ID` 환경변수로 스위칭. `asyncio.Queue(maxsize=50)`으로 부하 제어.

### 2.4 `modules/live2d` — Live2D 아바타

**주요 파일:**
- `app/main.py` — `NoCacheMiddleware` 래핑, 정적 파일 마운트
- `app/router.py` — 표정/모션/립싱크/반응 REST API, `/live2d/chat`(ai_chat 직접 호출)
- `app/ws_manager.py` — WebSocket 클라이언트 풀, 실패 시 자동 제거

### 2.5 `modules/voice` — TTS

**주요 파일:**
- `app/voice_engine.py` — ElevenLabs SDK 래핑, 감정→VoiceSettings 매핑, 커스텀 음성 관리
- `app/main.py` — 동기/스트리밍 TTS, Voice Design API

### 2.6 `modules/music` — AI DJ

**주요 파일:**
- `app/main.py` — ACE-Step 음악 생성, DJ 큐, 오디오 믹서, WebSocket PCM 스트림
- `app/dj_controller.py` — 큐 기반 DJ 자동화, 크로스페이드 스킵
- `app/audio_mixer.py` — 오디오 재생/믹싱
- `app/music_engine.py` — ACE-Step 모델 래퍼

### 2.7 `chatbot/` — Vercel 웹 챗봇

- `app/api/chat/route.js` — Next.js API Route, Ollama 직접 호출 (ai_chat 모듈 미사용)

---

## 3. 코드 품질 이슈

### 3.1 시스템 프롬프트 중복 (심각)

동일한 시온 캐릭터 설정이 세 곳에 분산됨:

| 위치 | 비고 |
|------|------|
| `modules/chat/app/chat_engine.py:51` | 풀버전 (PET_SYSTEM_PROMPT) |
| `modules/chat/app/chat_engine.py:93` | 풀버전 (BROADCAST_SYSTEM_PROMPT) |
| `chatbot/app/api/chat/route.js:6` | 축약버전 — ai_chat과 **내용 불일치** |

chatbot은 오케스트레이터를 거치지 않고 Ollama를 직접 호출하므로, 프롬프트가 따로 관리되어 캐릭터 일관성이 깨질 수 있음.

**해결책:** 캐릭터 프롬프트를 공유 설정 파일로 추출하거나, chatbot이 ai_chat API를 호출하도록 변경.

### 3.2 aiohttp.ClientSession 매 요청 생성

`orchestrator.py`, `chat_collector.py`, `router.py` 등 모든 HTTP 호출에서 `async with aiohttp.ClientSession() as session:` 패턴을 사용. 요청마다 TCP 연결 생성/해제가 반복됨.

```python
# 현재 패턴 (매 요청마다 세션 생성)
async with aiohttp.ClientSession() as session:
    async with session.post(url, ...) as resp:
        ...

# orchestrator.py run_chat_pipeline: Step 1, 2, 3에서 각각 새 세션 생성
```

**해결책:** 앱 lifespan에서 공유 세션을 생성하고 의존성 주입으로 전달.

### 3.3 `on_event` deprecated 사용

`modules/broadcast/app/main.py`에서 FastAPI 0.93+ 이후 deprecated된 `@app.on_event("startup"/"shutdown")`을 사용:

```python
@app.on_event("startup")  # deprecated
async def _startup():
    ...
```

`modules/chat/app/main.py`와 `modules/music/app/main.py`는 이미 올바른 `lifespan` 패턴 사용 중. broadcast 모듈만 미전환.

### 3.4 NoCacheMiddleware 래핑 문제

`modules/live2d/app/main.py:111-112`:

```python
_fastapi_app = app
app = NoCacheMiddleware(_fastapi_app)  # 문제: app이 ASGIApp으로 교체됨
```

`uvicorn.run("app.main:app")`이 FastAPI 앱이 아닌 ASGIMiddleware 인스턴스를 참조하게 됨. OpenAPI 문서(`/docs`)나 FastAPI 내부 기능이 올바르게 동작하지 않을 수 있음.

**해결책:** `app.middleware()`로 등록하거나 `Middleware` 클래스로 추가.

### 3.5 모듈 레벨 전역 가변 상태

`modules/broadcast/app/main.py`:

```python
_manager: Optional[BroadcastChatManager] = None
_token_manager: Optional[ChzzkTokenManager] = None
_monitor_task: Optional[asyncio.Task] = None
_live_was_live: bool = False
```

동시 `/broadcast/start` 요청 시 race condition 가능. `_manager`를 교체하는 중에 다른 요청이 None 상태를 읽을 수 있음.

### 3.6 ElevenLabs 동기 함수의 이벤트 루프 블로킹

`modules/voice/app/voice_engine.py`의 `synthesize()`, `list_voices()`, `design_voice()`는 동기 함수이며, FastAPI 엔드포인트에서 `await` 없이 직접 호출됨:

```python
# voice/app/main.py:161
audio_bytes = engine.synthesize(req.text, ...)  # 동기 호출 — 이벤트 루프 블로킹
```

ElevenLabs API 응답 대기 중 다른 요청이 처리되지 못함.

**해결책:** `await asyncio.get_running_loop().run_in_executor(None, engine.synthesize, ...)`

### 3.7 처리 지연 누적 가능성

`BroadcastChatManager`의 응답 큐:
- `asyncio.Queue(maxsize=50)` — 최대 50개 적체
- Ollama 타임아웃: 60초/건
- 이론적 최대 지연: **50 × 60 = 3,000초(50분)**

실시간 방송에서 이전 채팅의 응답이 50분 뒤에 나올 수 있음.

**해결책:** 큐 maxsize를 5~10으로 축소하고, 오래된 채팅은 드롭.

---

## 4. 보안 취약점

### 4.1 OAuth CSRF state 검증 없음 (고위험)

`modules/broadcast/app/chzzk_auth.py`:

```python
def get_auth_url(self, state: str = "") -> Tuple[str, str]:
    if not state:
        state = secrets.token_urlsafe(16)  # state 생성
    # 하지만 어디에도 저장하지 않음
    return f"{CHZZK_AUTH_PAGE}?{params}", state

async def exchange_code(self, code: str, state: str = "") -> dict:
    # state를 받지만 검증하지 않음
    ...
```

콜백 엔드포인트(`/chzzk/auth/callback`)에서 state를 받지만, 서버에서 발급한 값과 비교 검증하지 않아 CSRF 공격에 취약.

**해결책:** state를 서버 세션(Redis 또는 인메모리 딕셔너리)에 저장하고 콜백에서 비교 후 삭제.

### 4.2 Access Token 평문 파일 저장

`modules/broadcast/app/chzzk_auth.py:31`:

```python
TOKEN_FILE = Path(__file__).parent / "chzzk_tokens.json"
```

`modules/broadcast/app/chzzk_tokens.json`이 git 미추적 파일로 존재 (gitignore 필요). Access Token + Refresh Token이 평문 JSON으로 저장됨.

**해결책:** `.gitignore`에 `**/chzzk_tokens.json` 추가 확인, 운영 환경에서는 OS 키체인 또는 시크릿 매니저 사용.

### 4.3 전체 모듈 CORS 와일드카드

5개 모든 FastAPI 모듈이 동일한 설정:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

로컬 개발 환경에서는 적합하지만, 운영 배포 시 외부에서 임의 오리진으로 API 호출 가능.

**해결책:** 환경변수 `ALLOWED_ORIGINS`로 허용 도메인 목록 관리.

### 4.4 디버그 엔드포인트 운영 노출

`modules/broadcast/app/main.py:360-373`:

```python
@app.get("/chzzk/live-status-raw")
async def chzzk_live_status_raw():
    """디버그용: /open/v1/lives raw 응답 그대로 반환."""
    import aiohttp, json
    from app.chzzk_auth import CHZZK_BASE_URL
    headers = _chzzk_client._client_headers()  # Client-Secret 포함 헤더 사용
    ...
```

Client-Secret이 담긴 헤더로 만든 응답을 외부에 그대로 노출.

**해결책:** 개발 환경에서만 활성화 (`DEBUG_MODE=1` 조건부), 또는 제거.

### 4.5 Rate Limiting 없음

`/chat`, `/voice/tts` 등 비용이 발생하는 엔드포인트에 rate limiting이 없음. 악의적 호출 시 Gemini API, ElevenLabs API 과금 발생 가능.

**해결책:** `slowapi` 라이브러리로 IP 기반 rate limiting 추가.

### 4.6 WebSocket 파라미터 직접 브로드캐스트

`modules/live2d/app/router.py:131-135`:

```python
@live2d_router.post("/params")
async def set_params(req: ParamRequest):
    await ws_manager.broadcast({"cmd": "set_params", "params": req.params})
```

`params`가 임의 dict이므로 브라우저 측 JavaScript에서 처리 시 잠재적 XSS.

---

## 5. 성능 병목 가능성

### 5.1 Ollama 응답 지연 (방송 모드)

방송 모드에서 Ollama 타임아웃 60초:
- 로컬 GPU에서 `sion` 모델 응답 속도에 따라 3~30초 소요
- 방송 채팅 반응에는 **3초 이내** 응답이 이상적
- 큐에 적체될 경우 수십 채팅이 응답되지 못함

### 5.2 ChromaDB 임베딩 지연

`memory.py`의 `DefaultEmbeddingFunction(onnxruntime)`:
- 첫 실행 시 모델 파일 다운로드 + 초기화 (~2~5초)
- `run_in_executor`로 처리하지만 스레드 풀 스로틀링 가능
- 방송 모드 RAG 타임아웃이 1초인데 ChromaDB 응답이 느리면 매번 타임아웃

### 5.3 ElevenLabs TTS 동기 블로킹

앞서 언급한 것처럼, TTS 요청 중(보통 2~5초) 전체 voice 서버가 응답 불가.

### 5.4 ACE-Step 음악 생성 GPU 경합

`ai_music`이 Ollama와 동일 GPU를 사용할 경우:
- 음악 생성 중 LLM 추론 속도 저하
- 두 모듈의 GPU 메모리 충돌 가능

### 5.5 치지직 WebSocket 재연결 루프

`ChzzkChatCollector`는 서버를 랜덤(`kr-ss1~9`) 선택해 연결하며, 실패 시 5초 후 재연결:

```python
n = random.randint(1, 9)
ws_url = f"wss://kr-ss{n}.chat.naver.com/chat"
```

연결 실패가 반복되면 로그가 과다하게 쌓이고, 방송 채팅이 수신되지 않음.

---

## 6. 개선 제안 (리팩토링 우선순위)

### 우선순위 1 — 보안 (즉시)

| 항목 | 파일 | 조치 |
|------|------|------|
| OAuth CSRF state 검증 | `chzzk_auth.py`, `broadcast/main.py` | state를 인메모리 dict에 저장, 콜백에서 검증 후 삭제 |
| 토큰 파일 gitignore | `modules/broadcast/app/.gitignore` | `chzzk_tokens.json` 추가 |
| 디버그 엔드포인트 제거 | `broadcast/main.py:360` | `/chzzk/live-status-raw` 삭제 또는 DEBUG 조건부 |
| Rate limiting | 모든 모듈 | `slowapi` 또는 nginx rate limit 적용 |

### 우선순위 2 — 버그/안정성 (단기)

| 항목 | 파일 | 조치 |
|------|------|------|
| NoCacheMiddleware 방식 수정 | `live2d/main.py:111` | `app.middleware` 방식으로 변경 |
| `on_event` → `lifespan` | `broadcast/main.py:75` | `asynccontextmanager` lifespan으로 전환 |
| ElevenLabs 동기 함수 | `voice_engine.py:159` | `run_in_executor` 래핑 |
| 전역 상태 race condition | `broadcast/main.py` | `asyncio.Lock()` 또는 클래스 인스턴스로 캡슐화 |
| 방송 큐 크기 축소 | `chat_collector.py:513` | `maxsize=50` → `maxsize=10`, 오래된 채팅 드롭 |

### 우선순위 3 — 성능 (중기)

| 항목 | 파일 | 조치 |
|------|------|------|
| aiohttp 세션 재사용 | `orchestrator.py`, `chat_collector.py` | lifespan에서 글로벌 세션 생성, DI로 전달 |
| CORS 운영 제한 | 모든 모듈 | `ALLOWED_ORIGINS` 환경변수로 화이트리스트 관리 |
| 방송 Ollama 타임아웃 단축 | `chat_collector.py:618` | 60초 → 15초, 응답 없으면 드롭 |
| GPU 모듈 분리 권고 | `ai_music`, Ollama | 별도 GPU 인스턴스 또는 스케줄링 |

### 우선순위 4 — 코드 품질 (장기)

| 항목 | 파일 | 조치 |
|------|------|------|
| 시스템 프롬프트 단일화 | `chat_engine.py`, `route.js` | 공유 설정 파일 또는 chatbot → ai_chat API 호출 전환 |
| 채팅 파이프라인 중복 제거 | core·broadcast·live2d | 오케스트레이터 경유 표준화 |
| 모듈 URL 의존 제거 | `chat_collector.py`, `router.py` | 모든 내부 호출을 core 오케스트레이터 경유 |
| ai_music core 통합 | `orchestrator.py`, `core/main.py` | 포트 8005 헬스 체크 및 라우팅 추가 |

---

## 7. 긍정적 평가

바이브코딩으로 짜여진 프로젝트임에도 다음 설계 결정은 잘 되어 있음:

- **Ollama-Gemini 하이브리드**: 후원 채팅만 Gemini로 라우팅해 비용 효율적
- **asyncio.gather() 병렬 헬스 체크**: 오케스트레이터의 4개 모듈 동시 체크
- **RAG 타임아웃 분리**: 방송(1초) vs 펫(2초) 모드별 타임아웃 차별화
- **ThreadSafe Queue 삽입**: `call_soon_threadsafe`로 pytchat 스레드→asyncio 큐 안전 전달
- **ChzzkChatManager 재연결 로직**: WebSocket 끊김 시 자동 재연결
- **감정 태그 이중 포맷 지원**: `[감정:happy]` (Ollama) + `[happy]` (Gemini) 모두 파싱
- **지연 import 패턴**: `chromadb`, `sentence_transformers` 등 heavy 모듈을 함수 내 지연 import로 서버 시작 속도 보호
