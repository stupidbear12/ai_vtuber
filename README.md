# AI 버튜버 시온(sion)

> Ollama 기반 AI 버튜버 시스템 — FastAPI 마이크로서비스 모노레포

시온(sion)은 Live2D 아바타, Ollama 채팅 엔진(RAG 메모리), 방송 채팅 연동, ElevenLabs 음성 합성, AI DJ 음악 생성을 갖춘 AI 버튜버 프로젝트입니다.

---

## 모듈 구성

```
ai_vtuber/
├── modules/
│   ├── core/        포트 8000 — 오케스트레이터 (헬스, 통합 파이프라인)
│   ├── live2d/      포트 8001 — Live2D 웹뷰어, WebSocket, Electron 펫
│   ├── chat/        포트 8002 — Ollama 채팅 엔진 + ChromaDB RAG
│   ├── broadcast/   포트 8003 — 치지직/유튜브 방송 채팅 수집
│   ├── voice/       포트 8004 — ElevenLabs TTS / Voice Design
│   └── music/       포트 8005 — ACE-Step AI DJ (선택 모듈)
├── docker-compose.yml
├── start-all.bat / stop-all.bat   로컬 5모듈 일괄 실행
├── start-pet.bat                    Electron 데스크톱 펫
├── open-obs-viewer.ps1              OBS URL 미리보기
├── check-ollama.bat                 Ollama 설치 확인
├── .env.example                     환경변수 템플릿
└── old/                             레거시 코드
```

### 모듈별 역할

| 모듈 | 포트 | 역할 |
|------|------|------|
| `core` | 8000 | 오케스트레이터 — 전체 상태, 통합 채팅 파이프라인, 방송 start/stop |
| `live2d` | 8001 | Live2D 웹 뷰어, WebSocket 실시간 제어, 표정/모션/립싱크, Electron 펫 |
| `chat` | 8002 | Ollama 기반 시온 응답 (pet/broadcast 모드), ChromaDB RAG 메모리 |
| `broadcast` | 8003 | 치지직/유튜브 채팅 수집, 시온 자동 반응, (선택) TTS |
| `voice` | 8004 | ElevenLabs TTS, 스트리밍, Voice Design, 감정별 음성 파라미터 |
| `music` | 8005 | ACE-Step 음악 생성, AI DJ 자동 선곡, 크로스페이드 믹싱 (선택) |
| `chromadb` | 8010 | RAG 벡터 DB (Docker, chat 모듈 전용) |

---

## 시스템 아키텍처

```
[방송 채팅]
  치지직 WebSocket / 유튜브 pytchat
        ↓
  broadcast (8003) — 채팅 선별 (키워드·후원·랜덤)
        ↓ POST /chat (mode=broadcast)
  chat (8002) — Ollama + RAG(ChromaDB)
        ↓ {reply, emotion}
  live2d (8001) — POST /live2d/emotion
        ↓ WebSocket broadcast
  [OBS 브라우저 / Electron 펫]

[데스크톱 펫]
  사용자 입력 → live2d (8001) /live2d/chat
        ↓ POST /chat (mode=pet)
  chat (8002) → live2d WebSocket 표정 반영

[통합 파이프라인 — core]
  POST /pipeline/chat (with_voice=true)
        ↓ chat → live2d emotion → voice TTS (audio_base64)

[AI DJ — music, 선택]
  POST /music/queue → ACE-Step 생성 → AudioMixer 재생
        ↓ WebSocket /music/stream (PCM)
  [OBS / 클라이언트]
```

> **참고:** OBS와 Electron 펫은 같은 `live2d` WebSocket 허브(`ws://localhost:8001/live2d/ws`)에 연결됩니다. 동시에 켜면 표정/모션이 함께 반영됩니다.

---

## 빠른 시작

### 1. 환경변수 설정

```bash
cp .env.example .env
# .env 에 Ollama 모델명, ElevenLabs API 키 등 입력
```

**Ollama (로컬 LLM, 무료):**

1. https://ollama.com 설치 후 **시작 메뉴 → Ollama** 실행
2. 모델 확인 및 생성:

```cmd
check-ollama.bat
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" list
```

`.env`에 모델명 지정:

```
OLLAMA_MODEL=sion
```

커스텀 모델이 없으면 `ollama create sion -f modules/chat/scripts/Modelfile` 로 생성하거나, 설치된 모델명을 사용하세요.

**ChromaDB (RAG 메모리, 선택):**

```bash
docker-compose up -d chromadb
```

로컬 개발 시 `.env`:

```
CHROMA_HOST=localhost
CHROMA_PORT=8010
CHAT_DISABLE_RAG=0
```

RAG 없이 실행하려면 `CHAT_DISABLE_RAG=1`.

**ElevenLabs (voice 모듈, 선택):**

```
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=          # /voice/design 으로 생성 후 설정
```

### 2. 로컬 실행 (추천)

```cmd
cd ai_vtuber
start-all.bat
```

종료:

```cmd
stop-all.bat
```

PowerShell:

```powershell
.\start-all.ps1
.\stop-all.ps1
.\start-pet.ps1
.\open-obs-viewer.ps1
```

> cmd에서 `.\start-all.ps1`을 입력하면 메모장만 열릴 수 있습니다. **`.bat` 파일을 사용**하세요.

### 3. Docker Compose

```bash
docker-compose up -d
```

ChromaDB + 5개 모듈이 함께 기동됩니다. Ollama는 호스트에서 `ollama serve` 실행이 필요합니다.

### 4. 서비스 확인

| URL | 설명 |
|-----|------|
| http://localhost:8000 | 오케스트레이터 대시보드 |
| http://localhost:8000/status | 전체 모듈 헬스 |
| http://localhost:8001/live2d/static/ | Live2D 웹 뷰어 |
| http://localhost:8002/docs | chat API (Swagger) |
| http://localhost:8004/docs | voice API (Swagger) |
| http://localhost:8005/docs | music API (Swagger, 별도 실행 시) |

---

## Electron 데스크톱 펫

화면 위 투명 창으로 시온과 채팅할 수 있습니다.

```cmd
start-all.bat
start-pet.bat
```

| 조작 | 동작 |
|------|------|
| 캐릭터 클릭 | 채팅창 열기/닫기 |
| 더블클릭 | 마우스 통과 모드 |
| 트레이 아이콘 우클릭 | 표시/숨기기, 통과 모드, 종료 |
| 드래그 | 창 위치 이동 |

기본 모델: **mao_pro** (브라우저 뷰어와 동일)

수동 실행:

```bash
cd modules/live2d/electron
npm install
npm start
```

---

## OBS 방송 송출

Live2D 서버(8001)가 실행 중이어야 합니다.

### 브라우저 소스 URL

| 용도 | URL |
|------|-----|
| **투명 배경 (추천)** | `http://localhost:8001/live2d/static/?transparent=1` |
| 크로마키 (녹색) | `http://localhost:8001/live2d/static/?chromakey=1` |
| mao_pro + 투명 | `http://localhost:8001/live2d/static/?transparent=1&model=models/mao_pro/runtime/mao_pro.model3.json` |
| Haru + 투명 | `http://localhost:8001/live2d/static/?transparent=1&model=models/Haru/Haru.model3.json` |

```powershell
.\open-obs-viewer.ps1
```

### OBS 설정

1. **소스 추가** → **브라우저**
2. URL에 위 주소 붙여넣기 (`transparent=1` 권장)
3. **너비** `1920` / **높이** `1080`, **30 FPS**
4. **투명 배경** 체크

### 방송 채팅 연동

```bash
# 치지직
curl -X POST http://localhost:8000/broadcast/start \
  -H "Content-Type: application/json" \
  -d '{"platform": "chzzk", "channel_id": "your_channel_id"}'

# 유튜브
curl -X POST http://localhost:8000/broadcast/start \
  -H "Content-Type: application/json" \
  -d '{"platform": "youtube", "channel_id": "your_video_id"}'

# 중지
curl -X POST http://localhost:8000/broadcast/stop
```

**치지직 공식 API (OAuth):** `modules/broadcast/.env.example` 참고.  
`CHZZK_CLIENT_ID`, `CHZZK_CLIENT_SECRET` 설정 후 `http://localhost:8003/chzzk/auth` 로 인증.

---

## 주요 API

### 통합 채팅 (core)

```bash
# 응답 + 표정 변경
curl -X POST http://localhost:8000/pipeline/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕 시온!", "mode": "pet"}'

# TTS 포함 (ElevenLabs API 키 필요)
curl -X POST http://localhost:8000/pipeline/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕 시온!", "mode": "pet", "with_voice": true}'
```

### ElevenLabs TTS (voice)

```bash
# 음성 생성 (Voice Design)
curl -X POST http://localhost:8004/voice/design \
  -H "Content-Type: application/json" \
  -d '{"name":"시온","description":"밝고 귀여운 20대 여성, 약간 높은 톤, 한국어"}'

# 기본 음성 설정
curl -X POST http://localhost:8004/voice/set-default \
  -H "Content-Type: application/json" \
  -d '{"voice_id":"<voice_id>","voice_name":"시온"}'

# TTS
curl -X POST http://localhost:8004/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"안녕! 나는 시온이야","emotion":"happy"}' \
  --output speech.mp3
```

### AI DJ (music, 별도 실행)

```bash
cd modules/music
set ACESTEP_STUB=1          # ACE-Step 없이 더미 wav 테스트
uvicorn app.main:app --host 0.0.0.0 --port 8005

# 곡 생성
curl -X POST http://localhost:8005/music/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"lo-fi chill beats","duration":30,"play":true}'

# DJ 큐 추가
curl -X POST http://localhost:8005/music/queue \
  -H "Content-Type: application/json" \
  -d '{"prompt":"upbeat EDM","priority":5}'
```

### Live2D 직접 제어

```bash
curl -X POST http://localhost:8001/live2d/emotion \
  -H "Content-Type: application/json" \
  -d '{"emotion": "happy"}'

curl -X POST http://localhost:8001/live2d/motion \
  -H "Content-Type: application/json" \
  -d '{"group": "", "index": 1}'
```

---

## 환경변수

| 변수명 | 필수 | 기본값 | 설명 |
|--------|------|--------|------|
| `OLLAMA_BASE_URL` | 선택 | `http://localhost:11434` | Ollama API |
| `OLLAMA_MODEL` | 선택 | `sion` | Ollama 모델명 |
| `CHROMA_HOST` | 선택 | `localhost` | ChromaDB 호스트 (RAG) |
| `CHROMA_PORT` | 선택 | `8010` | ChromaDB 포트 |
| `CHAT_DISABLE_RAG` | 선택 | `0` | `1`이면 RAG 비활성화 |
| `ELEVENLABS_API_KEY` | voice용 | — | ElevenLabs API 키 |
| `ELEVENLABS_VOICE_ID` | 선택 | — | 기본 음성 ID |
| `CHZZK_CLIENT_ID` | broadcast용 | — | 치지직 OAuth |
| `CHZZK_CLIENT_SECRET` | broadcast용 | — | 치지직 OAuth |
| `BROADCAST_VOICE_ENABLED` | 선택 | `true` | 방송 TTS 활성화 |
| `AI_LIVE2D_URL` | 선택 | `http://localhost:8001` | live2d 모듈 URL |
| `AI_CHAT_URL` | 선택 | `http://localhost:8002` | chat 모듈 URL |
| `AI_BROADCAST_URL` | 선택 | `http://localhost:8003` | broadcast 모듈 URL |
| `AI_VOICE_URL` | 선택 | `http://localhost:8004` | voice 모듈 URL |
| `ACESTEP_STUB` | 선택 | — | `1`이면 music 더미 wav 모드 |
| `ACESTEP_MODEL` | 선택 | `acestep-v15-turbo` | ACE-Step 모델 |

---

## RAG 메모리 (chat)

`chat` 모듈은 ChromaDB로 두 가지 기억을 관리합니다.

| 컬렉션 | 내용 |
|--------|------|
| 대화 기억 | 과거 Q&A 검색 후 프롬프트에 주입 |
| 지식 베이스 | `modules/chat/data/knowledge/*.md` (시온 프로필, FAQ, DJ 방송 가이드 등) |

지식 베이스 파일:

- `sion_profile.md` — 캐릭터 설정
- `sion_preferences.md` — 취향
- `sion_faq.md` — FAQ
- `sion_dj_broadcast.md` — DJ 방송 가이드

---

## 로컬 개발 (모듈별 실행)

`start-all.bat` 대신 모듈을 개별 실행할 때:

```bash
# ChromaDB (RAG 사용 시)
docker-compose up -d chromadb

# 각 모듈 — 터미널별로 실행
cd modules/live2d   && uvicorn app.main:app --reload --port 8001
cd modules/chat     && uvicorn app.main:app --reload --port 8002
cd modules/broadcast && uvicorn app.main:app --reload --port 8003
cd modules/voice    && uvicorn app.main:app --reload --port 8004
cd modules/core     && uvicorn app.main:app --reload --port 8000
cd modules/music    && uvicorn app.main:app --reload --port 8005
```

로그 파일: `logs_<module>.txt`, `logs_<module>_err.txt` (start-all 실행 시)

---

## 개발 현황

| 기능 | 상태 |
|------|------|
| Live2D 웹 뷰어 (mao_pro, Haru) | 완료 |
| WebSocket 실시간 제어 | 완료 |
| Electron 데스크톱 펫 | 완료 |
| Ollama 채팅 엔진 (pet/broadcast) | 완료 |
| ChromaDB RAG 메모리 | 완료 |
| 시온 캐릭터 지식 베이스 | 완료 |
| 치지직/유튜브 채팅 연동 | 완료 |
| 치지직 공식 API OAuth | 완료 |
| 오케스트레이터 API | 완료 |
| ElevenLabs TTS / Voice Design | 완료 |
| TTS → Live2D 립싱크 연동 | 미구현 |
| 펫/broadcast TTS 자동 재생 | 미구현 |
| ACE-Step AI DJ (music) | 구현 중 (스텁 모드 가능) |
| music 모듈 start-all 포함 | 미구현 |

---

## 라이선스

MIT License
