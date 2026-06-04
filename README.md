# AI 버튜버 에메스(emeth)

> Google Gemini 기반 AI 버튜버 시스템 — 모노레포 구조

에메스(emeth)는 Live2D 아바타, AI 채팅 엔진, 방송 채팅 연동, 음성 합성을 갖춘 AI 버튜버입니다.  
히브리어로 **"진실"** 을 뜻하는 이름처럼, 시청자와 솔직하고 진심 어린 대화를 나눕니다.

---

## 모듈 구성

```
ai_vtuber/
├── modules/
│   ├── live2d/        포트 8001 — Live2D 아바타 서버 (웹뷰어, WebSocket, Electron 펫)
│   ├── chat/          포트 8002 — Google Gemini 채팅 엔진 (에메스 캐릭터)
│   ├── broadcast/     포트 8003 — 치지직/유튜브 방송 채팅 수집
│   ├── voice/         포트 8004 — 음성 합성 (ElevenLabs 연동 예정)
│   └── core/          포트 8000 — 오케스트레이터 (메인 API, 전체 상태 관리)
├── docker-compose.yml 전체 모듈 한번에 실행
├── .env.example       환경변수 템플릿
├── requirements.txt   전체 의존성 (로컬 개발용)
└── old/               이전 레거시 코드
```

### 모듈별 역할

| 모듈 | 포트 | 역할 |
|------|------|------|
| `core` | 8000 | 오케스트레이터 — 전체 상태 모니터링, 통합 채팅 파이프라인, 방송 시작/중지 |
| `live2d` | 8001 | Live2D 웹 뷰어, WebSocket 실시간 제어, 표정/모션/립싱크 API, Electron 데스크톱 펫 |
| `chat` | 8002 | Gemini API 기반 에메스 캐릭터 응답 생성 (pet/broadcast 두 가지 모드) |
| `broadcast` | 8003 | 치지직/유튜브 라이브 채팅 수집, 에메스 반응 자동화 |
| `voice` | 8004 | 음성 합성 (현재 스텁 — ElevenLabs 연동 예정) |

---

## 시스템 아키텍처

```
[시청자 채팅]
  치지직 WebSocket / 유튜브 pytchat
        ↓
  modules/broadcast (8003)
        ↓ POST /chat
  modules/chat (8002) — Gemini API
        ↓ {reply, emotion}
  modules/live2d (8001) — POST /live2d/emotion
        ↓ WebSocket broadcast
  [브라우저 / Electron 펫]

[데스크톱 펫]
  사용자 입력 → modules/core (8000) → modules/chat → modules/live2d
```

---

## 빠른 시작 (Docker Compose)

### 1. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 GEMINI_API_KEY 입력
```

### 2. 전체 실행

```bash
docker-compose up -d
```

### 3. 서비스 확인

| URL | 설명 |
|-----|------|
| http://localhost:8000 | 오케스트레이터 대시보드 |
| http://localhost:8000/status | 전체 모듈 상태 |
| http://localhost:8001/live2d/static/ | Live2D 웹 뷰어 |
| http://localhost:8000/docs | 통합 API 문서 (Swagger) |

---

## 로컬 개발 (모듈별 실행)

각 모듈을 독립적으로 실행할 수 있습니다. 터미널을 5개 열어 각각 실행:

```bash
# 터미널 1 — Live2D 아바타 서버
cd modules/live2d
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# 터미널 2 — 채팅 엔진
cd modules/chat
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002

# 터미널 3 — 방송 채팅 수집
cd modules/broadcast
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8003

# 터미널 4 — 음성 합성 (스텁)
cd modules/voice
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8004

# 터미널 5 — 오케스트레이터
cd modules/core
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Electron 데스크톱 펫 실행

```bash
# live2d 서버가 먼저 실행 중이어야 합니다 (포트 8001)
cd modules/live2d/electron
npm install
npm start
```

---

## 주요 API

### 채팅 파이프라인 (core)

```bash
# 에메스에게 채팅 — 응답 생성 + 표정 자동 변경
curl -X POST http://localhost:8000/pipeline/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕 에메스!", "mode": "pet"}'
```

### 방송 채팅 수집 (core 경유)

```bash
# 치지직 방송 시작
curl -X POST http://localhost:8000/broadcast/start \
  -H "Content-Type: application/json" \
  -d '{"platform": "chzzk", "channel_id": "your_channel_hash_id"}'

# 유튜브 라이브 시작
curl -X POST http://localhost:8000/broadcast/start \
  -H "Content-Type: application/json" \
  -d '{"platform": "youtube", "channel_id": "your_video_id"}'

# 수집 중지
curl -X POST http://localhost:8000/broadcast/stop
```

### Live2D 직접 제어

```bash
# 표정 변경
curl -X POST http://localhost:8001/live2d/emotion \
  -d '{"emotion": "happy"}'

# 모션 재생
curl -X POST http://localhost:8001/live2d/motion \
  -d '{"group": "Idle", "index": 0}'
```

---

## 환경변수

| 변수명 | 필수 | 기본값 | 설명 |
|--------|------|--------|------|
| `GEMINI_API_KEY` | 필수 | — | Google Gemini API 키 |
| `GEMINI_MODEL` | 선택 | `gemini-2.5-flash` | Gemini 모델명 |
| `AI_LIVE2D_URL` | 선택 | `http://localhost:8001` | live2d 모듈 URL |
| `AI_CHAT_URL` | 선택 | `http://localhost:8002` | chat 모듈 URL |
| `AI_BROADCAST_URL` | 선택 | `http://localhost:8003` | broadcast 모듈 URL |
| `AI_VOICE_URL` | 선택 | `http://localhost:8004` | voice 모듈 URL |

---

## 개발 현황

| 기능 | 상태 |
|------|------|
| Live2D 웹 뷰어 (Haru 모델) | 완료 |
| WebSocket 실시간 제어 | 완료 |
| Electron 데스크톱 펫 | 완료 |
| Google Gemini 채팅 엔진 | 완료 |
| 에메스 캐릭터 시스템 프롬프트 | 완료 |
| 치지직 채팅 연동 | 완료 |
| 유튜브 채팅 연동 | 완료 |
| 오케스트레이터 API | 완료 |
| 음성 합성 (ElevenLabs) | 예정 |

---

## 라이선스

MIT License
