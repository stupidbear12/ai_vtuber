# AI VTuber 시온(sion)

> Ollama + Live2D + ElevenLabs 기반 AI DJ VTuber — 치지직(Chzzk) 라이브 방송

시온(sion)은 치지직에서 라이브 방송하며 시청자와 채팅으로 소통하는 AI DJ VTuber입니다.
파인튜닝된 LLM이 캐릭터 응답을 생성하고, ElevenLabs TTS로 음성을 합성하며, Live2D 아바타가 감정에 맞춰 표정과 립싱크를 동기화합니다.

---

## 시스템 아키텍처

```
[치지직 시청자 채팅]
      ↓ Chzzk Session API (WebSocket) + 공식 API (OAuth)
  broadcast (8003) — 채팅 수집 + 선별 (키워드·후원·랜덤)
      ↓ POST /chat (viewer_name 포함)
  chat (8002) — Ollama LLM (파인튜닝 sion 모델) + ChromaDB RAG
      ↓ {reply, emotion}
      ├→ voice (8004) — ElevenLabs TTS → 음성 합성
      ├→ live2d (8001) — 감정 태그 → 표정 변경 + 립싱크
      └→ Chzzk 공식 API — 채팅창에 시온 응답 전송
              ↓
  [OBS 브라우저 소스] → 방송 송출

[데스크톱 펫]
  사용자 입력 → live2d (8001) /live2d/chat
        ↓ POST /chat (mode=pet)
  chat (8002) → live2d WebSocket 표정 반영

[통합 파이프라인 — core]
  POST /pipeline/chat (with_voice=true)
        ↓ chat → live2d emotion → voice TTS (audio_base64)

[AI DJ — music, 개발중]
  POST /music/queue → ACE-Step 생성 → AudioMixer 재생
        ↓ WebSocket /music/stream (PCM)
```

---

## 모듈 구성

```
ai_vtuber/
├── modules/
│   ├── core/        8000 — 오케스트레이터 (상태 관리, 통합 파이프라인)
│   ├── live2d/      8001 — Live2D 웹뷰어, WebSocket, Electron 펫
│   ├── chat/        8002 — Ollama 채팅 엔진 + ChromaDB RAG 메모리
│   ├── broadcast/   8003 — 치지직/유튜브 채팅 수집 + 시온 반응 자동화
│   ├── voice/       8004 — ElevenLabs TTS / Voice Design
│   └── music/       8005 — ACE-Step AI DJ (개발중)
├── ACE-Step-1.5/              AI 음악 생성 엔진
├── llama.cpp/                 GGUF 모델 변환/추론 도구
├── Modelfile.from-chat        Ollama 모델 설정 (파인튜닝 sion)
├── docker-compose.yml         전체 모듈 일괄 실행
├── start-all.bat / .ps1       로컬 일괄 실행 스크립트
├── .env.example               환경변수 템플릿
└── .env                       실제 환경변수 (git 미추적)
```

| 모듈 | 포트 | 역할 |
|------|------|------|
| `core` | 8000 | 오케스트레이터 — 헬스 체크, 통합 채팅 파이프라인 |
| `live2d` | 8001 | Live2D 웹 뷰어, WebSocket 실시간 제어, 표정/모션/립싱크, Electron 펫 |
| `chat` | 8002 | Ollama 기반 시온 응답 (pet/broadcast 모드), ChromaDB RAG 메모리 |
| `broadcast` | 8003 | 치지직/유튜브 채팅 수집, 시온 자동 반응, 시청자 닉네임 전달 |
| `voice` | 8004 | ElevenLabs TTS, Voice Design, 감정별 음성 파라미터 |
| `music` | 8005 | ACE-Step 음악 생성, AI DJ 자동 선곡, 크로스페이드 믹싱 (개발중) |

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| LLM | Ollama + Llama 3.1 8B 파인튜닝 (Q8_0 GGUF) |
| TTS | ElevenLabs API (Voice Design + 감정별 파라미터) |
| 아바타 | Live2D Cubism SDK (mao_pro 모델) |
| 방송 플랫폼 | 치지직(Chzzk) — 공식 API (OAuth) + Session API (WebSocket) |
| RAG | ChromaDB 벡터 DB (대화 기억 + 캐릭터 지식 베이스) |
| 백엔드 | FastAPI (Python 3.11), 마이크로서비스 6모듈 구조 |
| 음악 생성 | ACE-Step 1.5 (개발중) |
| 방송 도구 | OBS Studio (브라우저 소스) |
| GPU | NVIDIA GeForce RTX 4060 Ti |

---

## 빠른 시작

### 1. 환경변수

```bash
cp .env.example .env
# .env에 OLLAMA_MODEL, ELEVENLABS_API_KEY, CHZZK_CLIENT_ID 등 설정
```

### 2. Ollama 모델 준비

```bash
# Ollama 설치: https://ollama.com
# 파인튜닝 모델 생성
ollama create sion -f Modelfile.from-chat

# 확인
ollama list
```

### 3. 전체 실행

```cmd
# cmd (권장)
start-all.bat

# PowerShell
.\start-all.ps1
```

> cmd에서 `.ps1`을 실행하면 메모장만 열립니다. `.bat` 파일을 사용하세요.

### 4. 방송 시작

```bash
curl -X POST http://localhost:8003/broadcast/start \
  -H "Content-Type: application/json" \
  -d '{"platform": "chzzk", "channel_id": "auto"}'
```

### 5. OBS 설정

브라우저 소스에 아래 URL 추가 (1920x1080, 투명 배경 체크):

```
http://localhost:8001/live2d/static/?transparent=1&model=models/mao_pro/runtime/mao_pro.model3.json
```

```powershell
.\open-obs-viewer.ps1   # URL 목록 + 브라우저 미리보기
```

### 6. 서비스 확인

| URL | 설명 |
|-----|------|
| http://localhost:8000/status | 전체 모듈 헬스 |
| http://localhost:8001/live2d/static/ | Live2D 웹 뷰어 |
| http://localhost:8002/docs | chat API (Swagger) |
| http://localhost:8003/broadcast/status | 방송 채팅 수집 상태 |
| http://localhost:8004/docs | voice API (Swagger) |

---

## 주요 API

### 통합 채팅 (core)

```bash
curl -X POST http://localhost:8000/pipeline/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕 시온!", "mode": "pet", "with_voice": true}'
```

### 방송 수집 (broadcast)

```bash
curl -X POST http://localhost:8003/broadcast/start \
  -d '{"platform": "chzzk", "channel_id": "auto"}'
curl -X POST http://localhost:8003/broadcast/stop
curl http://localhost:8003/broadcast/status
```

### 채팅 (chat)

```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕 시온!", "mode": "broadcast", "viewer_name": "시청자닉네임"}'
```

### Live2D 제어

```bash
curl -X POST http://localhost:8001/live2d/emotion -d '{"emotion": "happy"}'
curl -X POST http://localhost:8001/live2d/motion -d '{"group": "", "index": 1}'
```

### TTS (voice)

```bash
curl -X POST http://localhost:8004/voice/tts \
  -d '{"text":"안녕! 나는 시온이야","emotion":"happy"}' --output speech.mp3
```

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

---

## 환경변수

| 변수명 | 필수 | 기본값 | 설명 |
|--------|------|--------|------|
| `OLLAMA_BASE_URL` | 선택 | `http://localhost:11434` | Ollama API |
| `OLLAMA_MODEL` | 선택 | `sion` | Ollama 모델명 |
| `CHROMA_HOST` | 선택 | `localhost` | ChromaDB 호스트 |
| `CHROMA_PORT` | 선택 | `8010` | ChromaDB 포트 |
| `CHAT_DISABLE_RAG` | 선택 | `0` | `1`이면 RAG 비활성화 |
| `ELEVENLABS_API_KEY` | voice용 | — | ElevenLabs API 키 |
| `ELEVENLABS_VOICE_ID` | 선택 | — | 기본 음성 ID |
| `CHZZK_CLIENT_ID` | broadcast용 | — | 치지직 Open API Client ID |
| `CHZZK_CLIENT_SECRET` | broadcast용 | — | 치지직 Open API Client Secret |
| `BROADCAST_VOICE_ENABLED` | 선택 | `true` | 방송 TTS 활성화 |

---

## RAG 메모리

`chat` 모듈은 ChromaDB로 두 가지 기억을 관리합니다.

| 컬렉션 | 내용 |
|--------|------|
| 대화 기억 | 과거 Q&A 검색 후 프롬프트에 주입 |
| 지식 베이스 | `modules/chat/data/knowledge/*.md` |

지식 베이스 파일: `sion_profile.md` (캐릭터 설정), `sion_preferences.md` (취향), `sion_faq.md` (FAQ), `sion_dj_broadcast.md` (DJ 방송 가이드)

---

## 개발 현황

### 완료

- [x] Live2D 웹 뷰어 (mao_pro, Haru 모델 선택)
- [x] WebSocket 실시간 표정/모션 제어
- [x] Electron 데스크톱 펫
- [x] Ollama 채팅 엔진 (pet/broadcast 2모드)
- [x] Llama 3.1 8B 파인튜닝 (시온 캐릭터, Q8_0 GGUF)
- [x] 감정 태그 파싱 → Live2D 표정 매핑 (10종)
- [x] ChromaDB RAG 메모리 (대화 기억 + 지식 베이스)
- [x] 치지직 채팅 수집 (Session API WebSocket)
- [x] 치지직 공식 API 연동 (OAuth, 채팅 전송, 후원/구독 감지)
- [x] ElevenLabs TTS 음성 합성 + 립싱크
- [x] OBS 방송 송출 파이프라인
- [x] 시청자 닉네임 호칭 (viewer_name 전달)
- [x] 오케스트레이터 통합 파이프라인 (core)
- [x] FastAPI 마이크로서비스 6모듈 구조

### 진행 예정

**Phase 2 — LLM 품질 개선**
- [ ] 파인튜닝 데이터셋 확충 (500~1000개 대화쌍)
- [ ] 한국어 베이스 모델 재검토 (EXAONE, SOLAR)
- [ ] 방송 로그 자동 저장 → 학습 데이터 축적
- [ ] LoRA/QLoRA 파인튜닝 자동화 스크립트
- [ ] RAG 메모리 실전 활성화 (시청자 기억)
- [ ] 반복 응답 방지 (대화 히스토리 관리 강화)

**Phase 3 — DJ 기능**
- [ ] ACE-Step 음악 생성 서버 연동 (8006)
- [ ] 시청자 요청곡 큐 시스템
- [ ] 크로스페이드 / 자동 DJ 믹싱
- [ ] 채팅 명령어로 곡 스킵/요청
- [ ] OBS 오디오 소스 통합
- [ ] 음악 비주얼라이저 오버레이

**Phase 4 — 안정성 / 운영**
- [ ] 방송 자동 감지 (get_live_status 구현)
- [ ] 모듈 자동 복구 (watchdog / 프로세스 매니저)
- [ ] 에러 알림 (Discord/카톡 웹훅)
- [ ] 로그 대시보드 (응답률, 에러율)
- [ ] 원클릭 방송 시작/종료

**Phase 5 — 수익화 / 확장**
- [ ] 오리지널 Live2D 모델 커미션 (상업 라이선스)
- [ ] 유튜브 클립 채널 자동화
- [ ] 멀티 플랫폼 동시 방송 (유튜브 + 치지직)
- [ ] 시온 TTS 음성 클론 (ElevenLabs Voice Cloning)

**Phase 6 — 고도화**
- [ ] 게임 플레이 AI (osu!, 마인크래프트 등)
- [ ] 실시간 음성 대화 (STT → LLM → TTS)
- [ ] 3D 모델 전환 (VRM)
- [ ] 자체 LLM 반복 파인튜닝 자동화

---

## 라이선스

- 코드: MIT License
- LLM: [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (상업 이용 가능, "Built with Llama" 표기 필요)
- Live2D: mao_pro 모델 라이선스 별도 확인 필요
- TTS: ElevenLabs 유료 플랜 라이선스
