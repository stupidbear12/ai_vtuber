# 영상 분석 + 프로젝트 적용 제안

## 1. YouTube 영상 요약

**제목:** AI 버튜버 와이프 만들기  
**채널:** 코딩애플 | **게시일:** 2025-11-11 | **조회수:** 318,703  
**핵심 소개 프로젝트:** [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)

영상은 Open-LLM-VTuber라는 오픈소스 프로젝트를 소개하며, 누구나 로컬에서 AI VTuber를 만들 수 있다는 내용을 다룹니다. 6분 37초의 짧은 영상으로, 설치부터 실행까지의 과정을 보여줍니다.

### Open-LLM-VTuber의 핵심 기술/기능

| 기능 | 설명 |
|------|------|
| **실시간 음성 대화** | ASR(음성인식) + LLM + TTS를 연결한 핸즈프리 음성 인터랙션 |
| **음성 인터럽트** | AI가 말하는 도중에 끼어들 수 있음 (헤드폰 없이도 자기 목소리 안 들음) |
| **Live2D 아바타** | 감정 매핑으로 표정 자동 제어, 립싱크 지원 |
| **비전 기능** | 카메라/화면 캡처로 AI가 사용자와 화면을 볼 수 있음 |
| **데스크톱 펫 모드** | 투명 배경 + 항상 위 + 마우스 클릭 통과 |
| **터치 피드백** | 클릭/드래그로 모델과 상호작용 |
| **AI 자발적 발화** | AI가 먼저 말을 거는 기능 |
| **채팅 기록 영속화** | 이전 대화 이어가기 가능 |
| **장기 기억 (Letta)** | Letta 기반 장기 메모리 (v1.2.0+) |
| **MCP 지원** | Model Context Protocol로 외부 도구 연동 |
| **WebSocket 기반 아키텍처** | FastAPI 백엔드 + React 프론트엔드 |
| **완전 오프라인 실행** | 로컬 모델만으로 전체 파이프라인 실행 가능 |
| **모듈식 설계** | LLM/ASR/TTS를 설정 파일만으로 교체 가능 |
| **Bilibili 단막 클라이언트** | 중국 플랫폼 라이브 채팅 연동 |

---

## 2. ai_vtuber 프로젝트 (emeth) 현재 구조

"emeth"라는 이름의 성경/복음 기반 lofi 음악 버튜버 프로젝트. 7단계 파이프라인으로 구성됨.

| 단계 | 디렉토리 | 기능 | 현재 상태 |
|------|----------|------|-----------|
| 0 | `encodec_pipeline/`, `data/` | AI lofi 음악 생성 (EnCodec + DDPM) | 학습 중 |
| 1 | `step1_script/` | Ollama 기반 대본 자동 생성 | 완성 |
| 2 | `step2_tts/` | Edge TTS + RVC 음성 변환 | 완성 |
| 3 | `step3_vtube/` | Live2D 모델 제어 (VTube Studio API) | 완성 |
| 4 | `step4_obs/` | OBS WebSocket 제어 | 완성 |
| 5 | `step5_live/` | YouTube 라이브 방송 자동화 | 완성 |
| 6 | `step6_chat/` | YouTube 라이브 채팅 읽기 + 응답 | 완성 |
| 7 | `step7_cover/` | 커버 곡 파이프라인 | 구현 중 |
| - | `live2d_web/` | 웹 기반 Live2D 뷰어 + REST API | 완성 |
| - | `electron_pet/` | Electron 데스크톱 펫 | 완성 |

### 현재 아키텍처의 특징
- **일방향 파이프라인**: 대본 생성 → TTS → Live2D → OBS → 라이브 → 채팅 응답 (각 단계가 독립적)
- **채팅 응답**: Ollama에 직접 HTTP 요청, 대화 기록 없이 단발성 응답
- **Live2D 제어**: REST API 기반 (live2d_bridge.py), 감정/표정/립싱크 지원
- **TTS**: Edge TTS + 선택적 RVC 변환

---

## 3. 적용 제안: Open-LLM-VTuber에서 가져올 수 있는 것들

### 🔴 우선순위 높음 (즉시 효과가 큰 것)

#### 3-1. 실시간 음성 대화 파이프라인 도입
**현재 문제:** emeth는 텍스트 채팅만 받아서 텍스트로 응답 → TTS 변환하는 구조. 음성 입력 없음.

**제안:** Open-LLM-VTuber처럼 ASR → LLM → TTS → Live2D를 **실시간 스트리밍으로 연결**하는 파이프라인 추가.

**구체적 구현:**
```
[시청자 음성/텍스트] → ASR (sherpa-onnx 또는 Faster-Whisper)
    → Ollama (emeth 모델) 
    → TTS (Edge TTS + RVC)
    → Live2D 립싱크 + 감정 표현
```

- `step6_chat/chat_handler.py`의 `generate_response()`를 확장하여 스트리밍 응답 지원
- 현재 `ollama.chat(stream=False)`를 `stream=True`로 변경하면 TTS를 청크 단위로 생성 가능 → 응답 지연 대폭 감소

#### 3-2. 대화 기록 영속화 (Chat History Persistence)
**현재 문제:** `chat_handler.py`가 매 메시지를 독립적으로 처리. 이전 대화 맥락 없음.

**제안:** Open-LLM-VTuber의 채팅 기록 영속화 개념 적용.

**구체적 구현:**
- `chat_handler.py`의 `generate_response()`에서 Ollama 요청 시 이전 N개 메시지를 `messages` 리스트에 포함
- JSONL 로그 파일(`RESPONSE_LOG_PATH`)을 읽어 세션별 대화 기록 유지
- SQLite나 JSON 파일로 세션별 대화 저장/복원

#### 3-3. 감정 기반 Live2D 표정 자동 제어
**현재 상태:** `live2d_bridge.py`에 `set_emotion()` API가 이미 존재하지만, 채팅 응답 시 감정 분석이 없음.

**제안:** LLM 응답에서 감정을 추출하여 Live2D 표정과 연동.

**구체적 구현:**
- Ollama 시스템 프롬프트에 "응답 앞에 [감정:happy] 형태로 감정 태그를 붙여주세요" 추가
- `chat_handler.py`에서 감정 태그 파싱 → `live2d_bridge.set_emotion()` 호출
- 현재 지원 감정: neutral, calm, happy, joy, sad, fear, angry, surprise, thinking

---

### 🟡 우선순위 중간 (프로젝트 품질 향상)

#### 3-4. 음성 인터럽트 기능
**제안:** 시청자가 말하는 동안 AI 발화를 중단할 수 있는 기능.

**구현 방향:**
- WebSocket 기반으로 현재 REST API를 전환 (또는 병행)
- TTS 재생 중 새 입력이 오면 현재 재생을 취소하고 새 응답 시작
- `live2d_bridge.py`의 `lipsync()` 메서드에 취소 토큰(CancellationToken) 패턴 적용

#### 3-5. 비전 기능 (화면 인식)
**제안:** emeth가 자신의 방송 화면이나 시청자 공유 이미지를 인식할 수 있게 함.

**구현 방향:**
- Ollama의 멀티모달 모델(예: llava, gemma3) 활용
- 주기적 스크린샷 → 이미지를 LLM에 전달 → 상황 인식 기반 발화
- "지금 화면에 뭐가 보여?" 같은 시청자 요청에 응답 가능

#### 3-6. AI 자발적 발화 (Proactive Speaking)
**현재 문제:** emeth는 대본을 미리 생성하거나 채팅에 반응만 함. 자발적 발화 없음.

**제안:** 일정 시간 채팅이 없으면 AI가 먼저 말을 거는 기능.

**구현 방향:**
- `chat_pipeline.py`의 폴링 루프에 타이머 추가
- N초간 채팅 없으면 → "혼잣말" 또는 "시청자에게 질문" 프롬프트 자동 생성
- 현재 재생 중인 lofi 음악의 주제와 연결하면 자연스러운 멘트 가능

#### 3-7. WebSocket 기반 아키텍처 전환
**현재:** REST API (`live2d_bridge.py` → `api/main.py`)

**제안:** Open-LLM-VTuber처럼 FastAPI WebSocket으로 전환하여 양방향 실시간 통신.

**이점:**
- 폴링 없이 즉시 이벤트 전달
- 립싱크 데이터를 실시간 스트리밍 (현재는 전체 프레임을 한번에 전송)
- 다중 클라이언트 동시 연결 (웹 뷰어 + 데스크톱 펫 + OBS 동시 제어)

---

### 🟢 우선순위 낮음 (장기 개선)

#### 3-8. 장기 기억 시스템 (Long-term Memory)
**제안:** Letta 또는 Mem0 같은 장기 기억 프레임워크 도입.

**효과:** 시청자 이름/선호도/이전 대화 내용을 기억하여 개인화된 응답.

#### 3-9. MCP (Model Context Protocol) 지원
**제안:** 날씨, 성경 구절 검색, 음악 정보 등 외부 도구를 LLM에 연결.

**emeth 특화 활용:** 성경 구절 API 연동 → "오늘의 말씀" 자동 인용.

#### 3-10. 데스크톱 펫 모드 강화
**현재:** `electron_pet/`에 기본 구현 있음.

**Open-LLM-VTuber 참고 강화:**
- 투명 배경 + 마우스 클릭 통과 (이미 일부 구현됨)
- 터치/드래그 인터랙션 추가
- 트레이 아이콘으로 모드 전환 (펫 모드 ↔ 윈도우 모드)

---

## 4. 종합 로드맵 제안

```
Phase 1 (즉시): 대화 기록 영속화 + 감정 표정 연동
    → chat_handler.py + live2d_bridge.py 수정만으로 가능
    
Phase 2 (1~2주): 스트리밍 응답 + 음성 인터럽트
    → Ollama 스트리밍 + WebSocket 전환
    
Phase 3 (2~4주): 실시간 음성 대화 + 비전 기능
    → ASR 통합 + 멀티모달 LLM
    
Phase 4 (장기): 장기 기억 + MCP + 자발적 발화
    → 시청자 경험 고도화
```

---

**참고:** Open-LLM-VTuber는 현재 v2.0 재작성을 진행 중이므로, 전체 코드를 가져오기보다는 **아이디어와 아키텍처 패턴**을 참고하여 emeth 프로젝트에 맞게 구현하는 것을 권장합니다.
