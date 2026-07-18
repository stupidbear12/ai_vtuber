# AI VTuber DJ 프로젝트 — 해외 특허 분석 보고서

**작성일:** 2026-06-17  
**대상 프로젝트:** 실시간 AI 음악 생성 + 스트리밍 + 가상 DJ 캐릭터 통합 시스템  
**검색 범위:** USPTO (Google Patents), EPO (Espacenet), WIPO (PATENTSCOPE)

---

## 1. 검색 결과 요약

10가지 키워드 조합으로 3개 특허 데이터베이스를 검색한 결과, 프로젝트의 개별 기술 요소에 해당하는 선행 특허는 다수 존재하나, **"시청자 채팅 → AI 음악 생성 → FFT 기반 Live2D 모션 동기화"라는 엔드투엔드 통합 파이프라인을 청구하는 단일 특허는 발견되지 않았다.**

---

## 2. 관련 선행 특허 목록

### 2.1 실시간 AI 음악 생성 영역

| 특허번호 | 발명 명칭 | 출원인/권리자 | 출원일 | 핵심 내용 |
|----------|-----------|---------------|--------|-----------|
| US10467998B2 | Automated Music Composition and Generation System (emotion/style/accent descriptors) | Shutterstock (구 Amper Music) | 2017 | emotion-type, style-type, accent-type 음악 경험 디스크립터로 자동 작곡. 실시간 연주 증강 가능. 사용자 상호작용 기반 학습·진화. |
| US10790919B1 | Personalized Real-Time Audio Generation Based on User Physiological Response | — | ~2019 | 청력 테스트 + 스트리밍 플랫폼 청취 이력 → ML 모델로 개인화 음악 실시간 생성. |
| US12322402B2 | AI-Generated Music Derivative Works | Music IP Holdings (UMG 연계) | 2024-10-24 | 생성 AI로 파생 음악 제작 → ML 기반 승인 판별 → 디지털 워터마크 부착. 스트리밍 사용 실시간 추적. |
| US12266331B2 | Methods for Facilitating Interactive Creation of Live Music by Multiple Users | Mark L. Palmer (개인) | 2021-01-03 | 복수 사용자 기기에서 음악 세그먼트 선택 → 실시간 공통 비트 동기화 → 협업 음악 생성. |
| US7498504B2 | Cellular Automata Music Generator | — | ~2005 | 셀룰러 오토마타 수학 모델 기반 실시간 음악 생성·변형 소프트웨어 플랫폼. |

### 2.2 라이브 스트리밍 + AI 시청자 상호작용 영역

| 특허번호 | 발명 명칭 | 출원인/권리자 | 출원일 | 핵심 내용 |
|----------|-----------|---------------|--------|-----------|
| US11412271B2 | AI Response to Viewers of Live Stream Video | — | ~2020 | 라이브 스트림 중 소셜미디어 채팅 메시지 캡처 → NLP로 토론 패턴 분석 → AI 응답 자동 생성·전송. |
| US20250047927A1 | Live Playback Streams | Block, Inc. | 2024-10-23 | 플레이리스트 곡 전환 시 믹스 룰(템포/피치/음량/페이드) 적용. 봇이 ML 모델로 아티스트-관객 커뮤니케이션 자동화. |
| US20240379107A1 | Real-Time AI Screening and Auto-Moderation of Audio Comments in a Livestream | — | ~2024 | 라이브스트림 채팅에 음성 코멘트 추가 기능 + AI 실시간 스크리닝·자동 중재. |
| US20160286244A1 | Live Video Streaming Services | — | ~2016 | 라이브 비디오 스트리밍 + 인터랙티브 채팅 관리 메커니즘. |

### 2.3 가상 캐릭터 + 감정 분석 + 애니메이션 영역

| 특허번호 | 발명 명칭 | 출원인/권리자 | 출원일 | 핵심 내용 |
|----------|-----------|---------------|--------|-----------|
| US12293444B2 | Music Reactive Animation of Human Characters | Snap Inc. | 2023-10-06 | 음악 입력 신호 → 인코딩/디코딩 신경망(조건부 RNN) → 잠재 공간에서 댄스 포즈 생성. **프로젝트와 가장 직접적으로 관련.** |
| US11816773B2 | Music Reactive Animation of Human Characters (선행) | Snap Inc. | 2021-09-28 | 위 특허의 선행 출원. 음악 반응형 캐릭터 애니메이션 기본 구조. |
| WO2021158692A1 | Using Text for Avatar Animation | — (Microsoft 추정) | 2021 | 텍스트 수신 → 감정 상태 판별 → 신경망으로 음성 데이터 + 아바타 움직임 파라미터 생성. |
| US20120130717A1 | Real-Time Animation for an Expressive Avatar | — | ~2012 | 음성 + 모션 데이터로 애니메이션 모델 훈련 → 실시간 음성 입력에서 감정 상태 식별 → 상체 모션 시퀀스 생성. |
| WO2008049834A2 | Virtual Assistant with Real-Time Emotions | — | 2008 | EXML(Emotional XML) 파일로 아바타 감정 관리 → 3D 비디오 렌더링 엔진 구동. |
| US12249014B1 | Integrating Applications with Dynamic Virtual Assistant Avatars | Meta Platforms | ~2024 | XR 환경에서 가상 어시스턴트 아바타를 앱별 렌더링 사양에 맞게 동적 변환. |
| WO2020152657A1 | Real-Time Generation of Speech Animation | — | 2020 | 음성 애니메이션에 감정 표현 오버레이 → 가상 캐릭터 감정적 음성 변형. |
| US20140361974A1 | Karaoke Avatar Animation Based on Facial Motion Data | — | ~2014 | 노래방 환경에서 음성 인식 + 얼굴 모션 데이터 기반 아바타 애니메이션. |

### 2.4 TTS + 립싱크 영역

| 특허번호 | 발명 명칭 | 출원인/권리자 | 출원일 | 핵심 내용 |
|----------|-----------|---------------|--------|-----------|
| RE42647 | Text-to-Speech Conversion System for Synchronizing Synthesized Speech and Moving Picture | — | 재발행 | TTS 출력의 음소 스트림 + 지속시간 정보 → 립 애니메이션 동기화. |
| US20160078859A1 | Text-to-Speech with Emotional Content | — | ~2016 | 텍스트 + 감정 콘텐츠 결합 TTS 음성 합성. |

### 2.5 FFT/음악 시각화 영역

| 특허번호 | 발명 명칭 | 출원인/권리자 | 출원일 | 핵심 내용 |
|----------|-----------|---------------|--------|-----------|
| US20100205532A1 | Customizable Music Visualizer | — | ~2010 | FFT로 수신 음악 분석 → 스윕 암의 호 크기·속도 제어. |
| US7132595B2 | Beat Analysis of Musical Signals | — | ~2003 | FFT → 옥타브 기반 주파수 서브밴드 분할 → 비트 분석. |
| US7875787B2 | Visualization of Music Using Note Extraction | — | ~2006 | FFT로 시간→주파수 도메인 변환 → 주파수 대역별 파워 시각화. |

---

## 3. 프로젝트 기술 요소별 선행 특허 중첩 분석

### 3.1 시청자 채팅 실시간 분석 → 장르/무드/BPM 추출

**관련 선행 특허:**
- US11412271B2: 라이브 채팅 NLP 분석 → AI 응답 생성 (가장 유사)
- US20240379107A1: 라이브스트림 채팅 AI 스크리닝

**중첩 정도:** 중간. 채팅 메시지의 NLP 분석까지는 선행 기술이 존재하지만, **채팅에서 음악적 파라미터(장르/무드/BPM)를 추출하는 특정 방법론**은 선행 특허에서 발견되지 않음.

**차별화 포인트:** 채팅 텍스트 → "음악 프롬프트"로의 변환 알고리즘 (감정 분석 + 음악 도메인 매핑)이 신규성의 핵심.

### 3.2 AI 음악 생성 모델에 실시간 프롬프트 전달 → 음원 스트리밍 출력

**관련 선행 특허:**
- US10467998B2: emotion/style descriptor 기반 자동 작곡 (가장 유사)
- US12322402B2: AI 생성 음악 파생물 + 스트리밍 추적
- US10790919B1: 사용자 반응 기반 실시간 오디오 생성

**중첩 정도:** 높음. 감정/스타일 디스크립터 기반 AI 음악 생성은 US10467998B2가 광범위하게 커버. 단, **시청자 채팅에서 실시간으로 추출한 프롬프트를 음악 생성 모델에 연속 전달하는 파이프라인**은 미발견.

**차별화 포인트:** "채팅 텍스트 → 프롬프트 → 스트리밍 음악"의 연속적 실시간 루프 구조.

### 3.3 생성된 음악의 FFT 분석 → Live2D DJ 캐릭터 모션 자동 동기화

**관련 선행 특허:**
- **US12293444B2 (Snap):** 가장 직접적 위협. 음악 → 신경망 → 캐릭터 댄스 포즈 생성. 단, 3D 캐릭터 대상이며 Live2D 특화 아님.
- US20100205532A1: FFT 기반 음악 시각화
- US7132595B2: FFT 비트 분석

**중첩 정도:** 중간~높음. 음악 반응형 캐릭터 애니메이션 자체는 Snap 특허가 커버하나, **Live2D 2D 캐릭터에 특화된 FFT → 모션 매핑**은 별도 영역.

**차별화 포인트:**
1. Live2D 파라미터(ParamAngleX, ParamBodyAngleZ 등)에 직접 매핑하는 FFT 주파수 대역별 바인딩
2. 2D 모델 특유의 물리 시뮬레이션 연동
3. AI 생성 음악(플레이백이 아닌)에 대한 실시간 분석이라는 점

### 3.4 시청자 피드백 루프 — 반응에 따른 다음 곡 스타일 자동 조정

**관련 선행 특허:**
- US10467998B2: 사용자 상호작용 기반 시스템 진화·적응 (개념적 유사)
- US10790919B1: 사용자 생리 반응 기반 개인화

**중첩 정도:** 낮음. 기존 특허는 개인 사용자 반응이나 오프라인 학습 기반이며, **다수 시청자의 실시간 집합적 피드백을 기반으로 다음 생성 음악의 스타일을 동적 조정**하는 메커니즘은 미발견.

**차별화 포인트:** 군중 감정의 실시간 집계 → 다음 곡 파라미터 자동 조정 폐루프(closed-loop) 시스템.

### 3.5 Web Audio API 기반 브라우저 네이티브 립싱크

**관련 선행 특허:**
- RE42647: TTS + 립 애니메이션 동기화 (범용)
- US20160078859A1: 감정 TTS

**중첩 정도:** 낮음. TTS 립싱크 자체는 오래된 기술이나, **Web Audio API + AudioWorklet을 사용한 브라우저 네이티브 구현**은 특허보다는 오픈소스 구현(lipsync-engine, wawa-lipsync 등)이 주류. 특허 장벽 낮음.

**차별화 포인트:** 별도 소프트웨어 없이 브라우저만으로 완결되는 립싱크 파이프라인.

### 3.6 엔드투엔드 통합 파이프라인 (채팅→AI→TTS→Live2D)

**관련 선행 특허:** 직접적으로 대응하는 단일 선행 특허 없음.

**중첩 정도:** 낮음. 이것이 가장 강력한 신규성 주장 가능 영역.

**차별화 포인트:** 채팅 분석 → AI 음악 생성 → FFT 분석 → Live2D 모션 동기화 → TTS → 시청자 피드백 루프를 하나의 통합 시스템으로 구현하는 아키텍처 자체가 신규.

---

## 4. 위험도 평가 매트릭스

| 기술 요소 | 선행 특허 위험도 | 신규성 가능성 | 주요 주의 특허 |
|-----------|:---:|:---:|-------------|
| 채팅 NLP 분석 | ★★☆☆☆ | 높음 | US11412271B2 |
| AI 음악 생성 (디스크립터 기반) | ★★★★☆ | 중간 | US10467998B2 |
| 음악 반응형 캐릭터 모션 | ★★★☆☆ | 높음 | US12293444B2 (Snap) |
| FFT 음악 시각화 | ★★☆☆☆ | 높음 | US20100205532A1 |
| 시청자 피드백 루프 | ★☆☆☆☆ | 매우 높음 | — |
| TTS + 립싱크 | ★★☆☆☆ | 높음 | RE42647 |
| 엔드투엔드 통합 파이프라인 | ★☆☆☆☆ | 매우 높음 | — |

---

## 5. 해외 출원 전략 제안

### 5.1 출원 우선 타겟 청구항

신규성이 가장 강한 영역을 중심으로 청구항을 구성할 것을 권장한다:

1. **통합 시스템 청구항 (가장 강력):** "복수 시청자의 실시간 채팅 메시지를 수신하여 음악적 파라미터를 추출하고, 해당 파라미터를 AI 음악 생성 모델에 프롬프트로 전달하여 음원을 스트리밍 출력하며, 출력된 음원의 주파수 분석 결과를 2D 가상 캐릭터의 모션 파라미터에 실시간 매핑하는 시스템"

2. **피드백 루프 청구항:** "복수 시청자의 집합적 반응 데이터를 실시간 집계하여 다음 생성 음악의 장르, 무드, BPM 파라미터를 자동 조정하는 폐루프 시스템"

3. **FFT-to-Live2D 매핑 청구항:** "생성된 음악의 FFT 분석 결과에서 추출한 주파수 대역별 에너지를 2D 캐릭터 모델의 특정 모션 파라미터에 바인딩하여 실시간 애니메이션을 생성하는 방법"

### 5.2 출원 관할권 전략

| 관할권 | 우선순위 | 근거 |
|--------|:---:|------|
| **미국 (USPTO)** | 1순위 | AI/스트리밍 기술 특허의 최대 시장. 선행 특허 대부분 미국 출원. 소프트웨어 특허 인정 범위 넓음. |
| **PCT (WIPO)** | 1순위 | 미국 출원과 동시에 PCT 출원하여 30개월 내 개별국 진입 유연성 확보. |
| **일본 (JPO)** | 2순위 | VTuber 시장의 본거지. Live2D 기술의 원산지. 관련 사업화 가능성 높음. |
| **EU (EPO)** | 3순위 | 기술적 효과가 명확한 청구항이면 소프트웨어 특허 등록 가능. GDPR 관련 데이터 처리 측면 고려. |
| **한국 (KIPO)** | 3순위 | 국내 시장 보호. AI 소프트웨어 특허 등록 기준 상대적으로 유연. |

### 5.3 회피 설계 권고사항

아래 선행 특허의 청구항 범위를 회피해야 한다:

- **US10467998B2 (Shutterstock):** emotion/style descriptor 기반 자동 작곡 시스템. 회피 방법 → 채팅 텍스트에서 직접 프롬프트를 생성하는 방식으로, descriptor 체계와 다른 입력 메커니즘 사용.
- **US12293444B2 (Snap):** 음악 → RNN 기반 댄스 포즈 생성. 회피 방법 → (1) Live2D 2D 파라미터 직접 매핑 방식 사용, (2) FFT 기반 규칙 매핑(신경망 미사용) 또는 다른 아키텍처의 신경망 사용.
- **US12322402B2 (UMG):** AI 음악 파생물의 승인·워터마크. 회피 방법 → 원본 음악에서 파생하지 않고 완전 신규 생성하는 방식 유지.

### 5.4 출원 타임라인

1. **즉시:** 한국 가출원(임시 명세서) 제출하여 우선일 확보
2. **12개월 이내:** 파리조약 우선권 주장하여 미국 정규출원 + PCT 출원
3. **PCT 출원 후 30개월 이내:** 일본, EU 개별국 진입 판단

---

## 6. 결론

프로젝트의 **개별 기술 요소(채팅 NLP, AI 음악 생성, 캐릭터 애니메이션, FFT 분석, TTS 립싱크)는 각각 선행 기술이 존재**하나, 이들을 **하나의 실시간 파이프라인으로 통합하여 시청자 참여형 AI DJ 캐릭터를 구현하는 시스템**으로서의 신규성은 충분히 인정받을 수 있다.

특히 아래 3가지가 핵심 차별화 포인트이다:
1. 다수 시청자 채팅 → 음악 파라미터 실시간 변환 매핑
2. AI 생성 음악의 FFT → Live2D 파라미터 직접 바인딩
3. 군중 피드백 기반 폐루프 음악 스타일 자동 조정

가장 주의해야 할 선행 특허는 **Snap Inc.의 US12293444B2** (음악 반응형 캐릭터 애니메이션)와 **Shutterstock의 US10467998B2** (감정/스타일 기반 자동 작곡)이다.

---

## Sources

- [US10467998B2 - Automated Music Composition](https://patents.google.com/patent/US10467998B2/en)
- [US10790919B1 - Personalized Real-Time Audio Generation](https://patents.google.com/patent/US10790919B1)
- [US12322402B2 - AI-Generated Music Derivative Works](https://patents.google.com/patent/US12322402B2/en)
- [US11412271B2 - AI Response to Viewers of Live Stream](https://patents.google.com/patent/US11412271B2/en)
- [US12293444B2 - Music Reactive Animation (Snap)](https://patents.google.com/patent/US12293444)
- [US12266331B2 - Interactive Creation of Live Music](https://patents.google.com/patent/US12266331B2/en)
- [US20250047927A1 - Live Playback Streams](https://patents.google.com/patent/US20250047927A1/en)
- [US20240379107A1 - AI Screening of Livestream Comments](https://patents.google.com/patent/US20240379107A1/)
- [WO2021158692A1 - Using Text for Avatar Animation](https://patents.google.com/patent/WO2021158692A1/en)
- [US20120130717A1 - Expressive Avatar Animation](https://patents.google.com/patent/US20120130717)
- [US12249014B1 - Dynamic Virtual Assistant Avatars (Meta)](https://patents.google.com/patent/US12249014B1/en)
- [WO2020152657A1 - Real-Time Speech Animation](https://patents.google.com/patent/WO2020152657A1/en)
- [US20100205532A1 - Customizable Music Visualizer](https://patents.google.com/patent/US20100205532)
- [US7132595B2 - Beat Analysis of Musical Signals](https://patents.google.com/patent/US7132595B2/en)
- [US11816773 - Music Reactive Animation (Snap, 선행)](https://patents.justia.com/patent/11816773)
- [UMG AI Patent Strategy Analysis](https://www.musicbusinessworldwide.com/inside-the-umg-backed-patent-portfolio-targeting-ai-music-derivatives-a-technical-blueprint-for-the-walled-garden-model/)
