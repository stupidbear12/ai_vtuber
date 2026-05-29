# vtuber-auto - emeth 방송 자동화 프로젝트

**캐릭터:** emeth  
**콘텐츠:** 성경과 복음 기반 lofi 음악 소개  
**말투:** 차분하고 따뜻한 존댓말, 시청자를 '여러분'이라 부름

---

## 폴더 구조

```
vtuber-auto/
├── step1_script/       ← AI 대본 생성 (현재 단계)
│   ├── Modelfile           - emeth 캐릭터 정의 (Ollama용)
│   ├── config.py           - 모델/캐릭터/경로 설정
│   ├── generate_script.py  - 대본 자동 생성 스크립트
│   └── output_scripts/     - 생성된 대본 저장 폴더 (자동 생성)
├── step2_tts/          ← TTS 음성 합성 (예정)
├── step3_vtube/        ← VTube Studio 연동 (예정)
├── step4_obs/          ← OBS 자동화 (예정)
└── step5_upload/       ← 유튜브 업로드 (예정)
```

---

## 빠른 시작

### 1. 사전 준비

```bash
# Ollama 설치 확인
ollama --version

# Python 패키지 설치
pip install ollama
```

### 2. emeth 모델 등록

```bash
cd C:\Users\thtgg\mydream\vtuber-auto\step1_script
ollama create emeth -f Modelfile
```

### 3. 대본 생성

```bash
# 대화형으로 주제 입력
python generate_script.py

# 주제 직접 지정
python generate_script.py --topic "시편 23편과 함께하는 새벽 lofi"

# 저장 없이 화면 출력만
python generate_script.py --topic "주님의 평안" --no-save
```

---

## 설정 변경 (config.py)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_NAME` | `emeth` | ollama 커스텀 모델명 |
| `BASE_MODEL` | `gemma3:4b` | 기반 모델 |
| `SCRIPT_DURATION_MIN` | `3` | 최소 대본 분량 (분) |
| `SCRIPT_DURATION_MAX` | `5` | 최대 대본 분량 (분) |

---

## 문제 해결

- **`ollama` 명령 안 됨** → Ollama 설치 후 PC 재시작 필요
- **모델 생성 오류** → `ollama pull gemma3:4b` 먼저 실행
- **응답이 느림** → GPU 드라이버 확인, 또는 `llama3.2:3b`로 모델 변경
