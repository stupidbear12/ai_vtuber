# lofi-diffusion

**Lofi 음악 생성을 위한 스펙트로그램 디퓨전 모델 (JAX/Flax)**

> 현재 단계: **1단계 — 데이터 파이프라인**

---

## 프로젝트 구조

```
lofi-diffusion/
├── data/
│   ├── download.py      # CC 라이선스 트랙 자동 수집
│   ├── preprocess.py    # 오디오 → 멜 스펙트로그램 변환
│   ├── dataset.py       # JAX 데이터로더 + augmentation
│   ├── visualize.py     # 스펙트로그램 시각화
│   ├── raw/             # 원본 오디오 파일 저장 위치
│   ├── processed/       # 전처리된 .npy 파일 저장 위치
│   └── visualizations/  # 시각화 출력 저장 위치
├── requirements.txt
└── README.md
```

---

## 환경 설정

### 1. Python 3.10+ 가상환경 생성

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. JAX 설치 (GPU 권장)

**CUDA 12 (GPU):**
```powershell
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**CPU 전용:**
```powershell
pip install -U "jax[cpu]"
```

### 3. 나머지 패키지 설치

```powershell
pip install -r requirements.txt
```

### 4. ffmpeg 설치 (yt-dlp 오디오 추출 필수)

```powershell
# winget 사용
winget install ffmpeg

# 또는 https://ffmpeg.org/download.html 에서 수동 설치 후 PATH 추가
```

---

## 데이터 파이프라인 실행 순서

### Step 1 — 오디오 데이터 수집

```powershell
# 방법 A: FMA small 전체 데이터셋 다운로드 (~7.2GB, 8000곡)
python data/download.py --source fma_small --output data/raw

# 방법 B: yt-dlp로 YouTube CC 트랙 추출 (소량 테스트용)
python data/download.py --source ytdlp --output data/raw --limit 20

# 방법 C: FMA API (API 키 필요)
python data/download.py --source fma --fma_api_key YOUR_KEY --genre hip-hop --limit 200
```

> **저작권 안내**: 이 스크립트는 CC BY / CC0 라이선스 트랙만 수집합니다.
> FMA 데이터셋 라이선스: https://github.com/mdeff/fma

### Step 2 — 전처리 (오디오 → 멜 스펙트로그램)

```powershell
# 단일 프로세스 (기본)
python data/preprocess.py --input data/raw --output data/processed

# 멀티프로세싱 (CPU 코어 4개 활용)
python data/preprocess.py --input data/raw --output data/processed --workers 4

# 옵션 상세
python data/preprocess.py --help
```

**출력 파일 형식:**
- 경로: `data/processed/<원본파일명>_chunk<번호>.npy`
- Shape: `(128, 256)` — (mel bins, time frames)
- dtype: `float32`, 값 범위: `[-1, 1]`

### Step 3 — 결과 검증 (시각화)

```powershell
# 오디오 파일: 파형 + 멜 스펙트로그램
python data/visualize.py --audio data/raw/my_track.wav

# 전처리된 .npy 파일 확인
python data/visualize.py --npy data/processed/my_track_chunk0000.npy

# 여러 청크 그리드 표시
python data/visualize.py --grid data/processed --n 16

# Augmentation 비교 확인
python data/visualize.py --augment data/raw/my_track.wav

# 이미지 파일로 저장
python data/visualize.py --audio data/raw/my_track.wav --save data/visualizations/preview.png --no_show
```

### Step 4 — 데이터로더 테스트

```powershell
python data/dataset.py data/processed
```

예상 출력:
```
총 샘플 수: 3420
샘플 shape: (1, 128, 256)
값 범위: [-0.997, 0.998]
배치 shape: (4, 1, 128, 256)
배치 dtype: float32
데이터로더 테스트 완료!
```

---

## 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `SAMPLE_RATE` | 22050 Hz | 오디오 샘플레이트 |
| `N_MELS` | 128 | 멜 필터뱅크 수 |
| `N_FFT` | 1024 | FFT 윈도우 크기 |
| `HOP_LENGTH` | 256 | 프레임 이동 샘플 수 |
| `CHUNK_FRAMES` | 256 | 청크당 프레임 수 (~2.97초) |
| 출력 Shape | (128, 256) | (mel bins, frames) |
| 정규화 범위 | [-1, 1] | 디퓨전 모델 표준 입력 범위 |

---

## 데이터 Augmentation

`dataset.py` 에 구현된 on-the-fly augmentation:

| 방법 | 범위 | 설명 |
|---|---|---|
| Pitch shift | ±2 semitone | 멜 빈 축 이동으로 근사 |
| Time stretch | 0.9 ~ 1.1× | 선형 보간 후 크롭/패딩 |
| Gaussian noise | σ=0.005 | 정규화 공간에서 노이즈 추가 |

각 augmentation은 독립적으로 `p=0.5` 확률로 적용됩니다.

---

## 다음 단계

- [ ] **2단계**: U-Net 기반 노이즈 예측 네트워크 (JAX/Flax)
- [ ] **3단계**: DDPM 훈련 루프 (cosine noise schedule)
- [ ] **4단계**: 샘플링 및 오디오 복원 (Griffin-Lim / Vocoder)

---

## 참고 자료

- [FMA Dataset](https://github.com/mdeff/fma) — Free Music Archive
- [JAX 공식 문서](https://jax.readthedocs.io)
- [Flax NNX 가이드](https://flax.readthedocs.io)
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Diff-Wave: Diffusion for Waveforms](https://arxiv.org/abs/2009.09761)
