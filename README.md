# lofi-diffusion — AI Lofi 음악 생성 모델

> emeth 버튜버를 위한 자체 AI lofi 음악 생성 모델  
> **EnCodec + Latent Diffusion** 기반, PyTorch / RTX 4060 Ti

---

## 프로젝트 개요

FMA small 데이터셋(8,000곡)을 기반으로, Meta EnCodec의 오디오 표현을 잠재 공간으로 활용하여 DDPM으로 lofi 음악을 생성하는 모델입니다.

---

## 디렉토리 구조

```
lofi-diffusion/
├── data/
│   ├── download.py           # FMA small CC 트랙 수집
│   ├── preprocess.py         # 오디오 → 멜 스펙트로그램 변환
│   └── dataset.py            # 데이터로더
├── encodec_pipeline/         # 메인: EnCodec + Latent Diffusion
│   ├── preprocess_encodec.py # 오디오 → EnCodec 잠재 벡터 추출
│   ├── unet_1d.py            # 1D U-Net (잠재 공간용)
│   ├── ddpm_encodec.py       # DDPM 학습 루프
│   ├── train_encodec.py      # 학습 진입점
│   └── generate_encodec.py   # 음악 생성
├── models/
│   └── unet.py               # mel-spectrogram U-Net (구버전)
├── diffusion/
│   └── ddpm.py               # DDPM 스케줄러
├── encodec_latents/          # 추출된 EnCodec 잠재 벡터 캐시
├── checkpoints/              # 학습 체크포인트
├── train.py                  # 구버전 학습 스크립트
├── generate.py               # 구버전 생성 스크립트
└── requirements.txt
```

---

## 모델 아키텍처

### 1단계 — EnCodec 오디오 표현

Meta [EnCodec](https://github.com/facebookresearch/encodec) (24kHz)을 오디오 인코더로 사용합니다. EnCodec은 오디오 파형을 **잔차 벡터 양자화(RVQ)** 코드북으로 압축하며, 이 잠재 표현을 디퓨전 모델의 입력으로 활용합니다.

```
오디오 파형 (24kHz)
    ↓  EnCodec Encoder
잠재 벡터 (128-dim, ~75 fps)
    ↓  DDPM (1D U-Net)
복원된 잠재 벡터
    ↓  EnCodec Decoder
생성된 오디오 파형
```

EnCodec을 잠재 공간으로 사용함으로써 원시 파형 대신 압축된 의미 표현 위에서 디퓨전을 수행, 학습 효율과 생성 품질을 동시에 확보합니다.

### 2단계 — Latent Diffusion (DDPM)

**DDPM** (Denoising Diffusion Probabilistic Models, Ho et al. 2020)을 기반으로, EnCodec 잠재 공간에서 가우시안 노이즈를 제거하는 방식으로 음악을 생성합니다.

**노이즈 스케줄:** Nichol & Dhariwal (2021)의 **cosine schedule**을 채택합니다. 선형 스케줄(Ho et al.)에 비해 중간 타임스텝에서 신호 대 노이즈 비율이 완만하게 감소하여, 음악처럼 장기 의존성이 중요한 데이터에 더 유리합니다.

```
코사인 스케줄:
α̅_t = cos²( (t/T + s) / (1 + s) · π/2 )

여기서 s = 0.008 (오프셋, 극단적 노이즈 방지)
```

**역방향 과정 (denoising):**
```
x_{t-1} = 1/√α_t · (x_t - (1-α_t)/√(1-α̅_t) · ε_θ(x_t, t)) + σ_t · z
```

### 3단계 — 1D U-Net 노이즈 예측기

잠재 벡터가 시퀀스(1D) 형태이므로, 이미지용 2D U-Net 대신 **1D U-Net** (`encodec_pipeline/unet_1d.py`)을 사용합니다. 타임스텝 임베딩을 각 ResBlock에 주입하여 현재 노이즈 레벨 정보를 전달합니다.

---

## 데이터셋

**FMA small** ([Defferrard et al. 2017](https://arxiv.org/abs/1612.01840))

- 8,000곡 / CC 라이선스 / 30초 클립
- 전처리 결과: **151,886개 mel 청크** (mel-spectrogram 파이프라인)
- EnCodec 파이프라인: 원본 오디오 → EnCodec 잠재 벡터로 재전처리

---

## 설치

```bash
# PyTorch (CUDA 12, RTX 4060 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# EnCodec
pip install encodec

# 나머지 의존성
pip install -r requirements.txt
```

---

## 실행

```bash
# EnCodec 잠재 벡터 추출
python encodec_pipeline/preprocess_encodec.py

# 모델 학습
python encodec_pipeline/train_encodec.py

# 음악 생성
python encodec_pipeline/generate_encodec.py
```

배치 실행은 `run_encodec_preprocess.bat`, `run_encodec_train.bat`, `run_generate.bat` 사용.

---

## 개발 현황

| 단계 | 상태 |
|---|---|
| FMA small 데이터 수집 | ✅ 완성 |
| 멜 스펙트로그램 전처리 (151,886 청크) | ✅ 완성 |
| EnCodec 잠재 벡터 추출 | ✅ 완성 |
| 1D U-Net + DDPM 학습 | 🔧 진행 중 |
| 음악 생성 및 품질 평가 | ⏳ 예정 |
| emeth 방송 실시간 연동 | ⏳ 예정 |

---

## 참고 논문

- Ho, J. et al. (2020). **Denoising Diffusion Probabilistic Models.** NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol, A. & Dhariwal, P. (2021). **Improved Denoising Diffusion Probabilistic Models.** ICML. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Défossez, A. et al. (2022). **High Fidelity Neural Audio Compression (EnCodec).** [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
- Rombach, R. et al. (2022). **High-Resolution Image Synthesis with Latent Diffusion Models.** CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Defferrard, M. et al. (2017). **FMA: A Dataset For Music Analysis.** ISMIR. [arXiv:1612.01840](https://arxiv.org/abs/1612.01840)

---

## 라이선스

MIT License  
학습 데이터: FMA small (CC BY 4.0)
