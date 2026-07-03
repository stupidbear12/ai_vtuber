# ai_voice — Chatterbox TTS 서버 Dockerfile
# GPU(CUDA) 환경에서 실행을 권장합니다 (RTX 4060 Ti 기준)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
# torch/torchaudio는 베이스 이미지에 포함 (CUDA 버전)
RUN pip install --no-cache-dir chatterbox-tts soundfile fastapi uvicorn[standard] python-dotenv

COPY app/ app/

# 레퍼런스 오디오 볼륨 마운트 포인트
VOLUME ["/app/data/reference_audio"]

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
