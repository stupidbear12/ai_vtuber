# ai_voice — GPT-SoVITS API 클라이언트 Dockerfile
# 실제 추론(GPU)은 별도로 실행되는 GPT-SoVITS api_v2.py 서버가 담당하므로
# 이 컨테이너는 경량 Python 이미지로 충분하다.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
