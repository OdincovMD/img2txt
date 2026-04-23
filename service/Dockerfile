FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY service/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY service ./service

EXPOSE 8000

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
