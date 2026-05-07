FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./app.py
COPY .env.example ./.env.example

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir --retries 10 --timeout 120 .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
