FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY info ./info
COPY scripts ./scripts
COPY run.py README.md .env.example langgraph.json pytest.ini ./

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "scripts/container_boot.py"]
