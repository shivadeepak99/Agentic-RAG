FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Many transitive ML deps (via sentence-transformers) can pull CUDA-enabled PyTorch
# wheels on Linux, which can balloon images by multiple GB. We install CPU-only
# PyTorch explicitly first, then install the rest of the requirements.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch \
    && pip install --no-cache-dir --extra-index-url ${TORCH_INDEX_URL} -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY scripts ./scripts
COPY run.py README.md .env.example langgraph.json pytest.ini ./

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "scripts/container_boot.py"]
