FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml requirements.txt README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs

RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install --upgrade pip \
    && uv pip install -e . \
    && uv pip install -r requirements.txt

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

CMD ["python", "scripts/smoke_test.py", "--config", "configs/baselines/mnist_baseline.yaml"]
