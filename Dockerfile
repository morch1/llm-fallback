FROM ghcr.io/astral-sh/uv:alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1 \
    UV_NO_CACHE=1

COPY . /app
WORKDIR /app

RUN uv sync --locked

EXPOSE 8000

CMD ["uv", "run", "llm-fallback", "--config", "/etc/llm-fallback.yaml"]
