# Multi-stage Dockerfile for unified UI+API (Render-compatible)
# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /build/frontend
ENV CI=true
COPY frontend/package*.json ./
COPY frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ .
ENV NODE_ENV=production
RUN npm run build

# Stage 2: Build backend, install Python deps into venv
FROM python:3.11-slim AS backend-builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates gcc \
    && rm -rf /var/lib/apt/lists/*
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final image
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ARG APP_USER=appuser
RUN groupadd -r $APP_USER && useradd -r -g $APP_USER $APP_USER

COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY backend/app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app
ENV PORT=8000

RUN chown -R $APP_USER:$APP_USER /app
USER $APP_USER

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
