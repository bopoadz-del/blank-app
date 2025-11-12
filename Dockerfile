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
WORKDIR /build/backend
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates gcc \
    && rm -rf /var/lib/apt/lists/*
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

COPY backend/ ./backend
WORKDIR /build/backend/backend
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Stage 3: Final image
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG APP_USER=appuser
RUN groupadd -r $APP_USER && useradd -r -g $APP_USER $APP_USER
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG APP_USER=appuser
RUN groupadd -r $APP_USER && useradd -r -g $APP_USER $APP_USER

COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --from=backend-builder /build/backend/backend ./backend
COPY --from=frontend-builder /build/frontend/dist ./backend/frontend/dist
WORKDIR /app/backend
ENV PYTHONPATH=/app/backend
ENV PORT=8000
RUN chown -R $APP_USER:$APP_USER /app
USER $APP_USER

RUN chown -R $APP_USER:$APP_USER /app
USER $APP_USER

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
