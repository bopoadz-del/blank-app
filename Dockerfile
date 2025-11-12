# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /build/frontend
ENV CI=true
# Copy package manifests first for cache-friendly builds
COPY frontend/package*.json ./ 
# If you use lockfile(s), copy them too
COPY frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ .
# Ensure production build env
ENV NODE_ENV=production
RUN npm run build

# Stage 2: Build backend and install Python deps into a virtualenv
FROM python:3.11-slim AS backend-builder
WORKDIR /build/backend
# Install system deps needed for building wheels if any
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates gcc \
    && rm -rf /var/lib/apt/lists/*
# Create a venv to make runtime copy predictable
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

# Copy backend sources
COPY backend/ ./backend
WORKDIR /build/backend/backend
# Install python deps if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt; fi

# Stage 3: Final runtime image
FROM python:3.11-slim
# Minimal packages
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
ARG APP_USER=appuser
RUN groupadd -r $APP_USER && useradd -r -g $APP_USER $APP_USER

# Copy venv from builder
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend code
WORKDIR /app
COPY --from=backend-builder /build/backend/backend ./backend

# Copy frontend build artifacts into backend so FastAPI can serve them
COPY --from=frontend-builder /build/frontend/dist ./backend/frontend/dist

WORKDIR /app/backend
ENV PYTHONPATH=/app/backend
ENV PORT=8000

# Use non-root for better security; ensure folder permissions
RUN chown -R $APP_USER:$APP_USER /app
USER $APP_USER

# Render sets $PORT at runtime; fallback to 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
