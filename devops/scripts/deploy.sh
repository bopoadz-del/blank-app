#!/bin/bash
#
# ==============================================================================
# The Reasoner Platform - One-Command Deployment Script
# ==============================================================================
# This script deploys The Reasoner Platform on a VPS with Docker
#
# Usage: ./deploy.sh [environment]
# Example: ./deploy.sh production
#

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Error handler
trap 'echo ""; echo "❌ Deployment failed at line $LINENO"; echo "Check logs above for details"; exit 1' ERR

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Go up two levels: scripts -> devops -> root

echo "=========================================="
echo "  The Reasoner Platform Deployment"
echo "=========================================="
echo "Environment: $ENVIRONMENT"
echo "Script Dir: $SCRIPT_DIR"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Validate project structure
if [ ! -f "$PROJECT_ROOT/docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found in $PROJECT_ROOT"
    echo "   Make sure you're running from the correct location"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/backend" ]; then
    echo "❌ Error: backend directory not found in $PROJECT_ROOT"
    exit 1
fi

echo "✅ Project structure validated"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Install from https://docs.docker.com/get-docker/"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Install from https://docs.docker.com/compose/install/"; exit 1; }
echo "✅ Prerequisites OK"
echo ""

# Check .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and set:"
    echo "   - SECRET_KEY (generate with: openssl rand -hex 32)"
    echo "   - API_KEY (generate with: openssl rand -hex 32)"
    echo "   - Database passwords"
    echo "   - Redis password"
    echo "   - CORS_ORIGINS (your domain)"
    echo ""
    read -p "Press Enter after updating .env file..."
fi

# Load environment variables
set -a
source "$PROJECT_ROOT/.env"
set +a

echo "🔑 Generating secrets if needed..."
if [ "$SECRET_KEY" == "your-secret-key-here-change-this-in-production" ]; then
    NEW_SECRET=$(openssl rand -hex 32)
    sed -i "s/SECRET_KEY=.*/SECRET_KEY=$NEW_SECRET/" "$PROJECT_ROOT/.env"
    echo "✅ Generated new SECRET_KEY"
fi

if [ "$API_KEY" == "your-api-key-here-change-this-in-production" ]; then
    NEW_API_KEY=$(openssl rand -hex 32)
    sed -i "s/API_KEY=.*/API_KEY=$NEW_API_KEY/" "$PROJECT_ROOT/.env"
    echo "✅ Generated new API_KEY: $NEW_API_KEY"
    echo "⚠️  SAVE THIS API KEY - you'll need it to access the API"
fi
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$PROJECT_ROOT/data/formulas"
mkdir -p "$PROJECT_ROOT/data/datasets"
mkdir -p "$PROJECT_ROOT/data/backups"
mkdir -p "$PROJECT_ROOT/logs"
echo "✅ Directories created"
echo ""

# Stop existing containers
echo "🛑 Stopping existing containers..."
cd "$PROJECT_ROOT"
docker-compose down || true
echo ""

# Build images
echo "🔨 Building Docker images..."
if docker-compose build --no-cache; then
    echo "✅ Build complete"
else
    echo "❌ Build failed. Check errors above."
    exit 1
fi
echo ""

# Start services
echo "🚀 Starting services..."
if docker-compose up -d; then
    echo "✅ Services started"
else
    echo "❌ Failed to start services"
    docker-compose ps
    exit 1
fi
echo ""

# Wait for database
echo "⏳ Waiting for database to be ready..."
sleep 10
echo ""

# Run migrations
echo "🔄 Running database migrations..."
docker-compose exec -T backend alembic upgrade head || {
    echo "⚠️  Migrations failed. Creating tables with init script..."
    docker-compose exec -T backend python -m app.core.init_db
}
echo "✅ Database initialized"
echo ""

# Load initial formulas
echo "📚 Loading formula library (30 formulas)..."
docker-compose exec -T backend python -m app.core.init_db
echo "✅ Formulas loaded"
echo ""

# Run health check
echo "🏥 Running health check..."
sleep 5

MAX_RETRIES=6
RETRY_COUNT=0
HEALTH_OK=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:${PORT:-8000}/health > /dev/null 2>&1; then
        HEALTH_CHECK=$(curl -s http://localhost:${PORT:-8000}/health 2>/dev/null || echo '{"status":"unknown"}')
        if echo "$HEALTH_CHECK" | grep -q "healthy"; then
            echo "✅ Health check passed"
            HEALTH_OK=true
            break
        fi
    fi
    
    ((RETRY_COUNT++))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "⏳ Waiting for services to be ready... (attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 5
    fi
done

if [ "$HEALTH_OK" = false ]; then
    echo "⚠️  Health check did not pass after $MAX_RETRIES attempts"
    echo "   Services may still be starting. Check with:"
    echo "   docker-compose ps"
    echo "   docker-compose logs backend"
fi
echo ""

# Display status
echo "=========================================="
echo "  ✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "🌐 Services:"
echo "   API:        http://localhost:${PORT:-8000}"
echo "   Docs:       http://localhost:${PORT:-8000}/docs"
echo "   Health:     http://localhost:${PORT:-8000}/health"
echo "   Metrics:    http://localhost:${PORT:-8000}/metrics"
echo ""
echo "🔑 Authentication:"
echo "   API Key:    $API_KEY"
echo "   Header:     X-API-Key: $API_KEY"
echo ""
echo "🧪 Test the API:"
echo '   curl -H "X-API-Key: '$API_KEY'" http://localhost:'${PORT:-8000}'/api/v1/formulas'
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f backend"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"
echo ""
echo "📖 See devops/docs/DEPLOYMENT.md for more information"
echo "=========================================="
