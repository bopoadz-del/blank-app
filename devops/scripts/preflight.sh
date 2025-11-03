#!/bin/bash
#
# ==============================================================================
# Pre-Flight Check Script
# ==============================================================================
# Validates the package before deployment
# Run this to check if everything is ready
#
# Usage: ./preflight.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

ERRORS=0
WARNINGS=0

echo "=========================================="
echo "  Pre-Flight Check"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# ==============================================================================
# CHECK 1: Project Structure
# ==============================================================================
echo "📁 Checking project structure..."

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $2"
        return 0
    else
        echo "  ❌ $2 - NOT FOUND"
        ((ERRORS++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "  ✅ $2"
        return 0
    else
        echo "  ❌ $2 - NOT FOUND"
        ((ERRORS++))
        return 1
    fi
}

# Core files
check_file "$PROJECT_ROOT/docker-compose.yml" "docker-compose.yml"
check_file "$PROJECT_ROOT/.env.example" ".env.example"
check_file "$PROJECT_ROOT/alembic.ini" "alembic.ini"

# Backend
check_dir "$PROJECT_ROOT/backend" "backend/"
check_file "$PROJECT_ROOT/backend/Dockerfile" "backend/Dockerfile"
check_file "$PROJECT_ROOT/backend/requirements.txt" "backend/requirements.txt"
check_file "$PROJECT_ROOT/backend/app/main.py" "backend/app/main.py"

# Alembic
check_dir "$PROJECT_ROOT/alembic" "alembic/"
check_file "$PROJECT_ROOT/alembic/env.py" "alembic/env.py"
check_file "$PROJECT_ROOT/alembic/versions/001_initial.py" "alembic/versions/001_initial.py"

# Data
check_dir "$PROJECT_ROOT/data" "data/"
check_file "$PROJECT_ROOT/data/formulas/initial_library.json" "data/formulas/initial_library.json"

# DevOps
check_dir "$PROJECT_ROOT/devops" "devops/"
check_file "$PROJECT_ROOT/devops/scripts/deploy.sh" "devops/scripts/deploy.sh"
check_file "$PROJECT_ROOT/devops/scripts/backup.sh" "devops/scripts/backup.sh"
check_file "$PROJECT_ROOT/devops/scripts/restore.sh" "devops/scripts/restore.sh"

echo ""

# ==============================================================================
# CHECK 2: Python Syntax
# ==============================================================================
echo "🐍 Checking Python syntax..."

if command -v python3 >/dev/null 2>&1; then
    PYTHON_FILES=(
        "$PROJECT_ROOT/backend/app/main.py"
        "$PROJECT_ROOT/backend/app/services/reasoner.py"
        "$PROJECT_ROOT/backend/app/services/orchestration.py"
        "$PROJECT_ROOT/alembic/env.py"
    )
    
    for file in "${PYTHON_FILES[@]}"; do
        if [ -f "$file" ]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                echo "  ✅ $(basename $file)"
            else
                echo "  ❌ $(basename $file) - SYNTAX ERROR"
                ((ERRORS++))
            fi
        fi
    done
else
    echo "  ⚠️  Python3 not found - skipping syntax check"
    ((WARNINGS++))
fi

echo ""

# ==============================================================================
# CHECK 3: Script Executability
# ==============================================================================
echo "🔧 Checking script permissions..."

SCRIPTS=(
    "$PROJECT_ROOT/devops/scripts/deploy.sh"
    "$PROJECT_ROOT/devops/scripts/backup.sh"
    "$PROJECT_ROOT/devops/scripts/restore.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo "  ✅ $(basename $script) - executable"
    else
        echo "  ⚠️  $(basename $script) - not executable (will fix)"
        chmod +x "$script" 2>/dev/null && echo "     Fixed!" || echo "     Could not fix"
        ((WARNINGS++))
    fi
done

echo ""

# ==============================================================================
# CHECK 4: Docker Prerequisites
# ==============================================================================
echo "🐳 Checking Docker..."

if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version)
    echo "  ✅ Docker installed: $DOCKER_VERSION"
else
    echo "  ❌ Docker not installed"
    ((ERRORS++))
fi

if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_VERSION=$(docker-compose --version)
    echo "  ✅ Docker Compose installed: $COMPOSE_VERSION"
else
    echo "  ❌ Docker Compose not installed"
    ((ERRORS++))
fi

echo ""

# ==============================================================================
# CHECK 5: Formula Library
# ==============================================================================
echo "📚 Checking formula library..."

FORMULA_FILE="$PROJECT_ROOT/data/formulas/initial_library.json"
if [ -f "$FORMULA_FILE" ]; then
    if command -v python3 >/dev/null 2>&1; then
        FORMULA_COUNT=$(python3 -c "import json; print(len(json.load(open('$FORMULA_FILE'))))" 2>/dev/null || echo "0")
        if [ "$FORMULA_COUNT" -eq 30 ]; then
            echo "  ✅ Formula library: $FORMULA_COUNT formulas"
        elif [ "$FORMULA_COUNT" -gt 0 ]; then
            echo "  ⚠️  Formula library: $FORMULA_COUNT formulas (expected 30)"
            ((WARNINGS++))
        else
            echo "  ❌ Formula library: Invalid JSON or empty"
            ((ERRORS++))
        fi
    else
        echo "  ⚠️  Cannot verify formula count (Python not available)"
        ((WARNINGS++))
    fi
else
    echo "  ❌ Formula library not found"
    ((ERRORS++))
fi

echo ""

# ==============================================================================
# CHECK 6: Configuration
# ==============================================================================
echo "⚙️  Checking configuration..."

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "  ✅ .env file exists"
    
    # Check for dangerous defaults
    if grep -q "change_this_password" "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo "  ⚠️  .env contains default passwords - CHANGE BEFORE PRODUCTION"
        ((WARNINGS++))
    fi
    
    if grep -q "your-secret-key-here" "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo "  ⚠️  .env contains default SECRET_KEY - CHANGE BEFORE PRODUCTION"
        ((WARNINGS++))
    fi
else
    echo "  ⚠️  .env file not found (will be created from .env.example)"
    ((WARNINGS++))
fi

echo ""

# ==============================================================================
# CHECK 7: Required Dependencies in requirements.txt
# ==============================================================================
echo "📦 Checking requirements.txt..."

REQ_FILE="$PROJECT_ROOT/backend/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    REQUIRED_DEPS=("alembic" "prometheus-client" "fastapi" "sqlalchemy" "redis")
    
    for dep in "${REQUIRED_DEPS[@]}"; do
        if grep -q "^$dep" "$REQ_FILE"; then
            echo "  ✅ $dep"
        else
            echo "  ❌ $dep - NOT FOUND"
            ((ERRORS++))
        fi
    done
else
    echo "  ❌ requirements.txt not found"
    ((ERRORS++))
fi

echo ""

# ==============================================================================
# CHECK 8: Docker Compose Configuration
# ==============================================================================
echo "🐋 Checking docker-compose.yml..."

COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
if [ -f "$COMPOSE_FILE" ]; then
    SERVICES=("postgres" "redis" "backend" "mlflow")
    
    for service in "${SERVICES[@]}"; do
        if grep -q "^  $service:" "$COMPOSE_FILE"; then
            echo "  ✅ Service: $service"
        else
            echo "  ⚠️  Service: $service - NOT FOUND"
            ((WARNINGS++))
        fi
    done
    
    # Check for Redis
    if grep -q "redis:" "$COMPOSE_FILE"; then
        echo "  ✅ Redis configured"
    else
        echo "  ⚠️  Redis not found in docker-compose"
        ((WARNINGS++))
    fi
else
    echo "  ❌ docker-compose.yml not found"
    ((ERRORS++))
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ All checks passed! Ready to deploy."
    echo ""
    echo "Next steps:"
    echo "  1. Review .env configuration"
    echo "  2. Run: ./devops/scripts/deploy.sh production"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  Passed with $WARNINGS warning(s)"
    echo ""
    echo "Package is usable but review warnings above."
    echo "You can proceed with deployment if warnings are acceptable."
    exit 0
else
    echo "❌ Failed with $ERRORS error(s) and $WARNINGS warning(s)"
    echo ""
    echo "Fix the errors above before deploying."
    exit 1
fi
