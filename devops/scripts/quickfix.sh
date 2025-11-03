#!/bin/bash
#
# ==============================================================================
# Quick Fix Script
# ==============================================================================
# Fixes common issues automatically
#
# Usage: ./quickfix.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=========================================="
echo "  Quick Fix"
echo "=========================================="
echo ""

FIXED=0

# Fix 1: Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x "$PROJECT_ROOT/devops/scripts/"*.sh && echo "  ✅ Fixed" && ((FIXED++))

# Fix 2: Create .env if missing
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "📄 Creating .env from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env" && echo "  ✅ Fixed" && ((FIXED++))
fi

# Fix 3: Create data directories
echo "📁 Creating data directories..."
mkdir -p "$PROJECT_ROOT/data/formulas" \
         "$PROJECT_ROOT/data/datasets" \
         "$PROJECT_ROOT/data/backups" \
         "$PROJECT_ROOT/logs" && echo "  ✅ Fixed" && ((FIXED++))

# Fix 4: Generate secrets if using defaults
if grep -q "your-secret-key-here" "$PROJECT_ROOT/.env" 2>/dev/null; then
    if command -v openssl >/dev/null 2>&1; then
        echo "🔑 Generating SECRET_KEY..."
        NEW_SECRET=$(openssl rand -hex 32)
        sed -i.bak "s/SECRET_KEY=.*/SECRET_KEY=$NEW_SECRET/" "$PROJECT_ROOT/.env"
        echo "  ✅ Fixed"
        ((FIXED++))
    else
        echo "  ⚠️  OpenSSL not found - cannot generate SECRET_KEY"
    fi
fi

if grep -q "your-api-key-here" "$PROJECT_ROOT/.env" 2>/dev/null; then
    if command -v openssl >/dev/null 2>&1; then
        echo "🔑 Generating API_KEY..."
        NEW_API_KEY=$(openssl rand -hex 32)
        sed -i.bak "s/API_KEY=.*/API_KEY=$NEW_API_KEY/" "$PROJECT_ROOT/.env"
        echo "  ✅ Fixed"
        echo "  📝 Your new API Key: $NEW_API_KEY"
        echo "     SAVE THIS KEY!"
        ((FIXED++))
    fi
fi

echo ""
echo "=========================================="
echo "  Fixed $FIXED issue(s)"
echo "=========================================="
echo ""
echo "Run ./devops/scripts/preflight.sh to verify"
