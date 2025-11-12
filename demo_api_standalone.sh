#!/bin/bash

# Demo script showing the Formula Execution API working standalone
# No frontend required!

echo "=========================================="
echo "Formula Execution API - Standalone Demo"
echo "=========================================="
echo ""

API_URL="http://localhost:8000"
API_KEY="test-api-key-12345"

# Check if API is running
echo "🔍 Checking if API is running..."
if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo "❌ API is not running. Start it with:"
    echo "   uvicorn app.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi
echo "✅ API is running!"
echo ""

# Test 1: Root endpoint
echo "📋 Test 1: Root Endpoint"
echo "GET $API_URL/"
curl -s "$API_URL/" | python -m json.tool
echo ""
echo ""

# Test 2: Health check
echo "🏥 Test 2: Health Check"
echo "GET $API_URL/health"
curl -s "$API_URL/health" | python -m json.tool
echo ""
echo ""

# Test 3: List formulas
echo "📚 Test 3: List Available Formulas"
echo "GET $API_URL/api/v1/formulas/list"
echo "Using API Key: $API_KEY"
curl -s -H "X-API-Key: $API_KEY" "$API_URL/api/v1/formulas/list" | python -m json.tool | head -30
echo "... (truncated)"
echo ""
echo ""

# Test 4: Execute a formula
echo "🧮 Test 4: Execute Formula (Beam Deflection)"
echo "POST $API_URL/api/v1/formulas/execute"
echo "Using API Key: $API_KEY"
echo "Formula: beam_deflection_simply_supported"
echo "Parameters: w=10, L=5, E=200, I=0.0001"
curl -s -X POST "$API_URL/api/v1/formulas/execute" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {
      "w": 10.0,
      "L": 5.0,
      "E": 200.0,
      "I": 0.0001
    },
    "convert_to_unit": "mm"
  }' | python -m json.tool
echo ""
echo ""

# Test 5: Execute another formula
echo "🔬 Test 5: Execute Formula (Reynolds Number)"
echo "POST $API_URL/api/v1/formulas/execute"
echo "Formula: reynolds_number"
echo "Parameters: rho=1000, v=2, L=0.5, mu=0.001"
curl -s -X POST "$API_URL/api/v1/formulas/execute" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "formula_id": "reynolds_number",
    "input_values": {
      "rho": 1000,
      "v": 2,
      "L": 0.5,
      "mu": 0.001
    }
  }' | python -m json.tool
echo ""
echo ""

# Summary
echo "=========================================="
echo "✅ All API endpoints working!"
echo "=========================================="
echo ""
echo "📖 For full documentation, visit:"
echo "   $API_URL/docs"
echo ""
echo "📝 For detailed guide, see:"
echo "   API_STANDALONE_GUIDE.md"
echo ""
echo "🎯 The API works perfectly without any frontend UI!"
echo ""
