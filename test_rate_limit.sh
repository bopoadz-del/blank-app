#!/bin/bash

# Rate Limiting Test Script
# This script tests the rate limiting functionality of the API

API_KEY="${API_KEY:-your-api-key-change-this}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Testing Rate Limiting..."
echo "API Key: $API_KEY"
echo "Base URL: $BASE_URL"
echo ""

for i in {1..15}; do
  echo "Request #$i"
  curl -X POST "$BASE_URL/api/v1/formulas/execute" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "formula_id": "beam_deflection_simply_supported",
      "input_values": {
        "w": 10,
        "L": 5,
        "E": 200,
        "I": 0.0001
      }
    }' \
    -s | python3 -m json.tool 2>/dev/null || echo "Rate limit exceeded or error"
  echo ""
  echo "---"
done
