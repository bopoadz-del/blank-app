# 🔌 API Quick Reference

**Test the API with these curl commands**

---

## 🔑 Authentication

All requests (except /health and /metrics) require API key:

```bash
# Get your API key from .env
grep API_KEY .env

# Set as variable
export API_KEY="your_api_key_here"
```

---

## 🏥 Health & Status

### Health Check
```bash
curl http://localhost:8000/health | jq
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-03T06:00:00",
  "version": "1.0.0",
  "components": {
    "database": {"status": "up"},
    "redis": {"status": "up"},
    "formulas": {"status": "up", "count": 30}
  }
}
```

### Root
```bash
curl http://localhost:8000/ | jq
```

### Metrics (Prometheus)
```bash
curl http://localhost:8000/metrics
```

---

## 📚 Formulas

### List All Formulas
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas | jq
```

### List by Domain
```bash
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/v1/formulas?domain=structural_engineering" | jq
```

### Get Specific Formula
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas/beam_deflection_simply_supported | jq
```

### Filter by Confidence
```bash
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/v1/formulas?min_confidence=0.8" | jq
```

---

## ⚡ Execute Formulas

### Simple Beam Deflection
```bash
curl -X POST \
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
     http://localhost:8000/api/v1/formulas/execute | jq
```

### NPV Calculation
```bash
curl -X POST \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "formula_id": "npv_net_present_value",
       "input_values": {
         "cash_flows": [-100000, 30000, 40000, 50000],
         "r": 0.10
       }
     }' \
     http://localhost:8000/api/v1/formulas/execute | jq
```

### Concrete Strength
```bash
curl -X POST \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "formula_id": "concrete_compressive_strength_maturity",
       "input_values": {
         "S_ultimate": 40,
         "k": 0.005,
         "maturity": 500
       }
     }' \
     http://localhost:8000/api/v1/formulas/execute | jq
```

---

## 🔄 Unit Conversion

### Convert Units
```bash
curl -X POST \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "value": 100,
       "from_unit": "psi",
       "to_unit": "MPa"
     }' \
     http://localhost:8000/api/v1/units/convert | jq
```

**Response:**
```json
{
  "original_value": 100,
  "original_unit": "psi",
  "converted_value": 0.689,
  "converted_unit": "MPa"
}
```

### Get Unit Info
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/units/info/MPa | jq
```

---

## 📊 Execution History

### List Executions
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/executions | jq
```

### Get Specific Execution
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/executions/{execution_id} | jq
```

### Get Formula Analytics
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas/beam_deflection_simply_supported/analytics | jq
```

---

## 🧪 Test Scenarios

### Test Suite
```bash
#!/bin/bash
# Save as test_api.sh

API_KEY="your_key_here"
BASE_URL="http://localhost:8000"

echo "1. Health Check"
curl -s $BASE_URL/health | jq -r '.status'

echo -e "\n2. List Formulas"
FORMULA_COUNT=$(curl -s -H "X-API-Key: $API_KEY" $BASE_URL/api/v1/formulas | jq 'length')
echo "Formulas available: $FORMULA_COUNT"

echo -e "\n3. Execute Beam Deflection"
RESULT=$(curl -s -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"formula_id":"beam_deflection_simply_supported","input_values":{"w":10,"L":5,"E":200,"I":0.0001}}' \
  $BASE_URL/api/v1/formulas/execute)
echo $RESULT | jq -r '.status'

echo -e "\n4. Convert Units"
curl -s -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"value":100,"from_unit":"psi","to_unit":"MPa"}' \
  $BASE_URL/api/v1/units/convert | jq

echo -e "\n✅ All tests complete"
```

---

## 📝 Common Workflows

### 1. Execute Formula with Validation
```bash
curl -X POST \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "formula_id": "beam_deflection_simply_supported",
       "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001},
       "expected_output": {"deflection": 0.00203},
       "context_data": {
         "project": "Building A",
         "location": "Floor 3",
         "engineer": "John Doe"
       }
     }' \
     http://localhost:8000/api/v1/formulas/execute | jq
```

### 2. Batch Processing (Shell Script)
```bash
#!/bin/bash
# process_batch.sh

for beam_load in 5 10 15 20; do
  echo "Processing load: $beam_load kN/m"
  
  curl -s -X POST \
       -H "X-API-Key: $API_KEY" \
       -H "Content-Type: application/json" \
       -d "{
         \"formula_id\": \"beam_deflection_simply_supported\",
         \"input_values\": {\"w\": $beam_load, \"L\": 5, \"E\": 200, \"I\": 0.0001}
       }" \
       http://localhost:8000/api/v1/formulas/execute | jq '.output_values'
  
  sleep 1
done
```

### 3. Monitor Performance
```bash
#!/bin/bash
# monitor.sh

while true; do
  clear
  echo "=== API Monitor ==="
  echo ""
  
  # Health
  HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
  echo "Status: $HEALTH"
  
  # Formula count
  COUNT=$(curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/formulas | jq 'length')
  echo "Formulas: $COUNT"
  
  # Response time
  START=$(date +%s%N)
  curl -s http://localhost:8000/health > /dev/null
  END=$(date +%s%N)
  DURATION=$(( ($END - $START) / 1000000 ))
  echo "Response: ${DURATION}ms"
  
  sleep 5
done
```

---

## 🔍 Debugging

### Verbose Request
```bash
curl -v \
     -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas
```

### Save Response
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas \
     -o formulas.json
```

### Pretty Print
```bash
curl -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas | \
     jq '.' | less
```

### Check Response Time
```bash
curl -w "\nTime: %{time_total}s\n" \
     -H "X-API-Key: $API_KEY" \
     http://localhost:8000/api/v1/formulas \
     -o /dev/null -s
```

---

## 📖 Interactive Docs

Open in browser:
```
http://localhost:8000/docs
```

This provides:
- ✅ Interactive API testing
- ✅ Request/response examples
- ✅ Schema documentation
- ✅ Try it out functionality

---

## 🐍 Python Example

```python
import requests

API_KEY = "your_key_here"
BASE_URL = "http://localhost:8000"

headers = {"X-API-Key": API_KEY}

# Execute formula
response = requests.post(
    f"{BASE_URL}/api/v1/formulas/execute",
    headers=headers,
    json={
        "formula_id": "beam_deflection_simply_supported",
        "input_values": {
            "w": 10,
            "L": 5,
            "E": 200,
            "I": 0.0001
        }
    }
)

print(response.json())
```

---

## 📚 All Endpoints

```
GET  /                                    # API info
GET  /health                              # Health check
GET  /metrics                             # Prometheus metrics
GET  /docs                                # API documentation
GET  /api/v1/formulas                     # List formulas
GET  /api/v1/formulas/{formula_id}        # Get formula
POST /api/v1/formulas/execute             # Execute formula
GET  /api/v1/executions                   # List executions
GET  /api/v1/executions/{execution_id}    # Get execution
POST /api/v1/units/convert                # Convert units
GET  /api/v1/units/info/{unit}            # Unit info
GET  /api/v1/formulas/{id}/analytics      # Formula analytics
```

---

**Tip:** Always check `/docs` for the most up-to-date API documentation!
