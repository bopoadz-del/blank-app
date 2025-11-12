# Formula Execution API - Standalone Guide

## Overview

The Formula Execution API is a fully functional standalone backend service that **does not require** the React frontend UI to operate. This guide explains how to use the API independently.

## Quick Start

### API Endpoints

The API is accessible at `http://localhost:8000` (or your deployed URL) with the following key endpoints:

- **`/`** - API information and available endpoints
- **`/docs`** - Interactive Swagger UI documentation  
- **`/health`** - Health check endpoint
- **`/api/v1/formulas/*`** - Formula execution and management endpoints

### Root Endpoint Response

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "name": "Formula Execution API",
  "version": "1.0.0",
  "environment": "development",
  "docs": "/docs",
  "health": "/health"
}
```

## Running the API

### Option 1: Local Development

1. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Set environment variables (optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the API:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or use the Makefile:
   ```bash
   make run
   ```

4. **Access the API:**
   - API Root: http://localhost:8000/
   - Swagger Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the API at http://localhost:8000
```

## API Documentation

### Interactive Documentation (Swagger UI)

Access the full API documentation with interactive testing at:
```
http://localhost:8000/docs
```

The Swagger UI provides:
- Complete API endpoint documentation
- Request/response schemas
- Interactive "Try it out" functionality
- Authentication testing

### Alternative Documentation (ReDoc)

Access alternative documentation format at:
```
http://localhost:8000/redoc
```

## Using the API

### Health Check

Check if the API is running:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "timestamp": "2025-11-12T05:41:27.442575"
}
```

### List Available Formulas

```bash
curl -H "X-API-Key: test-api-key-12345" \
     http://localhost:8000/api/v1/formulas/list
```

### Execute a Formula

```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {
      "w": 10.0,
      "L": 5.0,
      "E": 200.0,
      "I": 0.0001
    },
    "convert_to_unit": "mm"
  }'
```

**Response:**
```json
{
  "success": true,
  "formula_id": "beam_deflection_simply_supported",
  "result": 0.65104,
  "unit": "mm",
  "original_unit": "m",
  "error": null,
  "execution_time_ms": 1.23,
  "execution_id": 42,
  "mlflow_run_id": "abc123def456",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Authentication

The API uses API key authentication. Include your API key in requests:

```bash
-H "X-API-Key: your-api-key-here"
```

**Default Test API Key:** `test-api-key-12345`

**Note:** Change the default API key in production by setting the `API_KEY` environment variable.

## Configuration

The API can be configured via environment variables or `.env` file:

### Core Settings
```bash
# API Configuration
PROJECT_NAME="Formula Execution API"
VERSION="1.0.0"
ENVIRONMENT="development"
API_V1_PREFIX="/api/v1"

# Server
HOST="0.0.0.0"
PORT=8000

# Security
SECRET_KEY="your-secret-key"
API_KEY="your-api-key"

# Database (Optional - uses SQLite if not specified)
DATABASE_URL="postgresql://user:pass@localhost:5432/formulas"

# Redis (Optional - uses in-memory fallback if not available)
REDIS_HOST="localhost"
REDIS_PORT=6379

# Rate Limiting
RATE_LIMIT_PER_MINUTE=10
```

## No External Dependencies Required

The API is designed to work standalone:

- ✅ **No Frontend Required** - Works without React UI
- ✅ **Optional Redis** - Falls back to in-memory rate limiting
- ✅ **Optional Database** - Can use SQLite instead of PostgreSQL
- ✅ **Built-in Documentation** - Swagger UI included

## Testing

Run the test suite to verify API functionality:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=app --cov-report=html tests/

# Test specific functionality
pytest tests/test_app.py -v
```

All tests verify the API works independently without requiring any frontend components.

## Deployment

### Render.com

The API is configured for easy deployment on Render.com:

1. Connect your GitHub repository
2. Render automatically detects the `render.yaml` configuration
3. API deploys with minimal dependencies from `backend/requirements.txt`

### Docker Deployment

```bash
# Build the Docker image
docker build -t formula-api .

# Run the container
docker run -p 8000:8000 formula-api

# Or use docker-compose
docker-compose up -d
```

## Common Use Cases

### 1. API-Only Microservice

Use the API as a microservice in your architecture:
```python
import requests

api_url = "http://your-api-url.com"
api_key = "your-api-key"

response = requests.post(
    f"{api_url}/api/v1/formulas/execute",
    headers={"X-API-Key": api_key},
    json={
        "formula_id": "reynolds_number",
        "input_values": {
            "rho": 1000,
            "v": 2,
            "L": 0.5,
            "mu": 0.001
        }
    }
)

result = response.json()
print(f"Result: {result['result']} {result['unit']}")
```

### 2. CLI Integration

Integrate with command-line tools:
```bash
#!/bin/bash
API_KEY="test-api-key-12345"
API_URL="http://localhost:8000"

# Execute formula via CLI
result=$(curl -s -X POST "$API_URL/api/v1/formulas/execute" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "formula_id": "spring_deflection",
    "input_values": {"F": 100, "k": 1000}
  }')

echo $result | jq '.result'
```

### 3. Backend Service for Mobile Apps

Use as a backend for iOS/Android applications:
```swift
// Swift example
let url = URL(string: "http://api-url.com/api/v1/formulas/execute")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.setValue("your-api-key", forHTTPHeaderField: "X-API-Key")

let body: [String: Any] = [
    "formula_id": "beam_deflection_simply_supported",
    "input_values": ["w": 10, "L": 5, "E": 200, "I": 0.0001]
]
request.httpBody = try? JSONSerialization.data(withJSONObject: body)

// Execute request...
```

## Troubleshooting

### API Not Starting

**Issue:** `Connection refused` errors

**Solution:** Ensure the API is running:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Rate Limiting Errors

**Issue:** `429 Too Many Requests`

**Solution:** Adjust rate limits in `.env`:
```bash
RATE_LIMIT_PER_MINUTE=100
```

### Authentication Errors

**Issue:** `403 Forbidden`

**Solution:** Include API key in request:
```bash
-H "X-API-Key: test-api-key-12345"
```

## Support

For more information:
- **API Documentation:** `/docs` endpoint
- **Health Status:** `/health` endpoint
- **Backend API Guide:** See `BACKEND_API.md`
- **Deployment Guide:** See `DEPLOYMENT.md`

## Summary

The Formula Execution API is a **fully functional standalone backend service** that:

✅ Works without frontend UI  
✅ Provides comprehensive API documentation via Swagger UI  
✅ Includes built-in health checks  
✅ Supports API key authentication  
✅ Works with or without Redis/PostgreSQL  
✅ Can be deployed independently  

**No frontend required!** Access everything you need through the REST API and Swagger UI at `/docs`.
