# 🚀 Formula Execution API

A production-ready FastAPI backend for executing engineering formulas with API key authentication, rate limiting, and Docker support.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Available Formulas](#-available-formulas)
- [API Endpoints](#-api-endpoints)
- [Installation](#-installation)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Rate Limiting](#-rate-limiting)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [License](#-license)

## ✨ Features

- 🔐 **API Key Authentication**: Secure access with X-API-Key header
- ⏱️ **Rate Limiting**: Redis-based distributed rate limiting (10 requests/minute)
- 🧮 **Engineering Formulas**: 8+ pre-built formulas for structural and fluid mechanics
- 🐳 **Docker Support**: Complete containerization with Docker Compose
- 📊 **PostgreSQL Database**: Ready for formula storage and history
- ⚡ **Redis Cache**: High-performance rate limiting and caching
- ✅ **Comprehensive Testing**: Full test suite with pytest
- 📚 **Auto-generated Docs**: Interactive API docs with Swagger UI
- 🔄 **CI/CD Ready**: GitHub Actions workflow included

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Build containers
docker-compose build

# 2. Start services
docker-compose up -d

# 3. Check health
curl http://localhost:8000/health

# 4. View API documentation
open http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🧮 Available Formulas

### Structural Engineering

1. **beam_deflection_simply_supported** - Simply supported beam deflection
2. **beam_deflection_cantilever** - Cantilever beam deflection
3. **beam_stress** - Bending stress calculation
4. **column_buckling** - Euler column buckling load

### Mechanical Engineering

5. **pressure_vessel_stress** - Thin-walled pressure vessel stress
6. **spring_deflection** - Helical spring deflection

### Fluid Mechanics

7. **reynolds_number** - Reynolds number calculation
8. **flow_velocity** - Flow velocity from volumetric flow rate

## 📡 API Endpoints

### Public Endpoints

- `GET /` - API information
- `GET /health` - Health check (no auth required)
- `GET /docs` - Interactive API documentation (Swagger UI)

### Protected Endpoints (Require API Key)

- `POST /api/v1/formulas/execute` - Execute a formula
- `GET /api/v1/formulas/list` - List all available formulas
- `GET /api/v1/formulas/{formula_id}` - Get formula information

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Redis (for rate limiting)

### Step-by-Step Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/formula-api.git
cd formula-api
```

2. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env and set your API_KEY and SECRET_KEY
```

3. **Install dependencies**

```bash
make install
# or
pip install -r requirements.txt
```

4. **Run with Docker**

```bash
make docker-build
make docker-run
```

The API will be available at `http://localhost:8000`

## 🐳 Docker Deployment

### Build and Run

```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

### Services

The Docker Compose stack includes:

- **backend**: FastAPI application (port 8000)
- **db**: PostgreSQL 16 database (port 5432)
- **redis**: Redis 7 for rate limiting (port 6379)

## 📝 API Usage Examples

### Execute a Formula

```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {
      "w": 10,
      "L": 5,
      "E": 200,
      "I": 0.0001
    }
  }'
```

**Response:**

```json
{
  "success": true,
  "formula_id": "beam_deflection_simply_supported",
  "result": 0.00065104,
  "unit": "m",
  "error": null,
  "execution_time_ms": 1.23,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### List All Formulas

```bash
curl -X GET http://localhost:8000/api/v1/formulas/list \
  -H "X-API-Key: YOUR_API_KEY"
```

### Get Formula Information

```bash
curl -X GET http://localhost:8000/api/v1/formulas/beam_deflection_simply_supported \
  -H "X-API-Key: YOUR_API_KEY"
```

## ⚙️ Configuration

### Environment Variables

Configure the application via `.env` file:

```bash
# API Configuration
API_V1_PREFIX=/api/v1
PROJECT_NAME=Formula Execution API
VERSION=1.0.0
ENVIRONMENT=development

# Security
API_KEY=your-api-key-change-this
SECRET_KEY=your-secret-key-change-this

# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/formulas

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Rate Limiting
RATE_LIMIT_PER_MINUTE=10
```

## ✅ Testing

### Run Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run tests in Docker
make docker-test
```

### Test Rate Limiting

```bash
# Run rate limit test script
make test-rate-limit
```

This will send 15 requests to test the rate limiting (limit is 10/minute).

### Manual Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with API key
export API_KEY="your-api-key"

for i in {1..15}; do
  curl -X POST http://localhost:8000/api/v1/formulas/execute \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "formula_id": "beam_deflection_simply_supported",
      "input_values": {"w":10,"L":5,"E":200,"I":0.0001}
    }'
  echo ""
done
```

## 🔒 Rate Limiting

The API implements distributed rate limiting using Redis:

- **Default Limit**: 10 requests per minute per API key
- **Burst Allowance**: 5 additional requests
- **Response**: HTTP 429 (Too Many Requests) when limit exceeded

Configure rate limits in `.env`:

```bash
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_BURST=5
```

## 📁 Project Structure

```
formula-api/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── formulas.py      # Formula API endpoints
│   ├── core/
│   │   ├── config.py            # Configuration settings
│   │   ├── security.py          # API key authentication
│   │   └── rate_limit.py        # Rate limiting middleware
│   ├── schemas/
│   │   └── formula.py           # Pydantic schemas
│   ├── services/
│   │   └── formula_service.py   # Formula execution logic
│   └── main.py                  # FastAPI application
├── tests/
│   └── test_app.py              # Test suite
├── .env.example                 # Environment template
├── docker-compose.yml           # Docker Compose config
├── Dockerfile                   # Docker image definition
├── Makefile                     # Task automation
├── requirements.txt             # Python dependencies
├── test_rate_limit.sh           # Rate limit test script
└── README.md                    # This file
```

## 🛠️ Development

### Available Make Commands

```bash
make help              # Show all commands
make install           # Install dependencies
make run               # Run locally (requires Redis)
make test              # Run tests
make test-cov          # Run tests with coverage
make test-rate-limit   # Test rate limiting
make docker-build      # Build Docker images
make docker-run        # Run Docker containers
make docker-stop       # Stop containers
make docker-logs       # View logs
make docker-test       # Run tests in Docker
make clean             # Clean cache files
```

### Adding New Formulas

1. Add formula to `app/services/formula_service.py`:

```python
"your_formula_id": {
    "name": "Your Formula Name",
    "description": "Formula description",
    "parameters": {
        "param1": "Parameter 1 description",
        "param2": "Parameter 2 description"
    },
    "unit": "unit",
    "category": "Category",
    "formula": lambda param1, param2: param1 * param2
}
```

2. Add tests in `tests/test_app.py`

3. Update documentation

### API Documentation

Interactive API documentation is automatically generated:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🚢 Deployment

### Production Checklist

- [ ] Change `API_KEY` and `SECRET_KEY` in `.env`
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure proper database credentials
- [ ] Set up SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up monitoring and logging
- [ ] Enable backup for PostgreSQL
- [ ] Configure Redis persistence

### Deploy to Cloud

#### AWS (ECS)

```bash
# Build and push image
docker build -t formula-api .
docker tag formula-api:latest YOUR_ECR_REPO/formula-api:latest
docker push YOUR_ECR_REPO/formula-api:latest
```

#### Google Cloud (Cloud Run)

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/formula-api
gcloud run deploy formula-api --image gcr.io/YOUR_PROJECT/formula-api --platform managed
```

#### Heroku

```bash
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

- 📖 Documentation: http://localhost:8000/docs
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/formula-api/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/formula-api/discussions)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Redis](https://redis.io/) - Rate limiting and caching
- [PostgreSQL](https://www.postgresql.org/) - Database
- [Uvicorn](https://www.uvicorn.org/) - ASGI server

---

Built with ❤️ using [FastAPI](https://fastapi.tiangolo.com/)
