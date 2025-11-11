# Copilot Instructions for The Reasoner AI Platform

## 🎯 Project Overview

This is **The Reasoner AI Platform** - an enterprise-grade mathematical reasoning infrastructure combining symbolic reasoning, machine learning, and autonomous formula execution with ethical safeguards. The platform serves engineering, finance, energy, and manufacturing domains.

### Key Technologies
- **Backend**: FastAPI (Python 3.9-3.11), SQLAlchemy, PostgreSQL, Redis
- **Frontend**: React 18+, TypeScript 5+, Vite, TailwindCSS
- **ML Stack**: PyTorch, TensorFlow, scikit-learn, Ray Tune, MLflow
- **Deployment**: Docker, Render.com, Jetson AGX Orin for edge devices
- **Authentication**: JWT with OAuth 2.0 (Google Drive integration)

### Architecture
```
Frontend (React/TypeScript) 
    ↓ REST API
Backend (FastAPI/Python)
    ↓
Data Layer (PostgreSQL + MLflow)
```

---

## 🚀 Setup and Installation

### Backend Setup
1. **Python Version**: Use Python 3.9, 3.10, or 3.11 (3.11 recommended)
2. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt  # For production/deployment
   # OR
   pip install -r requirements.txt  # For full ML development
   ```
3. **Environment Variables**: Copy `.env.example` to `.env` and configure:
   - `DATABASE_URL`: PostgreSQL connection string
   - `REDIS_URL`: Redis connection string
   - `SECRET_KEY`: JWT secret key
   - `OPENAI_API_KEY`: OpenAI API key (optional)
   - `GOOGLE_DRIVE_*`: Google Drive OAuth credentials (optional)

4. **Database Setup**:
   ```bash
   # Run migrations
   alembic upgrade head
   ```

5. **Run Backend Locally**:
   ```bash
   make run
   # OR
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup
1. **Node.js Version**: Use Node.js 18+ (check `frontend/.nvmrc` if present)
2. **Install Dependencies**:
   ```bash
   cd frontend
   npm install
   ```
3. **Environment Variables**: Create `frontend/.env` with:
   - `VITE_API_URL`: Backend API URL (default: `http://localhost:8000`)

4. **Run Frontend Locally**:
   ```bash
   cd frontend
   npm run dev
   ```

### Docker Setup
```bash
# Build and run all services
make docker-build
make docker-run

# View logs
make docker-logs
```

---

## 🧪 Testing

### Backend Tests
- **Framework**: pytest with pytest-asyncio and pytest-cov
- **Location**: `backend/tests/` and root `tests/` directory
- **Run Tests**:
  ```bash
  make test
  # OR
  pytest tests/ -v
  ```
- **Test Coverage**:
  ```bash
  make test-cov
  # OR
  pytest --cov=app --cov-report=html --cov-report=term tests/
  ```
- **Test Configuration**: See `pytest.ini` and `backend/pytest.ini`

### Frontend Tests
- **Framework**: Vitest (if configured)
- **Location**: `frontend/src/**/*.test.tsx` or `frontend/tests/`
- **Run Tests**:
  ```bash
  cd frontend
  npm test  # If test script exists
  ```

### Rate Limiting Tests
```bash
make test-rate-limit
# OR
./test_rate_limit.sh
```

### CI/CD Tests
- Tests run automatically on push/PR via GitHub Actions (`.github/workflows/ci.yml`)
- Tested on Python 3.9, 3.10, and 3.11
- Coverage reports uploaded to Codecov

---

## 🎨 Code Style and Linting

### Backend (Python)
- **Formatter**: Black (if available)
  ```bash
  make format
  # OR
  black app/ tests/
  ```
- **Linter**: flake8
  ```bash
  make lint
  # OR
  flake8 app/ tests/ --max-line-length=120
  ```
- **Configuration**: 
  - Max line length: 120 characters
  - Exclude: `venv`, `env`, `.venv`, `.git`, `__pycache__`

### Frontend (TypeScript/React)
- **Linter**: ESLint
  ```bash
  cd frontend
  npm run lint
  ```
- **Formatter**: Prettier
  ```bash
  cd frontend
  npm run format
  ```
- **Configuration**: See `frontend/.eslintrc` and `frontend/.prettierrc`

---

## 📁 Project Structure

### Backend Structure
```
backend/
├── app/
│   ├── api/                  # API route handlers
│   │   ├── data_ingestion_routes.py  # Google Drive integration
│   │   ├── corrections_routes.py     # Corrections workflow
│   │   ├── certification_routes.py   # Tier certification
│   │   ├── ethical_routes.py         # Ethical safeguards
│   │   ├── safety_routes.py          # Safety monitoring
│   │   └── edge_device_routes.py     # Edge device management
│   ├── core/                 # Core configuration
│   │   ├── config.py         # Environment settings
│   │   ├── database.py       # Database connection
│   │   └── security.py       # Authentication/authorization
│   ├── models/               # SQLAlchemy ORM models
│   │   ├── auth.py           # User and token models
│   │   ├── corrections.py    # Corrections and certifications
│   │   ├── ethical_layer.py  # Credibility system
│   │   ├── safety_layer.py   # Safety monitoring
│   │   └── edge_devices.py   # Jetson device models
│   ├── schemas/              # Pydantic schemas
│   ├── services/             # Business logic
│   │   ├── reasoner.py       # Formula execution engine
│   │   ├── data_ingestion.py # Google Drive connector
│   │   ├── validation_pipeline.py # Multi-stage validation
│   │   ├── ethical_safeguards.py  # Ethical layer logic
│   │   └── safety_pipeline.py     # Safety monitoring logic
│   └── main.py               # FastAPI application entry point
├── tests/                    # Backend tests
├── requirements.txt          # Minimal dependencies (Render deployment)
├── requirements-full.txt     # Complete ML stack
└── requirements-minimal.txt  # Absolute minimum
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/           # Reusable React components
│   │   ├── TierBadge.tsx    # Credibility tier badges
│   │   ├── FormulaCard.tsx  # Formula display cards
│   │   ├── DeploymentWizard.tsx # One-click deployment
│   │   └── ...
│   ├── pages/                # Page components
│   │   ├── FormulaCatalog.tsx    # Formula catalog portal
│   │   ├── DashboardEnhanced.tsx # Chat interface
│   │   ├── FormulaExecution.tsx  # Formula execution
│   │   ├── AdminPanel.tsx        # Admin dashboard
│   │   └── AuditorDashboard.tsx  # Audit logs
│   ├── services/
│   │   └── api.ts            # API client with axios
│   ├── types/
│   │   └── index.ts          # TypeScript type definitions
│   └── App.tsx               # Main application component
└── package.json
```

### Root-Level ML Directories
- `deep-learning/`: Deep learning models and training scripts
- `traditional-ml/`: Classical ML algorithms
- `nlp-framework/`: Natural language processing tools
- `computer-vision/`: Computer vision models
- `yolov8-jetson/`, `jetson-rtdetr/`: Edge-optimized models for Jetson devices
- `automated-retraining/`: Continuous learning pipelines
- `model-optimization/`: Model compression and optimization
- `jetson-client/`: Edge device client code

---

## 🤝 Contribution Guidelines

### Code Changes
- **Make minimal changes**: Only modify code that directly addresses the task
- **Preserve existing functionality**: Do not remove or break working code unless absolutely necessary
- **Follow existing patterns**: Match the coding style and patterns already in the codebase
- **Add comments sparingly**: Only add comments if they match existing style or explain complex logic

### Testing Requirements
- **Always add tests** for new features or bug fixes
- **Run tests before committing**: Use `make test` or `pytest`
- **Maintain test coverage**: Aim for similar or higher coverage than existing code
- **Test both success and error cases**

### Pull Request Process
1. **Create descriptive commits**: Follow conventional commit format if possible
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for test additions
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

2. **Update documentation** if you change:
   - API endpoints
   - Configuration options
   - Setup/installation steps
   - Project structure

3. **Run the full test suite** before submitting:
   ```bash
   make test        # Backend tests
   make test-cov    # With coverage
   cd frontend && npm run lint  # Frontend linting
   ```

### Branch Naming
- Use descriptive branch names: `feature/add-formula-validation`, `fix/auth-token-expiry`, `docs/update-readme`

---

## ⚠️ Important Constraints and Guidelines

### Security
- **Never commit secrets**: Use environment variables for all sensitive data
- **JWT Authentication**: All protected routes require valid JWT tokens
- **Rate Limiting**: API has rate limits; test with `make test-rate-limit`
- **Input Validation**: Always validate and sanitize user inputs using Pydantic schemas

### Performance
- **Database Queries**: Use SQLAlchemy ORM with proper indexing
- **Async Operations**: Backend uses async/await; maintain async patterns
- **Caching**: Redis is available for caching; use for frequently accessed data

### Ethical and Safety Layers
- **4-Tier Credibility System**: 
  - Tier 1: Experimental (requires approval)
  - Tier 2: Validated (requires monitoring)
  - Tier 3: Production-Ready (auto-deploy with oversight)
  - Tier 4: Auto-Deploy (fully autonomous)
- **Safety Monitoring**: 12 prohibited content categories (see `app/services/safety_pipeline.py`)
- **Do not bypass ethical safeguards**: These are critical security features

### Deployment
- **Render.com**: Primary deployment platform
- **Docker**: Use `docker-compose.yml` for local development
- **Environment-specific configs**: Use appropriate requirements files:
  - `backend/requirements.txt` for production
  - `requirements.txt` for full ML development
  - `requirements-gpu.txt` for GPU workloads

### Dependencies
- **Backend**: Do not upgrade Python packages without testing across Python 3.9-3.11
- **Frontend**: Maintain React 18+ and TypeScript 5+ compatibility
- **ML Libraries**: Be cautious with version updates; many models depend on specific versions

### Files to Avoid Modifying (Unless Explicitly Required)
- `.github/workflows/*`: CI/CD configurations (unless updating CI)
- `alembic/versions/*`: Database migrations (create new migrations instead)
- `docker-compose.yml`: Docker orchestration (unless changing infrastructure)
- Legacy directories: Some directories may contain older code; check before modifying

---

## 🔧 Development Tools

### Available Make Commands
```bash
make help             # Show all available commands
make install          # Install backend dependencies
make run              # Run FastAPI app locally
make test             # Run pytest tests
make test-cov         # Run tests with coverage
make test-rate-limit  # Test rate limiting
make clean            # Remove cache and build files
make docker-build     # Build Docker images
make docker-run       # Run Docker containers
make docker-stop      # Stop Docker containers
make docker-logs      # View Docker logs
make docker-test      # Run tests in Docker
make format           # Format code with black
make lint             # Lint code with flake8
```

### Useful Commands
```bash
# Backend
uvicorn app.main:app --reload  # Run with hot reload
alembic revision --autogenerate -m "description"  # Create migration
alembic upgrade head  # Apply migrations

# Frontend
cd frontend && npm run dev  # Run with hot reload
cd frontend && npm run build  # Build for production

# Docker
docker-compose up -d  # Start all services
docker-compose logs -f backend  # Follow backend logs
docker-compose exec backend pytest  # Run tests in container
```

---

## 📚 Key Documentation Files

- `README.md`: Project overview and quick start
- `DEPLOYMENT.md`: Production deployment guide
- `CI_CD_TESTING.md`: CI/CD and testing documentation
- `GOOGLE_DRIVE_SETUP.md`: Google Drive integration setup
- `CREDENTIALS_SETUP.md`: Credentials and environment configuration
- `PRE_DEPLOYMENT_CHECKLIST.md`: Pre-deployment checklist
- `PRODUCTION_DEPLOYMENT.md`: Production deployment details
- `BACKEND_API.md`: Backend API documentation
- `INTEGRATION_GUIDE.md`: Integration guidelines
- `QUICK_REFERENCE.md`: Quick reference guide

---

## 🚨 Common Pitfalls and Gotchas

1. **Redis Dependency**: Backend requires Redis to be running. Use Docker or install locally.
2. **Database Migrations**: Always run `alembic upgrade head` after pulling changes
3. **Environment Variables**: Copy `.env.example` and configure before running
4. **Python Version**: Code is tested on 3.9-3.11; avoid using 3.12+ features
5. **CORS Configuration**: Frontend and backend must be properly configured for cross-origin requests
6. **Google Drive Integration**: Requires OAuth setup and credentials (optional feature)
7. **ML Dependencies**: Full ML stack is large; use `backend/requirements.txt` for lighter deployments
8. **Test Database**: Tests may require a separate test database; check test configuration

---

## 💡 Tips for AI Coding Agents

- **Read existing code first**: Understand the patterns and structure before making changes
- **Test incrementally**: Run tests after each significant change
- **Use type hints**: Backend uses Pydantic; frontend uses TypeScript - maintain type safety
- **Follow async patterns**: Backend is fully async; don't mix sync and async incorrectly
- **Check dependencies**: Ensure new packages are compatible with Python 3.9-3.11
- **Respect the architecture**: Don't bypass the ethical/safety layers
- **Update tests**: Every feature change should include corresponding test updates
- **Document API changes**: Update docstrings and API documentation for endpoint changes

---

## 📞 Questions or Issues?

If you encounter issues or need clarification:
1. Check the relevant documentation file in the repo
2. Review existing tests for usage examples
3. Check `.env.example` for required configuration
4. Review CI/CD workflows for build and test procedures
