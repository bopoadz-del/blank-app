.PHONY: help install run test clean docker-build docker-run docker-stop docker-logs format lint test-rate-limit

# Default target
help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make run            - Run the FastAPI app locally"
	@echo "  make test           - Run tests with pytest"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-rate-limit - Test rate limiting manually"
	@echo "  make clean          - Remove cache and build files"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-run     - Run Docker containers"
	@echo "  make docker-stop    - Stop Docker containers"
	@echo "  make docker-logs    - View Docker logs"
	@echo "  make docker-test    - Run tests inside Docker"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code with flake8"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the app locally (requires Redis running)
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest --cov=app --cov-report=html --cov-report=term tests/

# Test rate limiting
test-rate-limit:
	@chmod +x test_rate_limit.sh
	./test_rate_limit.sh

# Clean cache and build files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f backend

docker-test:
	docker-compose exec backend pytest -v

# Code formatting (optional - requires black)
format:
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Run: pip install black"; exit 1; }
	black app/ tests/

# Code linting (optional - requires flake8)
lint:
	@command -v flake8 >/dev/null 2>&1 || { echo "flake8 not installed. Run: pip install flake8"; exit 1; }
	flake8 app/ tests/ --exclude=venv,env,.venv,.git,__pycache__ --max-line-length=120
