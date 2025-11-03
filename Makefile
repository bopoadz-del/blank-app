.PHONY: help install run test clean docker-build docker-run docker-stop format lint

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make run          - Run the Streamlit app locally"
	@echo "  make test         - Run tests with pytest"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make clean        - Remove cache and build files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-stop  - Stop Docker container"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the app locally
run:
	streamlit run streamlit_app.py

# Run tests
test:
	pytest tests/

# Run tests with coverage
test-cov:
	pytest --cov=. --cov-report=html --cov-report=term tests/

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
	docker-compose logs -f

# Code formatting (optional - requires black)
format:
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Run: pip install black"; exit 1; }
	black .

# Code linting (optional - requires flake8)
lint:
	@command -v flake8 >/dev/null 2>&1 || { echo "flake8 not installed. Run: pip install flake8"; exit 1; }
	flake8 . --exclude=venv,env,.venv,.git,__pycache__
