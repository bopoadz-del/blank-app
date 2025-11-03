# 🎈 Streamlit Production Template

A comprehensive, production-ready Streamlit application template with best practices, testing, CI/CD, and Docker support.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)
[![CI](https://github.com/yourusername/blank-app/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/blank-app/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Docker](#-docker)
- [Testing](#-testing)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🎯 **Production-Ready**: Configured for production deployment
- 🐳 **Docker Support**: Full containerization with Docker and Docker Compose
- ✅ **Testing**: Comprehensive test suite with pytest
- 🔄 **CI/CD**: Automated testing and deployment with GitHub Actions
- 📊 **Data Explorer**: Upload and analyze CSV files
- 📈 **Visualizations**: Interactive charts and graphs
- ⚙️ **Configuration**: Environment-based configuration with `.env` support
- 📝 **Documentation**: Well-documented code and comprehensive README
- 🎨 **Custom Theme**: Streamlit configuration with custom styling
- 🛠️ **Makefile**: Common tasks automated with make commands

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blank-app.git
cd blank-app

# Install dependencies
make install

# Run the app
make run
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
make docker-build
make docker-run

# View at http://localhost:8501
```

### Option 3: Direct Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## 📦 Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip (Python package manager)
- Docker (optional, for containerization)

### Step-by-Step Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/blank-app.git
cd blank-app
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the application**

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 🎯 Usage

### Application Sections

The app includes four main sections accessible from the sidebar:

1. **Home**: Overview with metrics and quick start guide
2. **Data Explorer**: Upload and analyze CSV files
3. **Visualizations**: Interactive charts and maps
4. **About**: Information about the app and tech stack

### Example: Data Explorer

1. Navigate to the "Data Explorer" section
2. Upload a CSV file or view the sample data
3. Explore the data preview, statistics, and information
4. Download or analyze your data

### Example: Visualizations

1. Navigate to the "Visualizations" section
2. View interactive line charts, area charts, and maps
3. Customize the visualizations as needed

## 🐳 Docker

### Using Docker Compose (Recommended)

```bash
# Build the image
docker-compose build

# Run the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker CLI

```bash
# Build the image
docker build -t streamlit-app .

# Run the container
docker run -d -p 8501:8501 --name streamlit-app streamlit-app

# View logs
docker logs -f streamlit-app

# Stop and remove
docker stop streamlit-app
docker rm streamlit-app
```

## ✅ Testing

### Run Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_app.py -v
```

### Test Structure

```
tests/
├── __init__.py
└── test_app.py       # Application tests
```

### Writing Tests

Tests are written using pytest. Example:

```python
def test_streamlit_import():
    """Test that streamlit can be imported."""
    import streamlit as st
    assert st is not None
```

## 🛠️ Development

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make run           # Run the app locally
make test          # Run tests
make test-cov      # Run tests with coverage
make clean         # Remove cache and build files
make docker-build  # Build Docker image
make docker-run    # Run Docker container
make docker-stop   # Stop Docker container
make format        # Format code with black
make lint          # Lint code with flake8
```

### Code Formatting

```bash
# Install development dependencies
pip install black flake8

# Format code
make format

# Lint code
make lint
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📁 Project Structure

```
blank-app/
├── .devcontainer/
│   └── devcontainer.json       # Dev container configuration
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # CI/CD pipeline
│   └── CODEOWNERS              # Code ownership
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── tests/
│   ├── __init__.py
│   └── test_app.py             # Application tests
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── LICENSE                     # MIT License
├── Makefile                    # Task automation
├── packages.txt                # System packages
├── pytest.ini                  # Pytest configuration
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── streamlit_app.py            # Main application
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# App Configuration
APP_NAME="My Streamlit App"
APP_VERSION="1.0.0"
ENVIRONMENT="development"

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Streamlit Configuration

Customize the app appearance in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
enableCORS = false
```

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your app with one click

### Docker Deployment

```bash
# Build and push to Docker Hub
docker build -t yourusername/streamlit-app:latest .
docker push yourusername/streamlit-app:latest

# Deploy on your server
docker pull yourusername/streamlit-app:latest
docker run -d -p 8501:8501 yourusername/streamlit-app:latest
```

### Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

### AWS / GCP / Azure

Refer to the respective cloud provider documentation for container deployment:

- [AWS ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Follow PEP 8 style guide
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Docker](https://www.docker.com/) for containerization
- [pytest](https://pytest.org/) for testing framework
- [GitHub Actions](https://github.com/features/actions) for CI/CD

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/blank-app/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/blank-app/discussions)

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [pytest Documentation](https://docs.pytest.org/)

---

Built with ❤️ using [Streamlit](https://streamlit.io/)
