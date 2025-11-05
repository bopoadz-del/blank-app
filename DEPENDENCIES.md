# Dependencies Guide

Complete guide for installing and managing dependencies for the ML Framework.

## Quick Install

### Option 1: pip (Recommended)

```bash
# Install core dependencies
pip install -r requirements.txt

# Install with development tools
pip install -r requirements-dev.txt

# Install with GPU support
pip install -r requirements-gpu.txt
```

### Option 2: conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ml-framework
```

### Option 3: Package Installation

```bash
# Install as editable package
pip install -e .

# Install with all extras
pip install -e ".[all]"

# Install specific extras
pip install -e ".[dev]"
pip install -e ".[serving]"
```

## Dependency Categories

### Core Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Array operations and numerical computing |
| scipy | >=1.7.0 | Scientific computing and optimization |
| pandas | >=1.3.0 | Data manipulation and analysis |

### Machine Learning

#### Traditional ML
| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >=1.0.0 | Classic ML algorithms |
| xgboost | >=1.5.0 | Gradient boosting (GPU support available) |
| lightgbm | >=3.3.0 | Fast gradient boosting (GPU support available) |
| catboost | >=1.0.0 | Gradient boosting with categorical features |

#### Deep Learning - PyTorch
| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=1.10.0,<2.3.0 | Deep learning framework |
| torchvision | >=0.11.0 | Computer vision models and transforms |
| torchaudio | >=0.10.0 | Audio processing |

#### Deep Learning - TensorFlow
| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | >=2.8.0,<3.0.0 | Deep learning framework |

### NLP and Transformers

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | >=4.20.0 | Pre-trained language models |
| tokenizers | >=0.13.0 | Fast text tokenization |
| sentencepiece | >=0.1.96 | Unsupervised text tokenizer |

### Computer Vision

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | >=4.5.0 | Image processing and computer vision |
| Pillow | >=9.0.0 | Image loading and manipulation |
| albumentations | >=1.3.0 | Image augmentation library |

### Hyperparameter Optimization

| Package | Version | Purpose |
|---------|---------|---------|
| optuna | >=3.0.0 | Hyperparameter optimization |
| ray[tune] | >=2.0.0 | Distributed hyperparameter tuning |

### Model Serving

| Package | Version | Purpose |
|---------|---------|---------|
| onnx | >=1.12.0 | Model export format |
| onnxruntime | >=1.12.0 | ONNX model inference |
| joblib | >=1.1.0 | Model serialization |

### Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | >=3.5.0 | Plotting and visualization |
| seaborn | >=0.11.0 | Statistical visualization |

## Platform-Specific Installation

### macOS

```bash
# Install with Homebrew Python
brew install python@3.10

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows

```bash
# Using Anaconda (recommended)
conda env create -f environment.yml
conda activate ml-framework

# Or using pip
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## GPU Installation

### NVIDIA GPU Setup

#### Prerequisites
1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed
3. CUDA Toolkit (11.8 or 12.1)
4. cuDNN library

#### Install CUDA (Ubuntu)

```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### Install GPU Dependencies

```bash
# Install GPU-accelerated packages
pip install -r requirements-gpu.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
```

### GPU vs CPU Performance

| Task | CPU | GPU (CUDA) | Speedup |
|------|-----|------------|---------|
| Image Classification (ResNet-50) | 1x | 10-15x | 10-15x |
| NLP (BERT Fine-tuning) | 1x | 8-12x | 8-12x |
| XGBoost Training | 1x | 3-5x | 3-5x |
| Data Preprocessing | 1x | 1-2x | 1-2x |

## Development Setup

### Complete Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify installation
pytest tests/
```

### IDE Setup

#### VS Code

Install recommended extensions:
- Python
- Pylance
- Jupyter
- Black Formatter
- isort

Create `.vscode/settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

#### PyCharm

1. Configure interpreter: Settings → Project → Python Interpreter
2. Enable Black formatter: Settings → Tools → Black
3. Configure pytest: Settings → Tools → Python Integrated Tools

## Troubleshooting

### Common Issues

#### Issue: torch installation fails

```bash
# Solution: Install PyTorch separately
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Issue: TensorFlow GPU not detected

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow>=2.12.0
```

#### Issue: ray installation fails on Windows

```bash
# Solution: Use conda
conda install -c conda-forge ray-tune
```

#### Issue: opencv import error

```bash
# Solution: Install system dependencies (Linux)
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgomp1

# Or use headless version
pip uninstall opencv-python
pip install opencv-python-headless
```

## Dependency Management

### Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade transformers

# Check for outdated packages
pip list --outdated
```

### Creating Locked Requirements

```bash
# Generate exact versions
pip freeze > requirements-lock.txt

# Install from locked file
pip install -r requirements-lock.txt
```

### Minimal Installation

For minimal installation (core only):

```bash
pip install numpy scipy pandas scikit-learn joblib
```

## Docker Setup

### Dockerfile Example

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

CMD ["python", "main.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  ml-framework:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Version Compatibility Matrix

| Python | PyTorch | TensorFlow | Transformers | CUDA |
|--------|---------|------------|--------------|------|
| 3.8 | 1.10-2.0 | 2.8-2.15 | 4.20+ | 11.8 |
| 3.9 | 1.10-2.1 | 2.8-2.15 | 4.20+ | 11.8 |
| 3.10 | 1.10-2.2 | 2.8-2.15 | 4.20+ | 11.8, 12.1 |
| 3.11 | 2.0-2.2 | 2.12-2.15 | 4.30+ | 11.8, 12.1 |

## License Information

Most dependencies are licensed under permissive licenses:
- NumPy, SciPy, Pandas: BSD
- scikit-learn: BSD-3
- PyTorch: BSD-style
- TensorFlow: Apache 2.0
- Transformers: Apache 2.0

Check individual package licenses for commercial use.

## Support

For dependency-related issues:
1. Check this documentation
2. Search GitHub issues
3. Consult package documentation
4. Create issue with version info: `pip freeze > versions.txt`
