"""
ML Framework Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# Core dependencies
core_requirements = [
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "joblib>=1.1.0",
    "tqdm>=4.62.0",
    "pyyaml>=6.0",
    "fastapi>=0.109.0,<0.110.0",
    "uvicorn[standard]>=0.27.0,<0.28.0",
    "pydantic>=2.6.0,<2.7.0",
    "pydantic-settings>=2.1.0,<2.2.0",
    "python-multipart>=0.0.6,<0.0.8",
]

# Full ML dependencies
ml_requirements = core_requirements + [
    "xgboost>=1.5.0",
    "lightgbm>=3.3.0",
    "catboost>=1.0.0",
    "torch>=1.10.0,<2.3.0",
    "torchvision>=0.11.0",
    "tensorflow>=2.8.0,<3.0.0",
    "transformers>=4.20.0",
    "opencv-python>=4.5.0",
    "albumentations>=1.3.0",
    "optuna>=3.0.0",
    "ray[tune]>=2.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=7.3.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "jupyter>=1.0.0",
]

# Model serving dependencies
serving_requirements = [
    "onnx>=1.12.0",
    "onnxruntime>=1.12.0",
    "fastapi>=0.95.0",
    "uvicorn[standard]>=0.21.0",
]

setup(
    name="ml-framework",
    version="1.0.0",
    author="ML Framework Team",
    author_email="ml-team@example.com",
    description="Comprehensive ML framework with training, evaluation, and deployment tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-framework",
    packages=find_packages(
        exclude=[
            "tests",
            "docs",
            "examples",
            "fastapi",
            "fastapi.*",
            "sqlalchemy",
            "sqlalchemy.*",
            "pydantic_settings",
            "pydantic_settings.*",
        ]
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "full": ml_requirements,
        "dev": dev_requirements,
        "serving": serving_requirements,
        "all": ml_requirements + dev_requirements + serving_requirements,
    },
    entry_points={
        "console_scripts": [
            "ml-train=ml_framework.cli:train",
            "ml-evaluate=ml_framework.cli:evaluate",
            "ml-serve=ml_framework.cli:serve",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
