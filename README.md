# The Reasoner AI Platform

**Universal Mathematical Reasoning Infrastructure with Machine Learning**

A production-ready system that combines symbolic reasoning, deep learning, and traditional ML to provide context-aware mathematical solutions across multiple domains (construction, energy, finance, manufacturing).

## 🚀 Core Components

### 1. Formula Execution API
FastAPI backend for executing engineering formulas with authentication, rate limiting, and Docker support.

### 2. Deep Learning Framework
Comprehensive PyTorch implementation with CNNs, RNNs, LSTMs, Transformers, transfer learning, and mixed precision training.

### 3. Traditional ML Framework
Complete Scikit-learn implementation with Random Forest, XGBoost, LightGBM, SVM, clustering, and dimensionality reduction.

## 📁 Project Structure

```
reasoner-platform/
├── app/                          # Formula API (FastAPI)
│   ├── api/v1/                   # API endpoints
│   ├── core/                     # Config, security, rate limiting
│   ├── schemas/                  # Pydantic models
│   └── services/                 # Formula execution logic
│
├── deep-learning/                # Deep Learning Framework (PyTorch)
│   ├── architectures/            # CNN, RNN, LSTM, Transformer
│   ├── training/                 # Trainer, transfer learning
│   ├── optimizers/               # SGD, Adam, AdamW
│   ├── data/                     # DataLoaders, augmentation
│   ├── utils/                    # GPU utils, mixed precision
│   └── examples/                 # Training scripts
│
├── traditional-ml/               # Traditional ML (Scikit-learn)
│   ├── classifiers/              # RF, XGBoost, SVM, KNN, etc.
│   ├── clustering/               # K-Means, DBSCAN
│   ├── dimensionality_reduction/ # PCA, t-SNE, LDA
│   ├── feature_selection/        # Statistical selection
│   └── examples/                 # ML pipelines
│
├── automated-retraining/         # MLOps Pipeline
│   └── data_collection/          # Data collection service
│
├── jetson-rtdetr/                # RT-DETR for Jetson
│   ├── models/                   # TensorRT models
│   ├── inference/                # Inference engine
│   └── benchmarks/               # FPS benchmarking
│
└── yolov8-jetson/                # YOLOv8 for Jetson
    ├── model/                    # YOLOv8 Nano
    └── engine/                   # INT8 TensorRT engine
```

## ✨ Key Features

### Formula Execution API
- 🔐 API Key Authentication
- ⏱️ Redis-based Rate Limiting (10 req/min)
- 🧮 8+ Engineering Formulas (structural, fluid mechanics)
- 🐳 Docker Compose deployment
- 📊 PostgreSQL + Redis + MLflow integration

### Deep Learning Framework (5,800+ lines)
- **Architectures**: ResNet, LSTM, BiLSTM, Transformer
- **Training**: Complete training loop with forward/backward pass
- **Transfer Learning**: Pretrained models, layer freezing, fine-tuning
- **Optimizers**: Custom SGD, Adam, AdamW from scratch
- **GPU**: Multi-GPU (DataParallel, DistributedDataParallel)
- **Mixed Precision**: AMP for 2-3x speedup

### Traditional ML Framework (3,087+ lines)
- **Classification**: RF, XGBoost, LightGBM, SVM, KNN, Decision Trees, Naive Bayes, Logistic Regression
- **Clustering**: K-Means, DBSCAN
- **Dimensionality Reduction**: PCA, LDA, t-SNE
- **Feature Selection**: Univariate statistical tests

### Edge Computing
- **RT-DETR**: Real-time detection transformer for Jetson
- **YOLOv8**: 270 FPS inference with INT8 quantization
- **TensorRT**: GPU-accelerated inference

## 🚀 Quick Start

### Formula API
```bash
# Start services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Execute formula
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001}
  }'
```

### Deep Learning
```python
from architectures.cnn import ResNet18
from training.trainer import Trainer, TrainingConfig

# Create model
model = ResNet18(num_classes=10)

# Train
config = TrainingConfig(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    use_amp=True  # Mixed precision
)
trainer = Trainer(config)
trainer.train()
```

### Traditional ML
```python
from classifiers.ensemble import RandomForestClassifierWrapper, XGBoostClassifierWrapper
from feature_selection.statistical import UnivariateSelector

# Feature selection
selector = UnivariateSelector(score_func='f_classif', k=10)
X_selected = selector.fit_transform(X, y)

# Train XGBoost
xgb = XGBoostClassifierWrapper(n_estimators=100)
xgb.fit(X_selected, y)
predictions = xgb.predict(X_test)
```

## 📚 Documentation

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Framework Documentation
- **Deep Learning**: See `deep-learning/README.md`
- **Traditional ML**: See `traditional-ml/README.md`
- **RT-DETR**: See `jetson-rtdetr/README.md`
- **YOLOv8**: See `yolov8-jetson/README.md`

## 🧪 Testing

```bash
# API tests
pytest app/tests/

# Deep learning tests
cd deep-learning
python -m pytest

# Traditional ML tests
cd traditional-ml
python examples/complete_ml_pipeline.py
```

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Services running:
# - backend: FastAPI (port 8000)
# - db: PostgreSQL (port 5432)
# - redis: Redis (port 6379)
# - mlflow: MLflow tracking (port 5000)

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## 📊 Technology Stack

**Backend**: FastAPI, Python 3.11+, Pydantic, SQLAlchemy
**Database**: PostgreSQL 16, Redis 7
**ML Tracking**: MLflow 2.19.0
**Deep Learning**: PyTorch, TensorRT, CUDA
**Traditional ML**: Scikit-learn, XGBoost, LightGBM
**Edge**: NVIDIA Jetson (Orin Nano), TensorRT
**DevOps**: Docker, Docker Compose, GitHub Actions

## 🔧 Configuration

Environment variables (`.env`):

```bash
# API Configuration
API_KEY=your-api-key-change-this
SECRET_KEY=your-secret-key-change-this

# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/formulas

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Rate Limiting
RATE_LIMIT_PER_MINUTE=10
```

## 📈 Code Statistics

- **Formula API**: 2,000+ lines
- **Deep Learning**: 5,800+ lines
- **Traditional ML**: 3,087+ lines
- **Edge Computing**: 3,925+ lines
- **MLOps**: 1,331+ lines
- **Total**: 16,143+ lines of production code

## 🎯 Use Cases

### Engineering & Construction
- Beam deflection calculations
- Column buckling analysis
- Pressure vessel stress
- Concrete strength prediction

### Machine Learning
- Image classification with CNNs
- Sequence modeling with LSTMs
- Text classification with Transformers
- Tabular data with XGBoost

### Edge Computing
- Real-time object detection (270 FPS)
- Video analytics on Jetson devices
- Offline-first edge processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License

## 🔗 Links

- API Documentation: http://localhost:8000/docs
- MLflow Tracking: http://localhost:5000
- GitHub: https://github.com/yourusername/reasoner-platform

---

Built with ❤️ using FastAPI, PyTorch, and Scikit-learn
