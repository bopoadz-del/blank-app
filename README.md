# 🧠 The Reasoner AI Platform

**Enterprise-Grade Mathematical Reasoning Infrastructure with Continuous Learning**

A production-ready platform combining symbolic reasoning, machine learning, and autonomous formula execution with ethical safeguards and credibility-based trust hierarchy. Built for engineering, finance, energy, and manufacturing domains.

[![Deploy to Render](https://img.shields.io/badge/Deploy-Render-7B42BC?logo=render)](https://render.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)

---

## 🎯 What's New - Phase 1 Complete!

### ✨ **Formula Catalog Portal** (Just Released!)
- 📚 **Searchable Formula Library** - Find formulas by name, domain, or equation
- 🎨 **Visual Tier System** - Color-coded badges for 4 credibility tiers
- ⚡ **One-Click Deployment** - 3-step wizard for cloud/edge deployment
- 📊 **Real-Time Stats** - Execution counts, success rates, confidence scores
- 🔍 **Advanced Filtering** - Filter by tier, domain, status with live results
- 📱 **Responsive Design** - Grid/List views, mobile-friendly interface

### 🔗 **Google Drive Integration**
- ☁️ Automatic file syncing from Google Drive
- 📄 Parse PDF, DOCX, XLSX, CSV, JSON
- 🤖 Auto-extract numerical data and context hints
- 🔐 OAuth 2.0 authentication for secure access

### 🔒 **Ethical & Safety Layers**
- 🛡️ 4-Tier Credibility System (Experimental → Auto-Deploy)
- ⚠️ Real-time safety monitoring (12 prohibited content categories)
- 📈 Context-aware autonomy levels
- 🚨 Emergency kill-switch protocols

---

## 🚀 Live Demo

**Production URL**: `https://ml-platform-frontend.onrender.com`

**Test Credentials**:
- Email: `admin@platform.local`
- Password: `admin123` ⚠️ (Change immediately after login!)

**Routes**:
- `/dashboard` - Main dashboard with chat interface
- `/catalog` - **NEW!** Formula Catalog Portal
- `/formulas` - Formula execution interface
- `/admin` - Admin panel (admin role required)
- `/auditor` - Audit logs dashboard (auditor role required)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Dashboard  │ │   Catalog    │ │   Execution  │        │
│  │              │ │   Portal ⭐  │ │   Interface  │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTPS/REST API
┌───────────────────────────┴─────────────────────────────────┐
│                 Backend (FastAPI + Python)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Ethical    │ │    Safety    │ │   Formula    │        │
│  │    Layer     │ │    Layer     │ │   Reasoner   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  Google      │ │   OpenAI     │ │    Edge      │        │
│  │  Drive API   │ │     API      │ │   Devices    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│              Data Layer (PostgreSQL + MLflow)                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Formulas   │ │  Corrections │ │    Audit     │        │
│  │   Database   │ │  & Feedback  │ │     Logs     │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
reasoner-platform/
├── frontend/                        # React Frontend (TypeScript)
│   ├── src/
│   │   ├── components/              # Reusable components
│   │   │   ├── TierBadge.tsx       # ⭐ Credibility tier badges
│   │   │   ├── FormulaCard.tsx     # ⭐ Formula display cards
│   │   │   ├── DeploymentWizard.tsx # ⭐ One-click deployment
│   │   │   ├── CorrectionModal.tsx  # Operator corrections
│   │   │   └── CertificationPanel.tsx # Admin certifications
│   │   ├── pages/
│   │   │   ├── FormulaCatalog.tsx  # ⭐ NEW! Catalog portal
│   │   │   ├── DashboardEnhanced.tsx # Chat interface
│   │   │   ├── FormulaExecution.tsx  # Formula runner
│   │   │   ├── AdminPanel.tsx        # Admin dashboard
│   │   │   └── AuditorDashboard.tsx  # Audit interface
│   │   ├── services/
│   │   │   └── api.ts               # API client
│   │   └── types/
│   │       └── index.ts             # TypeScript types
│   └── package.json
│
├── backend/                         # FastAPI Backend (Python)
│   ├── app/
│   │   ├── api/                     # API endpoints
│   │   │   ├── data_ingestion_routes.py # ⭐ Google Drive
│   │   │   ├── corrections_routes.py    # Corrections workflow
│   │   │   ├── certification_routes.py  # Tier certification
│   │   │   ├── ethical_routes.py        # Ethical layer
│   │   │   ├── safety_routes.py         # Safety layer
│   │   │   └── edge_device_routes.py    # Edge management
│   │   ├── core/
│   │   │   ├── config.py            # Environment configuration
│   │   │   ├── database.py          # PostgreSQL connection
│   │   │   └── security.py          # JWT authentication
│   │   ├── models/                  # SQLAlchemy models
│   │   │   ├── auth.py              # Users & tokens
│   │   │   ├── corrections.py       # Corrections & certs
│   │   │   ├── ethical_layer.py     # Credibility system
│   │   │   ├── safety_layer.py      # Safety monitoring
│   │   │   └── edge_devices.py      # Jetson devices
│   │   └── services/                # Business logic
│   │       ├── reasoner.py          # Formula execution
│   │       ├── data_ingestion.py    # ⭐ Google Drive connector
│   │       ├── validation_pipeline.py # Multi-stage validation
│   │       ├── ethical_safeguards.py  # Ethical layer
│   │       └── safety_pipeline.py     # Safety layer
│   ├── requirements.txt             # Minimal (free tier)
│   ├── requirements-full.txt        # Complete (paid tier)
│   └── start.sh                     # Startup script
│
├── jetson-client/                   # Jetson AGX Orin 32GB
│   ├── edge_client.py               # Edge device client
│   ├── model_sync.py                # Model synchronization
│   └── jetson_optimizations.py     # TensorRT optimization
│
├── docs/                            # Documentation
│   ├── DEPLOYMENT.md                # Render deployment guide
│   ├── GOOGLE_DRIVE_SETUP.md       # ⭐ Drive integration
│   ├── CREDENTIALS_SETUP.md        # ⭐ OAuth & API keys
│   └── RENDER_NETWORK_CONFIG.md    # ⭐ Network/IP config
│
└── render.yaml                      # Render Blueprint (IaC)
```

---

## ✨ Key Features

### 🎨 Formula Catalog Portal (Phase 1 - NEW!)
- **Search & Discovery**: Search by name, domain, equation, or description
- **Visual Tier System**:
  - 🧪 Tier 1 (Gray): Experimental - requires supervision
  - ✓ Tier 2 (Blue): Validated - 70%+ confidence
  - ✓✓ Tier 3 (Green): Certified - 95%+ confidence
  - ⚡ Tier 4 (Purple): Auto-Deploy - near-perfect accuracy
- **Advanced Filters**: Filter by tier, domain (7 categories), status
- **Formula Cards**: Expandable cards with stats, parameters, validation
- **Deployment Wizard**: 3-step deployment (Cloud/Edge/Hybrid)
- **View Modes**: Grid view (responsive) or List view
- **Real-Time Stats**: Execution count, success rate, confidence score

### 🛡️ Ethical & Safety Layers
- **4-Tier Credibility System**: Progressive autonomy based on validation
- **Context-Aware Overrides**: Climate, materials, site conditions
- **Safety Monitoring**: 12 prohibited content categories
- **Multi-Layer Detection**: Pattern matching + ML-based + confidence scoring
- **Emergency Protocols**: Kill-switch, isolation, rollback
- **Red Lines**: Hard limits that cannot be overridden

### 🤖 Edge Computing
- **Jetson AGX Orin Support**: 32GB edge devices
- **Model Synchronization**: Auto-sync from cloud to edge
- **TensorRT Optimization**: GPU-accelerated inference
- **Offline-First**: Works without internet connection
- **Heartbeat Monitoring**: Real-time device status

### 📊 Operator Workflow
- **Corrections System**: Operators can correct formula outputs
- **Admin Review**: Admin approval for corrections
- **Auto-Retrain**: Approved corrections trigger retraining
- **Certification**: Promote formulas through tier levels
- **Audit Trail**: Complete history of all changes

### 🔗 Integrations
- **Google Drive**: Automatic file syncing and parsing (PDF, DOCX, XLSX, CSV)
- **OpenAI API**: AI-powered features (ready for future enhancements)
- **Slack**: Notifications for critical events
- **MLflow**: Experiment tracking and model versioning

### 🔐 Security & Authentication
- **JWT Tokens**: Access + refresh tokens
- **Role-Based Access**: Operator, Admin, Auditor, System
- **API Rate Limiting**: Prevent abuse
- **Audit Logging**: Complete action history

---

## 🚀 Quick Start

### Local Development

#### Frontend:
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

#### Backend:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
# Runs on http://localhost:8000
```

### Render Deployment (Production)

1. **Fork this repository**

2. **Set environment variables** in Render Dashboard:
   ```bash
   # Required
   DATABASE_URL=<auto-generated by Render>
   SECRET_KEY=<auto-generated>

   # Optional
   OPENAI_API_KEY=sk-proj-...
   GOOGLE_DRIVE_FOLDER_ID=1MFvAWURZGw-...
   GOOGLE_OAUTH_CLIENT_ID=382554705937-...
   ```

3. **Deploy via Blueprint**:
   - Go to Render Dashboard
   - New → Blueprint
   - Connect repository
   - Render reads `render.yaml` and deploys automatically

4. **Access your platform**:
   - Frontend: `https://ml-platform-frontend.onrender.com`
   - Backend: `https://ml-platform-backend.onrender.com`
   - Docs: `https://ml-platform-backend.onrender.com/docs`

See `docs/DEPLOYMENT.md` for detailed instructions.

---

## 📚 API Documentation

### Interactive Docs
- **Swagger UI**: https://ml-platform-backend.onrender.com/docs
- **ReDoc**: https://ml-platform-backend.onrender.com/redoc

### Key Endpoints

#### Formula Execution
```bash
POST /api/v1/formulas/execute
{
  "formula_id": "beam_deflection",
  "input_values": {"w": 10, "L": 5, "E": 200, "I": 0.0001},
  "context_data": {"climate": "hot_arid", "material": "steel"}
}
```

#### Formula Catalog
```bash
GET /api/v1/formulas?tier=3&domain=structural_engineering
```

#### Google Drive Sync
```bash
POST /api/v1/drive/sync
GET /api/v1/drive/files
POST /api/v1/drive/parse/{file_id}
```

#### Corrections Workflow
```bash
POST /api/v1/corrections
PATCH /api/v1/corrections/{id}/review
```

#### Certification
```bash
POST /api/v1/certifications
GET /api/v1/formulas/{id}/certification-history
```

---

## 🎯 Use Cases

### Engineering & Construction
- Beam deflection calculations with context-aware corrections
- Column buckling analysis (climate-adjusted)
- Concrete strength prediction with material validation
- Pressure vessel stress with safety thresholds

### Formula Discovery & Management
- Search 100+ validated formulas across 7 domains
- Deploy formulas to cloud or edge devices
- Track formula performance and confidence scores
- Promote formulas through certification tiers

### Operator Workflow
- Execute formulas with confidence-based autonomy
- Correct outputs when formulas are wrong
- Review and approve corrections
- Certify formulas for higher tier levels

### Edge Computing
- Deploy formulas to Jetson AGX Orin devices
- Offline-first formula execution
- Auto-sync models and configurations
- Real-time monitoring and heartbeats

---

## 📊 Technology Stack

### Frontend
- **React 18+** with TypeScript
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **React Router** - Client-side routing
- **Axios** - HTTP client

### Backend
- **FastAPI 0.104+** - Modern Python web framework
- **SQLAlchemy** - ORM for PostgreSQL
- **Pydantic** - Data validation
- **JWT** - Authentication tokens
- **Loguru** - Structured logging
- **Prometheus** - Metrics collection

### Database & Storage
- **PostgreSQL 16** - Primary database (Render)
- **Google Drive** - File storage and syncing
- **MLflow** - Experiment tracking (optional)

### Integrations
- **Google Drive API** - File syncing and parsing
- **OpenAI API** - AI-powered features
- **Slack API** - Notifications
- **Render** - Cloud hosting (free tier supported)

### Edge Computing
- **NVIDIA Jetson AGX Orin 32GB** - Edge devices
- **TensorRT** - GPU-accelerated inference
- **PyTorch** - Deep learning framework

---

## 🎨 Credibility Tier System

| Tier | Name | Badge | Confidence | Autonomy | Deployment |
|------|------|-------|------------|----------|------------|
| 1 | Experimental | 🧪 Gray | < 70% | Human supervision required | Testing only |
| 2 | Validated | ✓ Blue | ≥ 70% | Semi-autonomous | Staging/Prod with review |
| 3 | Certified | ✓✓ Green | ≥ 95% | Mostly autonomous | Production |
| 4 | Auto-Deploy | ⚡ Purple | ≥ 99% | Fully autonomous | Production + Edge |

### Tier Progression
1. **Tier 1 → 2**: Requires 70% confidence + empirical validation
2. **Tier 2 → 3**: Requires 95% confidence + admin certification
3. **Tier 3 → 4**: Requires 99% confidence + extensive production testing

---

## 🔧 Configuration

### Environment Variables

**Backend** (`backend/.env`):
```bash
# Database (auto-configured on Render)
DATABASE_URL=postgresql://user:pass@host:5432/db

# Security
SECRET_KEY=<auto-generated-strong-key>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Google Drive Integration
GOOGLE_DRIVE_FOLDER_ID=1MFvAWURZGw-...
GOOGLE_OAUTH_CLIENT_ID=382554705937-...
GOOGLE_DRIVE_CREDENTIALS_BASE64=<base64-encoded-json>

# OpenAI API
OPENAI_API_KEY=sk-proj-...

# Edge Devices
EDGE_NODES=http://jetson1:8080,http://jetson2:8080
```

**Frontend** (`frontend/.env.production`):
```bash
VITE_API_URL=https://ml-platform-backend.onrender.com
```

See `docs/CREDENTIALS_SETUP.md` for complete setup guide.

---

## 📖 Documentation

### Setup Guides
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Complete Render deployment guide
- **[GOOGLE_DRIVE_SETUP.md](docs/GOOGLE_DRIVE_SETUP.md)** - Google Drive integration
- **[CREDENTIALS_SETUP.md](docs/CREDENTIALS_SETUP.md)** - OAuth & API keys
- **[RENDER_NETWORK_CONFIG.md](docs/RENDER_NETWORK_CONFIG.md)** - IP whitelisting

### API Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Spec**: `/openapi.json`

---

## 🧪 Testing

### Frontend Tests
```bash
cd frontend
npm run test
npm run test:coverage
```

### Backend Tests
```bash
cd backend
pytest
pytest --cov=app tests/
```

### End-to-End Tests
```bash
# Test formula execution
curl -X POST https://ml-platform-backend.onrender.com/api/v1/formulas/execute \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"formula_id": "test", "input_values": {}}'

# Test catalog search
curl https://ml-platform-backend.onrender.com/api/v1/formulas?domain=structural_engineering

# Test Google Drive sync
curl -X POST https://ml-platform-backend.onrender.com/api/v1/drive/sync \
  -H "Authorization: Bearer $TOKEN"
```

---

## 📈 Performance

### Free Tier (Render)
- **Backend**: 512 MB RAM, 0.1 CPU
- **Frontend**: Static site (unlimited bandwidth)
- **Database**: 1 GB storage, 1 month retention
- **Cost**: $0/month 🎉

### Resource Usage
- **Backend**: ~200 MB RAM (typical)
- **Frontend**: ~2 MB bundle size
- **Database**: ~100 MB (1000 formulas)
- **Cold Start**: ~30 seconds (free tier)

### Optimization
- Minimal requirements.txt (50 MB vs 700 MB)
- Static site frontend (no server needed)
- Efficient database queries with indexes
- Background tasks for heavy operations

---

## 🗺️ Roadmap

### ✅ Phase 1: Formula Catalog Portal (Complete!)
- Search and filter system
- Visual tier badges
- One-click deployment wizard
- Responsive design

### 🚧 Phase 2: Production Monitoring (Week 2)
- Real-time monitoring dashboard
- Performance metrics visualization
- Alert system integration
- SLA tracking

### 📅 Phase 3: Kubernetes Migration (Week 3)
- Kubernetes manifests
- Helm charts
- GitOps with ArgoCD
- Multi-cluster support

### 📅 Phase 4: Progressive Deployment (Week 4)
- Canary releases
- Blue-green deployments
- A/B testing framework
- Rollback automation

### 📅 Phase 5: Self-Service Workbench (Week 5-6)
- Formula builder UI
- Visual workflow designer
- Template library
- Collaboration features

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow existing code style
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Live Demo**: https://ml-platform-frontend.onrender.com
- **API Docs**: https://ml-platform-backend.onrender.com/docs
- **GitHub**: https://github.com/bopoadz-del/blank-app
- **Render Dashboard**: https://dashboard.render.com

---

## 📞 Support

- **GitHub Issues**: https://github.com/bopoadz-del/blank-app/issues
- **Documentation**: See `docs/` folder
- **Render Support**: https://render.com/docs

---

## 🎉 Acknowledgments

Built with modern open-source technologies:
- FastAPI for blazing-fast APIs
- React for dynamic UIs
- PostgreSQL for reliable data storage
- Render for easy deployment
- Google Drive API for file integration
- OpenAI API for AI capabilities

---

## 📊 Code Statistics

- **Frontend**: 1,533+ lines (React/TypeScript)
- **Backend**: 8,000+ lines (Python)
- **Components**: 20+ React components
- **API Endpoints**: 50+ routes
- **Database Models**: 15+ tables
- **Total**: 10,000+ lines of production code

---

**Built with ❤️ using FastAPI, React, and TypeScript**

**Deployed on Render** 🚀
