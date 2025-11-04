# Jetson ML Platform Client

Production-ready client for Jetson AGX Orin 32GB to connect with the ML Platform backend.

## Features

✅ **Device Registration** - Auto-registers Jetson device on startup
✅ **Heartbeat Monitoring** - Sends system metrics every 30 seconds
✅ **GPU Metrics** - Tracks GPU usage, temperature, CPU, memory, disk
✅ **OTA Updates** - Automatic model downloads and deployment
✅ **Headless Execution** - Runs ML models without user interaction
✅ **Multi-Framework Support** - PyTorch, TensorFlow, ONNX, scikit-learn
✅ **Auto-Start** - Systemd service for boot-time launch
✅ **Production Logging** - Comprehensive logging to journald and file

## Supported Hardware

- **Jetson AGX Orin 32GB** (Primary target)
- Jetson AGX Orin 64GB
- Jetson Orin NX 16GB
- Jetson Orin Nano 8GB
- Jetson AGX Xavier
- Jetson Xavier NX

## Requirements

### Software
- **JetPack 5.x or 6.x** (includes Ubuntu 20.04/22.04)
- **Python 3.8+** (pre-installed)
- **pip3** (pre-installed)
- **Internet connection** to backend server

### Python Packages
- `requests` - HTTP client
- `psutil` - System metrics
- `jetson-stats` (recommended) - Jetson-specific GPU monitoring
- ML frameworks as needed (PyTorch, TensorFlow, ONNX)

## Installation

### Quick Install

```bash
# Clone or copy the jetson-client directory to your Jetson
cd jetson-client

# Run installer (will prompt for sudo password)
sudo bash install.sh
```

### Manual Install

```bash
# 1. Create directories
sudo mkdir -p /opt/jetson-client/src
sudo mkdir -p /opt/ml-platform/models
sudo mkdir -p /etc/jetson-client

# 2. Copy files
sudo cp -r src/* /opt/jetson-client/src/
sudo cp config/config.json.template /etc/jetson-client/config.json

# 3. Install dependencies
sudo pip3 install -r requirements.txt

# 4. Install jetson-stats (recommended)
sudo -H pip3 install jetson-stats

# 5. Install systemd service
sudo cp jetson-client.service /etc/systemd/system/
sudo systemctl daemon-reload
```

## Configuration

Edit `/etc/jetson-client/config.json`:

```json
{
  "backend_url": "http://your-backend-server.com:8000",
  "api_key": null,
  "heartbeat_interval": 30,
  "device_name": "Jetson AGX Orin Lab A",
  "device_type": "jetson_orin",
  "location": "Lab A - Building 1",
  "models_dir": "/opt/ml-platform/models",
  "auto_update": true,
  "log_level": "INFO"
}
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `backend_url` | Backend server URL | Required |
| `api_key` | API key (if backend requires) | `null` |
| `heartbeat_interval` | Heartbeat interval in seconds | `30` |
| `device_name` | Human-readable device name | Hostname |
| `device_type` | Device type identifier | `jetson_orin` |
| `location` | Physical location description | `Unknown` |
| `models_dir` | Directory to store models | `/opt/ml-platform/models` |
| `auto_update` | Enable automatic OTA updates | `true` |
| `log_level` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## Usage

### Start the Service

```bash
# Start immediately
sudo systemctl start jetson-client

# Enable auto-start on boot
sudo systemctl enable jetson-client

# Check status
sudo systemctl status jetson-client
```

### View Logs

```bash
# Real-time logs (systemd journal)
sudo journalctl -u jetson-client -f

# File logs
sudo tail -f /var/log/jetson-client.log

# Last 100 lines
sudo journalctl -u jetson-client -n 100
```

### Stop the Service

```bash
sudo systemctl stop jetson-client
```

### Restart the Service

```bash
sudo systemctl restart jetson-client
```

## How It Works

### 1. Device Registration

On first startup, the client:
1. Generates a unique device ID (based on MAC address)
2. Collects hardware information (CPU, GPU, memory, JetPack version)
3. Registers with the backend via `POST /api/v1/edge-devices/register`
4. Receives confirmation and device configuration

### 2. Heartbeat Loop

Every 30 seconds (configurable), the client:
1. Collects system metrics (CPU, GPU, memory, disk, temperature)
2. Sends heartbeat to `POST /api/v1/edge-devices/{device_id}/heartbeat`
3. Receives response with pending OTA update notifications
4. Logs metrics for monitoring

### 3. OTA Updates

When a new Tier 1 certified model is available:
1. Backend notifies client via heartbeat response
2. Client downloads model from backend
3. Model is saved to `/opt/ml-platform/models/`
4. Model is loaded and activated
5. Client confirms deployment success/failure
6. Old models are cleaned up (keeps latest 3)

### 4. Formula Execution

Models are executed in headless mode:
1. Client loads active model on startup
2. Executes formulas on-demand or via schedule
3. Sends results back to backend
4. Results are tracked for drift detection

## System Metrics Collected

- **CPU Usage**: Overall CPU utilization (%)
- **Memory Usage**: RAM utilization (%)
- **Disk Usage**: Root partition usage (%)
- **GPU Usage**: GPU utilization (%) via tegrastats or jtop
- **Temperature**: Maximum thermal zone temperature (°C)

## Model Support

The client supports multiple ML frameworks:

### PyTorch
- `.pt`, `.pth` files
- Automatic GPU acceleration (CUDA)
- Example: Loaded via `torch.load()`

### TensorFlow
- `.h5`, `.pb`, `.keras` files
- TensorFlow for Jetson optimizations
- Example: Loaded via `tf.keras.models.load_model()`

### ONNX
- `.onnx` files
- ONNX Runtime with CUDA support
- Example: Loaded via `onnxruntime.InferenceSession()`

### Scikit-learn / Pickle
- `.pkl`, `.pickle` files
- Traditional ML models
- Example: Loaded via `pickle.load()`

## Troubleshooting

### Service won't start

```bash
# Check service status
sudo systemctl status jetson-client

# Check logs for errors
sudo journalctl -u jetson-client -n 50

# Verify configuration
cat /etc/jetson-client/config.json

# Test backend connectivity
curl http://your-backend-server.com:8000/health
```

### Cannot connect to backend

1. Verify `backend_url` in config.json
2. Check network connectivity: `ping your-backend-server.com`
3. Verify firewall allows outbound HTTP/HTTPS
4. Check backend is running: `curl http://backend:8000/health`

### GPU metrics not working

```bash
# Install jetson-stats
sudo -H pip3 install jetson-stats

# Reboot (required after first jtop install)
sudo reboot

# Verify jtop works
jtop

# Check if tegrastats is available
which tegrastats
```

### Models not downloading

1. Check disk space: `df -h`
2. Verify models directory exists and is writable
3. Check backend logs for download URL errors
4. Verify network bandwidth is sufficient

### Permission errors

```bash
# Ensure correct ownership
sudo chown -R root:root /opt/jetson-client
sudo chown -R root:root /etc/jetson-client

# Ensure correct permissions
sudo chmod +x /opt/jetson-client/src/jetson_client.py
sudo chmod 644 /etc/jetson-client/config.json
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Jetson AGX Orin 32GB            │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   Jetson Client (jetson_client.py)│ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  System Metrics Collector   │ │ │
│  │  │  (GPU, CPU, Temp, Memory)   │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Heartbeat Service (30s)    │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  Model Manager              │ │ │
│  │  │  (OTA, Load, Execute)       │ │ │
│  │  └─────────────────────────────┘ │ │
│  │                                   │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │  API Client                 │ │ │
│  │  │  (REST communication)       │ │ │
│  │  └─────────────────────────────┘ │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
                  │
                  │ HTTPS/HTTP
                  │
        ┌─────────▼──────────┐
        │   Backend Server   │
        │   (FastAPI)        │
        │                    │
        │  - Registration    │
        │  - Heartbeat API   │
        │  - OTA Updates     │
        │  - Drift Detection │
        │  - Auto-Retrain    │
        └────────────────────┘
```

## Development

### Running in Development Mode

```bash
# Navigate to source directory
cd jetson-client/src

# Run directly (not as service)
sudo python3 jetson_client.py ../config/config.json
```

### Testing Individual Components

```bash
# Test system metrics
python3 system_metrics.py

# Test API client
python3 api_client.py

# Test model manager
python3 model_manager.py
```

## Files and Directories

```
jetson-client/
├── src/
│   ├── jetson_client.py      # Main application
│   ├── system_metrics.py     # Metrics collection
│   ├── api_client.py          # Backend communication
│   └── model_manager.py       # Model management
├── config/
│   └── config.json.template   # Configuration template
├── requirements.txt           # Python dependencies
├── jetson-client.service      # Systemd service file
├── install.sh                 # Installation script
└── README.md                  # This file

Installation directories:
/opt/jetson-client/src/        # Application code
/opt/ml-platform/models/       # Downloaded models
/etc/jetson-client/            # Configuration
/var/log/jetson-client.log     # Log file
```

## Performance

### Resource Usage
- **CPU**: < 5% (idle), 10-50% (during inference)
- **Memory**: ~100-500 MB (depends on model size)
- **Disk**: Minimal (models stored separately)
- **Network**: ~1-2 KB every 30s (heartbeat), variable (OTA downloads)

### Tested On
- Jetson AGX Orin 32GB (JetPack 5.1.2)
- Jetson AGX Orin 64GB (JetPack 6.0)

## Security Considerations

1. **API Key**: Set `api_key` in config if backend requires authentication
2. **HTTPS**: Use HTTPS for backend_url in production
3. **Firewall**: Only allow outbound connections to backend server
4. **File Permissions**: Keep config.json readable only by root

## License

Proprietary - Part of ML Platform

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u jetson-client -n 100`
2. Review troubleshooting section above
3. Contact platform administrator

## Version History

- **v1.0.0** - Initial release for AGX Orin 32GB
  - Device registration
  - Heartbeat monitoring
  - OTA updates
  - Multi-framework model support
