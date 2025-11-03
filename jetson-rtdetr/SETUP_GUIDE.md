# RT-DETR Jetson Setup Guide

Complete setup guide for running RT-DETR inference on NVIDIA Jetson devices.

---

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA Jetson device (Nano, TX2, Xavier NX, Orin, or AGX)
- MicroSD card (64GB+ recommended)
- Power supply appropriate for your Jetson model
- Camera or video source (optional for real-time inference)

### Software Requirements
- Ubuntu 18.04/20.04 (included in JetPack)
- Python 3.6+
- CUDA 10.2+ (included in JetPack)
- cuDNN 8.0+ (included in JetPack)
- TensorRT 8.0+ (included in JetPack)

---

## 🚀 Step-by-Step Setup

### Step 1: Install JetPack

1. **Download SD Card Image**
   - Visit: https://developer.nvidia.com/embedded/jetpack
   - Download JetPack for your Jetson model
   - JetPack includes: Ubuntu, CUDA, cuDNN, TensorRT, and more

2. **Flash SD Card**
   ```bash
   # On your host machine (Linux)
   # Download and extract the image
   unzip jetpack-image.zip

   # Flash to SD card (replace /dev/sdX with your SD card)
   sudo dd if=jetpack-image.img of=/dev/sdX bs=1M status=progress
   sudo sync
   ```

3. **Boot Jetson**
   - Insert SD card into Jetson
   - Connect monitor, keyboard, mouse
   - Power on and complete initial setup

4. **Verify Installation**
   ```bash
   # Check CUDA
   nvcc --version

   # Check TensorRT
   dpkg -l | grep TensorRT

   # Check cuDNN
   dpkg -l | grep cudnn
   ```

### Step 2: System Configuration

1. **Update System**
   ```bash
   sudo apt-get update
   sudo apt-get upgrade -y
   ```

2. **Install System Dependencies**
   ```bash
   # Build tools
   sudo apt-get install -y build-essential cmake git

   # Python development
   sudo apt-get install -y python3-dev python3-pip

   # OpenCV dependencies
   sudo apt-get install -y libopencv-dev python3-opencv

   # Other utilities
   sudo apt-get install -y nano htop
   ```

3. **Configure Swap Space** (Important for Jetson Nano)
   ```bash
   # Create 4GB swap file
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

   # Make permanent
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

   # Verify
   free -h
   ```

4. **Set Performance Mode**
   ```bash
   # Set to maximum performance mode
   sudo nvpmodel -m 0

   # Enable jetson_clocks for max performance
   sudo jetson_clocks
   ```

### Step 3: Install Python Dependencies

1. **Upgrade pip**
   ```bash
   python3 -m pip install --upgrade pip
   ```

2. **Install NumPy**
   ```bash
   # Use system NumPy for better compatibility
   sudo apt-get install -y python3-numpy

   # Or install via pip
   pip3 install numpy
   ```

3. **Install PyCUDA**
   ```bash
   # Install dependencies
   sudo apt-get install -y python3-dev

   # Install PyCUDA
   pip3 install pycuda

   # Verify
   python3 -c "import pycuda.driver as cuda; print('PyCUDA OK')"
   ```

4. **Verify TensorRT Python Bindings**
   ```bash
   # Check TensorRT Python API
   python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

   # If not available, install
   pip3 install tensorrt
   ```

5. **Install Other Dependencies**
   ```bash
   pip3 install opencv-python Pillow tqdm pytest
   ```

### Step 4: Clone and Setup Project

1. **Clone Repository**
   ```bash
   cd ~
   git clone <repository-url>
   cd jetson-rtdetr
   ```

2. **Install Requirements**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Add to Python Path** (Optional)
   ```bash
   # Add to ~/.bashrc
   echo 'export PYTHONPATH=$PYTHONPATH:~/jetson-rtdetr' >> ~/.bashrc
   source ~/.bashrc
   ```

### Step 5: Prepare RT-DETR Model

#### Option 1: Export from PyTorch (On Host Machine)

```python
# On your powerful PC/workstation
import torch
import onnx

# Load RT-DETR model
model = torch.hub.load('lyuwenyu/RT-DETR', 'rtdetr_r50vd')
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "rtdetr_r50.onnx",
    input_names=['images'],
    output_names=['boxes', 'scores', 'labels'],
    dynamic_axes={
        'images': {0: 'batch'},
        'boxes': {0: 'batch'},
        'scores': {0: 'batch'},
        'labels': {0: 'batch'}
    },
    opset_version=11
)

# Verify ONNX model
onnx_model = onnx.load("rtdetr_r50.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

#### Option 2: Use Pre-exported Model

```bash
# Download pre-exported ONNX model
wget https://example.com/rtdetr_r50.onnx

# Or copy from host machine
scp your_pc:~/rtdetr_r50.onnx ~/jetson-rtdetr/
```

### Step 6: Build TensorRT Engine

```bash
cd ~/jetson-rtdetr

# Build FP16 engine (recommended)
python3 models/rtdetr_tensorrt.py \
    --onnx rtdetr_r50.onnx \
    --output rtdetr_r50.fp16.engine \
    --precision fp16 \
    --batch-size 1 \
    --verbose

# This will take 5-15 minutes depending on your Jetson model
```

**Expected Output:**
```
Initialized TensorRT builder
  ONNX model: rtdetr_r50.onnx
  Engine path: rtdetr_r50.fp16.engine
  Precision: fp16
  Max batch size: 1

Parsing ONNX model: rtdetr_r50.onnx
FP16 mode enabled
Building TensorRT engine (this may take several minutes)...
Saving engine to: rtdetr_r50.fp16.engine
✓ TensorRT engine built successfully!
```

### Step 7: Test Installation

1. **Prepare Test Image**
   ```bash
   # Download sample image
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O test.jpg
   ```

2. **Run Basic Inference**
   ```bash
   python3 examples/basic_inference.py \
       --model rtdetr_r50.onnx \
       --image test.jpg \
       --output result.jpg \
       --precision fp16 \
       --conf 0.25
   ```

   **Expected Output:**
   ```
   Initializing RT-DETR Inference Engine...
     Model: rtdetr_r50.onnx
     Precision: fp16
     Input size: 640x640

   Loading image: test.jpg
     Image size: 768x576

   Running inference...

   Results:
     Found 3 detections
       [1] Detection(class=dog, conf=0.982, bbox=[123.4, 220.1, 456.7, 512.3])
       [2] Detection(class=bicycle, conf=0.874, bbox=[78.2, 145.6, 598.9, 423.1])
       [3] Detection(class=car, conf=0.756, bbox=[...]
   ```

3. **Run Benchmark**
   ```bash
   python3 examples/benchmark_example.py \
       --model rtdetr_r50.onnx \
       --image test.jpg \
       --iterations 100 \
       --precision fp16
   ```

4. **Test Video Inference** (With Camera)
   ```bash
   # USB camera (usually /dev/video0)
   python3 examples/video_inference.py \
       --model rtdetr_r50.onnx \
       --input 0 \
       --precision fp16

   # Press 'q' to quit
   ```

---

## 🔧 Optimization Tips

### 1. Power Management

```bash
# Check current power mode
sudo nvpmodel -q

# Set to max performance (mode 0)
sudo nvpmodel -m 0

# Enable max clocks
sudo jetson_clocks

# Make permanent (add to /etc/rc.local)
echo 'nvpmodel -m 0' | sudo tee -a /etc/rc.local
echo 'jetson_clocks' | sudo tee -a /etc/rc.local
```

### 2. Monitor Performance

```bash
# Real-time stats
sudo tegrastats

# Alternative with sudo jtop
sudo pip3 install jetson-stats
sudo jtop
```

### 3. Temperature Management

```bash
# Check temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# If overheating, consider:
# - Add heatsink/fan
# - Reduce power mode
# - Lower batch size
# - Reduce input resolution
```

### 4. Memory Optimization

```bash
# Monitor memory
free -h

# Clear cache
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# If OOM errors, increase swap:
sudo swapoff /swapfile
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 🐛 Troubleshooting

### Issue: TensorRT Not Found

```bash
# Verify TensorRT installation
dpkg -l | grep TensorRT

# If missing, install via JetPack
sudo apt-get install nvidia-jetpack

# Or specific TensorRT version
sudo apt-get install tensorrt
```

### Issue: PyCUDA Installation Fails

```bash
# Install build dependencies
sudo apt-get install -y python3-dev build-essential

# Try pip install with verbose output
pip3 install pycuda -v

# If still fails, install from source
git clone https://github.com/inducer/pycuda.git
cd pycuda
python3 configure.py --cuda-root=/usr/local/cuda
sudo python3 setup.py install
```

### Issue: ONNX Parser Errors

```bash
# Check ONNX opset version
python3 -c "import onnx; print(onnx.defs.onnx_opset_version())"

# Re-export with compatible opset
# In PyTorch export, use: opset_version=11
```

### Issue: Out of Memory During Engine Build

```bash
# Reduce workspace size
python3 models/rtdetr_tensorrt.py \
    --onnx model.onnx \
    --precision fp16 \
    --workspace-size 512  # MB (default is 1024)
```

### Issue: Low FPS

**Checklist:**
1. ✅ Power mode set to max: `sudo nvpmodel -m 0`
2. ✅ Jetson clocks enabled: `sudo jetson_clocks`
3. ✅ Using FP16 precision
4. ✅ TensorRT engine built (not ONNX)
5. ✅ Appropriate batch size
6. ✅ Thermal throttling not occurring

### Issue: Segmentation Fault

```bash
# Update CUDA/TensorRT
sudo apt-get update
sudo apt-get upgrade nvidia-jetpack

# Rebuild engine
rm *.engine
python3 models/rtdetr_tensorrt.py --onnx model.onnx
```

---

## 📊 Expected Performance

### Jetson Nano (4GB)
- **FP16**: 3-5 FPS (640x640)
- **Input 320x320**: 8-10 FPS
- **Batch processing**: Limited by memory

### Jetson Xavier NX
- **FP16**: 15-20 FPS (640x640)
- **Batch 4**: 35-45 FPS
- **Multi-stream**: 2-3 streams at 10 FPS each

### Jetson AGX Orin
- **FP16**: 40-60 FPS (640x640)
- **Batch 8**: 100-120 FPS
- **Multi-stream**: 4-6 streams at 15 FPS each

*Performance varies based on model complexity and scene content.*

---

## 🚀 Production Deployment

### 1. Create Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/rtdetr-inference.service
```

```ini
[Unit]
Description=RT-DETR Inference Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/jetson-rtdetr
ExecStart=/usr/bin/python3 examples/video_inference.py --model rtdetr_r50.onnx --input 0 --no-display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rtdetr-inference.service
sudo systemctl start rtdetr-inference.service

# Check status
sudo systemctl status rtdetr-inference.service
```

### 2. Auto-start on Boot

```bash
# Add to /etc/rc.local
sudo nano /etc/rc.local
```

```bash
#!/bin/bash
# Set max performance
nvpmodel -m 0
jetson_clocks

# Start inference
cd /home/jetson/jetson-rtdetr
python3 examples/video_inference.py --model rtdetr_r50.onnx --input 0 --no-display &

exit 0
```

```bash
# Make executable
sudo chmod +x /etc/rc.local
```

---

## 📚 Additional Resources

- **NVIDIA Jetson**: https://developer.nvidia.com/embedded/jetson
- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **RT-DETR Paper**: https://arxiv.org/abs/2304.08069
- **Community Forum**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

---

## ✅ Quick Setup Checklist

- [ ] JetPack installed
- [ ] System updated
- [ ] Swap space configured (4GB+)
- [ ] Power mode set to max
- [ ] Python dependencies installed
- [ ] TensorRT verified
- [ ] PyCUDA installed
- [ ] Repository cloned
- [ ] ONNX model available
- [ ] TensorRT engine built
- [ ] Test inference successful
- [ ] Benchmark completed

---

For questions or issues, please refer to the main README or open an issue on GitHub.
