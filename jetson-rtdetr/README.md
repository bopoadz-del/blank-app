# RT-DETR TensorRT Inference for NVIDIA Jetson

High-performance RT-DETR (Real-Time Detection Transformer) inference system optimized for NVIDIA Jetson devices using TensorRT.

## 🚀 Features

- ✅ **TensorRT Optimization**: Automatic ONNX to TensorRT conversion with FP32/FP16/INT8 support
- ✅ **Optimized Preprocessing**: Fast image/video preprocessing with letterbox padding
- ✅ **Smart Postprocessing**: Non-Maximum Suppression (NMS) with configurable thresholds
- ✅ **Batch Processing**: Efficient batch inference for higher throughput
- ✅ **Multi-Stream Support**: Concurrent processing of multiple video streams
- ✅ **Model Caching**: Automatic caching of built TensorRT engines
- ✅ **FPS Benchmarking**: Comprehensive performance measurement tools
- ✅ **Real-Time Inference**: Optimized for real-time video processing
- ✅ **Error Handling**: Robust error handling and logging
- ✅ **Easy Integration**: Simple API for quick integration

## 📋 Requirements

### Hardware
- NVIDIA Jetson (Nano, TX2, Xavier, Orin, or newer)
- Jetson JetPack 4.6+ or 5.0+

### Software
```bash
# Core dependencies
python >= 3.6
numpy
opencv-python
tensorrt >= 8.0
pycuda

# Optional for ONNX export
torch
torchvision
onnx
```

## 🔧 Installation

### 1. Install JetPack
Follow NVIDIA's guide to install JetPack on your Jetson device:
https://developer.nvidia.com/embedded/jetpack

### 2. Install Python Dependencies

```bash
# Install OpenCV
sudo apt-get update
sudo apt-get install python3-opencv

# Install Python packages
pip3 install numpy pycuda

# TensorRT is included with JetPack
# Verify installation:
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### 3. Clone Repository

```bash
git clone <repository-url>
cd jetson-rtdetr
```

### 4. Prepare RT-DETR Model

Export your RT-DETR model to ONNX format:

```python
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
    dynamic_axes={'images': {0: 'batch'}}
)
```

### 5. Build TensorRT Engine

```bash
python3 models/rtdetr_tensorrt.py \
    --onnx rtdetr_r50.onnx \
    --output rtdetr_r50.fp16.engine \
    --precision fp16 \
    --batch-size 1
```

## 📖 Usage

### Basic Image Inference

```python
from inference.inference_engine import InferenceEngine
import cv2

# Create inference engine
engine = InferenceEngine(
    model_path="rtdetr_r50.onnx",
    precision="fp16",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Load image
image = cv2.imread("test.jpg")

# Run inference
detections = engine.infer(image)

# Print results
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f} at {det.bbox}")
```

### Video Inference

```python
from preprocessing.image_preprocessor import VideoPreprocessor
from inference.inference_engine import InferenceEngine
from postprocessing.nms_filter import DetectionVisualizer
import cv2

# Create engine
engine = InferenceEngine(model_path="rtdetr_r50.onnx")

# Open video
video = VideoPreprocessor("input.mp4", engine.preprocessor)
visualizer = DetectionVisualizer()

# Process frames
for batch, originals, metadata in video:
    detections = engine.infer_batch([orig for orig in originals])

    for orig, dets in zip(originals, detections):
        output = visualizer.draw_detections(orig, dets)
        cv2.imshow("Detections", output)
        cv2.waitKey(1)

video.release()
```

### Batch Processing

```python
from inference.inference_engine import BatchProcessor

# Create batch processor
processor = BatchProcessor(
    engine=engine,
    batch_size=4,
    max_wait_time=0.1
)

# Process frames with automatic batching
def on_result(detections):
    print(f"Detected {len(detections)} objects")

for frame in frames:
    processor.add_frame(frame, callback=on_result)
```

### Multi-Stream Inference

```python
from inference.inference_engine import MultiStreamInference

# Create multi-stream processor
multi_stream = MultiStreamInference(
    engine=engine,
    num_workers=2,
    queue_size=10
)

# Start processing
multi_stream.start()

# Add streams
multi_stream.add_stream("camera_1")
multi_stream.add_stream("camera_2")

# Process frames
multi_stream.put_frame("camera_1", frame1)
multi_stream.put_frame("camera_2", frame2)

# Get results
detections1, _ = multi_stream.get_result("camera_1")
detections2, _ = multi_stream.get_result("camera_2")

# Cleanup
multi_stream.stop()
```

## 🎯 Examples

### Run Basic Inference

```bash
python3 examples/basic_inference.py \
    --model rtdetr_r50.onnx \
    --image test.jpg \
    --output output.jpg \
    --conf 0.25 \
    --iou 0.45
```

### Run Video Inference

```bash
# Process video file
python3 examples/video_inference.py \
    --model rtdetr_r50.onnx \
    --input video.mp4 \
    --output output.mp4

# Use camera
python3 examples/video_inference.py \
    --model rtdetr_r50.onnx \
    --input 0
```

### Run Benchmark

```bash
python3 examples/benchmark_example.py \
    --model rtdetr_r50.onnx \
    --image test.jpg \
    --iterations 100 \
    --precision fp16
```

## 📊 Performance

Typical performance on different Jetson devices (RT-DETR R50, 640x640, FP16):

| Device | FPS (Single) | FPS (Batch 4) | Latency (ms) |
|--------|--------------|---------------|--------------|
| Jetson Nano | 3-5 | 8-10 | 200-300 |
| Jetson TX2 | 8-12 | 20-25 | 80-120 |
| Jetson Xavier NX | 15-20 | 35-45 | 50-70 |
| Jetson Orin Nano | 20-30 | 50-60 | 30-50 |
| Jetson AGX Orin | 40-60 | 100-120 | 15-25 |

*Note: Performance varies based on model size, input resolution, and system configuration.*

## 🔧 Optimization Tips

### 1. Use FP16 Precision
```python
engine = InferenceEngine(
    model_path="model.onnx",
    precision="fp16"  # 2x faster than FP32
)
```

### 2. Increase Batch Size
```python
# Process multiple frames together
detections = engine.infer_batch([frame1, frame2, frame3, frame4])
```

### 3. Enable Model Caching
```python
engine = InferenceEngine(
    model_path="model.onnx",
    enable_cache=True  # Speeds up loading
)
```

### 4. Optimize Input Size
```python
# Smaller input = faster inference
engine = InferenceEngine(
    model_path="model.onnx",
    input_size=(320, 320)  # vs (640, 640)
)
```

### 5. Adjust Thresholds
```python
# Higher thresholds = fewer detections = faster postprocessing
engine = InferenceEngine(
    model_path="model.onnx",
    conf_threshold=0.5,  # vs 0.25
    iou_threshold=0.5    # vs 0.45
)
```

## 📁 Project Structure

```
jetson-rtdetr/
├── models/
│   └── rtdetr_tensorrt.py       # TensorRT engine builder
├── preprocessing/
│   └── image_preprocessor.py    # Image/video preprocessing
├── postprocessing/
│   └── nms_filter.py            # NMS and filtering
├── inference/
│   └── inference_engine.py      # Main inference engine
├── benchmarks/
│   └── fps_benchmark.py         # Performance benchmarking
├── examples/
│   ├── basic_inference.py       # Basic example
│   ├── video_inference.py       # Video processing
│   └── benchmark_example.py     # Benchmark example
├── tests/
│   └── test_inference.py        # Unit tests
└── README.md
```

## 🧪 Testing

Run unit tests:

```bash
python3 -m pytest tests/
```

Test individual components:

```bash
# Test TensorRT builder
python3 models/rtdetr_tensorrt.py --onnx model.onnx --verbose

# Test preprocessing
python3 preprocessing/image_preprocessor.py --image test.jpg --output preprocessed.jpg

# Test postprocessing
python3 postprocessing/nms_filter.py --conf 0.25 --iou 0.45
```

## 🐛 Troubleshooting

### TensorRT Build Fails

**Issue**: Engine building fails with CUDA/cuDNN errors

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Check TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Verify cuDNN
ls /usr/lib/aarch64-linux-gnu/libcudnn*

# If missing, reinstall JetPack
sudo apt-get install nvidia-jetpack
```

### Out of Memory

**Issue**: OOM error during inference

**Solution**:
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Or reduce batch size
engine = InferenceEngine(model_path="model.onnx", max_batch_size=1)
```

### Low FPS

**Issue**: Inference is slower than expected

**Solution**:
1. Enable FP16: `precision="fp16"`
2. Increase batch size for throughput
3. Reduce input resolution
4. Check power mode: `sudo nvpmodel -m 0` (max performance)
5. Enable jetson_clocks: `sudo jetson_clocks`

### Model Loading Slow

**Issue**: Model takes long to load

**Solution**:
```python
# Use model caching
engine = InferenceEngine(
    model_path="model.onnx",
    enable_cache=True  # Cache TensorRT engine
)
```

## 📚 API Reference

### InferenceEngine

Main inference engine class.

```python
InferenceEngine(
    model_path: str,          # Path to ONNX model
    precision: str = "fp16",  # Precision mode
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: Tuple = (640, 640),
    class_names: List[str] = None,
    enable_cache: bool = True,
    warmup_runs: int = 10
)
```

**Methods**:
- `infer(image) -> List[Detection]`: Single image inference
- `infer_batch(images) -> List[List[Detection]]`: Batch inference

### Detection

Detection result class.

```python
Detection(
    bbox: Tuple[float, float, float, float],  # x1, y1, x2, y2
    confidence: float,
    class_id: int,
    class_name: str
)
```

**Properties**:
- `x1, y1, x2, y2`: Bounding box coordinates
- `width, height`: Box dimensions
- `center`: Center coordinates
- `area`: Box area

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- RT-DETR: https://github.com/lyuwenyu/RT-DETR
- TensorRT: https://developer.nvidia.com/tensorrt
- NVIDIA Jetson: https://developer.nvidia.com/embedded-computing

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is optimized for NVIDIA Jetson devices. For other platforms, consider using ONNX Runtime or native PyTorch inference.
