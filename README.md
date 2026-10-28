# Real-Time Deep Learning System for Helmet Detection 🪖🔍

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision solution engineered to detect motorcycle riders and enforce safety compliance by identifying whether individuals are wearing helmets in real-time. Powered by **YOLOv8** transfer learning, **PyTorch**, and **OpenCV**, this model processes live video feeds and image streams with high precision and low latency.

---

## 📌 Features

- **Real-Time Object Detection**: High-FPS inference on live webcam feeds, pre-recorded video files, and image batches.
- **Custom Fine-Tuned Model**: Trained on custom annotated datasets using YOLOv8 transfer learning.
- **High Performance**: Optimized bounding box regression and classification achieving **mAP50 > 0.82**.
- **Automated Visual Bounding Boxes**: Displays bounding boxes with custom class labels (`Helmet`, `No Helmet`) and confidence scores.
- **Easy Deployment**: Lightweight inference script utilizing PyTorch weights (`best.pt`) for quick setup across local or edge devices.

---

## 🛠️ System Requirements & Dependencies

### Prerequisites
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: Python `3.8`, `3.9`, `3.10`, or `3.11`
- **Hardware**: 
  - *Minimum*: Quad-core CPU, 8 GB RAM
  - *Recommended*: NVIDIA GPU with CUDA support for accelerated training and real-time high-FPS inference.

### Python Packages (`requirements.txt`)
```text
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.5.0
PyYAML>=6.0
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/helmet-detection-yolov8.git
   cd helmet-detection-yolov8
   ```

2. **Create a Virtual Environment** *(Recommended)*:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Project Structure

```text
helmet-detection-yolov8/
│
├── data/
│   ├── train/            # Training images and YOLO format labels
│   ├── val/              # Validation images and labels
│   └── data.yaml         # Dataset configuration file
│
├── weights/
│   └── best.pt           # Trained PyTorch model weights
│
├── src/
│   ├── train.py          # Model training pipeline
│   ├── detect_image.py   # Image inference script
│   └── detect_video.py   # Live camera / Video inference script
│
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 📊 Dataset Details

The model was trained on a custom annotated dataset structured in YOLO format:
- **Total Images**: 600+ high-resolution annotated images.
- **Classes**:
  1. `Helmet` — Riders wearing protective headgear.
  2. `No Helmet` — Riders without headgear.
- **Annotations**: Bounding box bounding coordinates generated and verified using **Roboflow**.

### Dataset Config (`data.yaml`)
```yaml
path: ../data
train: train/images
val: val/images

nc: 2
names: ['Helmet', 'No Helmet']
```

---

## 🚀 Model Training

To retrain or fine-tune the model on your dataset:

```bash
python src/train.py --data data/data.yaml --epochs 15 --imgsz 640 --batch 16
```

### Training Parameters:
- **Base Architecture**: `yolov8n.pt` (YOLOv8 Nano for speed) / `yolov8s.pt` (YOLOv8 Small for higher accuracy)
- **Epochs**: 15
- **Optimizer**: Auto (AdamW/SGD)
- **Input Resolution**: 640x640 pixels

---

## 💻 Usage Instructions

### 1. Real-Time Detection via Live Webcam
Run live detection using your connected webcam:
```bash
python src/detect_video.py --weights weights/best.pt --source 0
```

### 2. Inference on Video File
Run inference on a stored video:
```bash
python src/detect_video.py --weights weights/best.pt --source path/to/video.mp4
```

### 3. Inference on Single/Batch Images
Process single images or a directory of images:
```bash
python src/detect_image.py --weights weights/best.pt --source path/to/image.jpg
```

---

## 📈 Evaluation & Performance Metrics

The fine-tuned model achieved robust performance across metrics evaluated on the validation set over 15 epochs:

| Metric | Score |
| :--- | :--- |
| **Precision (P)** | `~0.85` |
| **Recall (R)** | `~0.79` |
| **mAP@50** | **`> 0.82`** |
| **mAP@50-95** | `~0.58` |
| **Inference Speed** | ~12ms per frame (GPU) / ~45ms per frame (CPU) |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).