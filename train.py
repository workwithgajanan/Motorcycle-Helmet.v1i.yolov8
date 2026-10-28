from ultralytics import YOLO

# 1. Base YOLOv8 model load kara
model = YOLO('yolov8n.pt')

# 2. Custom Dataset var train kara
model.train(
    data='data.yaml',
    epochs=15,
    imgsz=640,
    batch=16,
    name='helmet_model'
)