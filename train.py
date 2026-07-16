from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  

# Train the model (Resume training from last checkpoint)
train_results = model.train(
    data="datasets/data.yaml",
    epochs=50,  # เพิ่มรอบการเทรนเป็น 30
    imgsz=640,
    device="cuda",
    batch=6,
    workers=0,
    project="runs/train",
    name="yolov8m_custom_optimized",
)


# Validate the trained model
metrics = model.val()
