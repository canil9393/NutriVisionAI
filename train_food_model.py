from ultralytics import YOLO

# Load YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train on IndianFood10
results = model.train(
    data=r"D:\nutrovision datasets\indianfood10.v1i.yolov11\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="nutrivision_food",
    patience=10
)

print("Training completed!")