from ultralytics import YOLO

# Load pre-trained model
model = YOLO("yolov8n.pt")

# Train
model.train(data="reaction.yaml", epochs=50, imgsz=640, batch=16)
