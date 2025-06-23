import os

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# CONFIG
image_dir = "inference_images"  # path to your image folder
output_dir = "inference_outputs"  # where to save annotated results
os.makedirs(output_dir, exist_ok=True)

# Load trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Get all image files in the folder
image_files = [
    f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Loop over each image
for image_file in sorted(image_files):
    image_path = os.path.join(image_dir, image_file)

    # Run inference
    results = model(image_path, conf=0.1)

    # Save result image
    output_path = os.path.join(output_dir, f"pred_{image_file}")
    results[0].save(filename=output_path)

    # Display
    img = cv2.imread(output_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Prediction: {image_file}")
    plt.show()

    # Print detected boxes
    print(f"\n🔍 Inference for {image_file}:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"  - Class: {cls_id}, Confidence: {conf:.2f}, BBox: {xyxy}")
