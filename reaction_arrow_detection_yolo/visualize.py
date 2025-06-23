import os

import cv2
import matplotlib.pyplot as plt

# CONFIG
images_dir = "/home/rahul/reaction_region_yolo/datasets/reaction_dataset/images/train"
labels_dir = "/home/rahul/reaction_region_yolo/datasets/reaction_dataset/labels/train"
class_names = ["reaction_region"]  # Update if you have more classes


def visualize_yolo_annotation(image_path, label_path, class_names=None):
    print(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        print(f"Label file missing for {image_path}")
        return

    with open(label_path, "r") as f:
        boxes = f.readlines()

    for box in boxes:
        parts = box.strip().split()
        if len(parts) != 5:
            continue
        cls_id, x_center, y_center, bw, bh = map(float, parts)
        cls_id = int(cls_id)
        print(f"Class ID: {cls_id}")
        # Normalize coordinates

        # Convert normalized to absolute coords
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        label = class_names[cls_id] if class_names else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(image_path))
    plt.show()


# Loop through all images
for filename in sorted(os.listdir(images_dir)):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
        visualize_yolo_annotation(image_path, label_path, class_names)
