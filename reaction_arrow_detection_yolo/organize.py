import os
import random
import shutil
from pathlib import Path

# CONFIGURATION
SOURCE_DIR = "data"  # where your images and txt files are now
TARGET_DIR = "reaction_dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% val

# Step 1: Create directory structure
for split in ["train", "val"]:
    os.makedirs(f"{TARGET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/labels/{split}", exist_ok=True)

# Step 2: Collect all images
all_images = [
    f for f in os.listdir(SOURCE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))
]
random.shuffle(all_images)
split_index = int(len(all_images) * TRAIN_SPLIT)

# Step 3: Move images and corresponding label files
for i, img_name in enumerate(all_images):
    base_name = Path(img_name).stem
    label_name = base_name + ".txt"

    split = "train" if i < split_index else "val"

    # Paths
    img_src = os.path.join(SOURCE_DIR, img_name)
    label_src = os.path.join(SOURCE_DIR, label_name)

    img_dst = os.path.join(TARGET_DIR, "images", split, img_name)
    label_dst = os.path.join(TARGET_DIR, "labels", split, label_name)

    # Copy files
    shutil.copy(img_src, img_dst)
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)
    else:
        print(f"⚠️ Warning: No label found for {img_name}")

print("✅ Dataset organized successfully.")
