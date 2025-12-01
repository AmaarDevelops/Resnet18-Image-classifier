import os
import shutil
import random
from PIL import Image

# -------------------- Paths --------------------
source_dir = "PetImages"  # Folder you already have
target_dir = "data"       # Will be created

categories = ["Cat", "Dog"]

# Create target folder structure
for split in ["train", "val", "test"]:
    for category in categories:
        os.makedirs(os.path.join(target_dir, split, category.lower()), exist_ok=True)

# -------------------- Settings --------------------
train_split = 0.8
val_split = 0.1
test_split = 0.1

# -------------------- Process images --------------------
for category in categories:
    print(f"Processing {category} images...")

    src_path = os.path.join(source_dir, category)
    all_images = os.listdir(src_path)

    # Filter only valid images
    valid_images = []
    for img_name in all_images:
        img_path = os.path.join(src_path, img_name)

        # Skip weird system files
        if img_name.startswith(".") or not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Check if image is corrupted
        try:
            img = Image.open(img_path)
            img.verify()
            valid_images.append(img_name)
        except:
            print("Corrupted image removed:", img_path)

    print(f"Valid images: {len(valid_images)}")

    # Shuffle
    random.shuffle(valid_images)

    # Split
    num_total = len(valid_images)
    num_train = int(train_split * num_total)
    num_val = int(val_split * num_total)

    train_files = valid_images[:num_train]
    val_files = valid_images[num_train:num_train + num_val]
    test_files = valid_images[num_train + num_val:]

    # Copy files
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, file_list in splits.items():
        for file in file_list:
            src = os.path.join(src_path, file)
            dest = os.path.join(target_dir, split_name, category.lower(), file)
            shutil.copy2(src, dest)

    print(f"Finished {category}!")

print("Dataset successfully organized!")
