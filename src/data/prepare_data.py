import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  

# ==========================
# Paths
# ==========================
RAW_DIR = Path("../../data/raw/chest_xray")
PROCESSED_DIR = Path("../../data/processed/chest_xray")
SPLITS = ["train", "val", "test"]

# Split ratios
SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# Transformations for training images only
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

eval_transform = transforms.Compose([
    # No transformations for evaluation
])

# ==========================
# Function to save image with optional transform
# ==========================
def save_image(img_path, save_path, transform=None):
    img = Image.open(img_path).convert("RGB")
    if transform:
        img = transform(img)
    img.save(save_path)

# ==========================
# Main function
# ==========================
def split_and_process_data():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for cls in os.listdir(RAW_DIR):
        cls_path = RAW_DIR / cls
        if not cls_path.is_dir():
            continue

        # Read all images in this class
        images = list(cls_path.glob("*"))
        random.shuffle(images)

        # Compute number of images for each split
        n = len(images)
        n_train = int(n * SPLIT_RATIO["train"])
        n_val   = int(n * SPLIT_RATIO["val"])
        n_test  = n - n_train - n_val

        splits_dict = {
            "train": images[:n_train],
            "val":   images[n_train:n_train+n_val],
            "test":  images[n_train+n_val:]
        }

        # Save images into processed/ with progress bar
        for split, split_images in splits_dict.items():
            dest_dir = PROCESSED_DIR / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing class '{cls}' for split '{split}' ({len(split_images)} images)...")
            for img_path in tqdm(split_images, desc=f"{split} images", ncols=100):
                if split == "train":
                    save_image(img_path, dest_dir / img_path.name, transform=train_transform)
                else:
                    save_image(img_path, dest_dir / img_path.name, transform=eval_transform)

        print(f"Finished preparing class '{cls}': "
              f"{n_train} train, {n_val} val, {n_test} test images\n")

if __name__ == "__main__":
    split_and_process_data()
    print(" Data preparation completed in /processed")
