import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================
# Config Transforms
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# Config DataLoaders
# ==========================
DATA_DIR = Path("../data/processed/chest_xray")

def build_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=eval_transform)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print("DataLoaders ready:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Classes: {train_ds.classes}")

    return train_loader, val_loader, test_loader, train_ds.class_to_idx

## Testing the result 

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_to_idx = build_loaders()
