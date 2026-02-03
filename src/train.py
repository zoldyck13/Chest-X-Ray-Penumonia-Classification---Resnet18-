# train.py
import os
import copy
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

from data.loaders import build_loaders
from models.resnet import build_resnet18
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from dataclasses import dataclass

# ==========================
#  Training config 
# ==========================
@dataclass
class Config:
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    out_dir: str = "../runs/pneumonia_resnet18"
    seed: int = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================
# evaluate function 
# ==========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    softmax = nn.Softmax(dim=1)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = softmax(logits)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    acc = (preds == targets).mean()
    try:
        auc = roc_auc_score(targets, probs)
    except:
        auc = float("nan")

    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, digits=4)
    return {"acc": acc, "auc": auc, "cm": cm, "report": report}

# ==========================
# Loop for every epoch
# ==========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total

# ==========================
# Main
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", type=str, default="runs/pneumonia_resnet18")
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_dir=args.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ========== Load data ==========
    train_loader, val_loader, test_loader, class_to_idx = build_loaders(batch_size=cfg.batch_size)
    print("Class mapping:", class_to_idx)

    # ========== Build model ==========
    model = build_resnet18(num_classes=len(class_to_idx), pretrained=True).to(device)

    # ========== Class weights ==========
    train_targets = [y for _, y in train_loader.dataset.samples]
    num0 = sum(1 for t in train_targets if t == 0)
    num1 = sum(1 for t in train_targets if t == 1)
    w0 = (num0 + num1) / (2.0 * max(num0, 1))
    w1 = (num0 + num1) / (2.0 * max(num1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_auc = -1.0
    best_model = copy.deepcopy(model.state_dict())

    # ========== Training loop ==========
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"[Epoch {epoch}/{cfg.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f} "
              f"time={dt:.1f}s")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(cfg.out_dir, "best_resnet18.pt"))
            print("✔ Saved best model")

    # ========== Test ==========
    model.load_state_dict(best_model)
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    print(f"Test ACC: {test_metrics['acc']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print("Confusion Matrix:\n", test_metrics["cm"])
    print("\nClassification Report:\n", test_metrics["report"])

if __name__ == "__main__":
    main()
