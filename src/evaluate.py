import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm

# ==========================
# Transform Config 
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# Grad-CAM
# ==========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_a = target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()

    def __call__(self, x, class_idx):
        logits = self.model(x)
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients
        weights = dA.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, logits.detach()

def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.45):
    H, W = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1 - alpha, 0)
    return overlay, heatmap_resized

# ==========================
# Build DataLoader
# ==========================
def build_loader(data_dir: Path):
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    return test_loader, test_ds.class_to_idx

# ==========================
# Load the model
# ==========================
def load_model(model_path, device, num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# ==========================
# eval with Grad-CAM
# ==========================
def evaluate(model, loader, class_to_idx, device, gradcam_dir):
    os.makedirs(gradcam_dir, exist_ok=True)
    softmax = nn.Softmax(dim=1)
    all_probs, all_preds, all_targets = [], [], []

    cam = GradCAM(model, model.layer4)

    for i, (imgs, labels) in enumerate(tqdm(loader, desc="Evaluating")):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        probs = softmax(logits)
        preds = torch.argmax(logits, dim=1)

        all_probs.append(probs[:, 1].detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())

        # Grad-CAM for every image 
        cam_map, _ = cam(imgs, preds.item())

        img_path = loader.dataset.samples[i][0]
        orig_bgr = cv2.imread(img_path)
        overlay, _ = overlay_heatmap_on_image(orig_bgr, cam_map)
        save_path = gradcam_dir / Path(img_path).name
        cv2.imwrite(str(save_path), overlay)


    cam.close()

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, digits=4)
    try:
        auc = roc_auc_score(targets, probs)
    except Exception:
        auc = float("nan")
    acc = (preds == targets).mean()

    return {"acc": acc, "auc": auc, "cm": cm, "report": report}

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../runs/pneumonia_resnet18/best_resnet18.pt")
    parser.add_argument("--data_dir", type=str, default="../data/processed/chest_xray")
    parser.add_argument("--gradcam_dir", type=str, default="../runs/gradcam_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, class_to_idx = build_loader(Path(args.data_dir))
    model = load_model(args.model, device)
    metrics = evaluate(model, test_loader, class_to_idx, device, Path(args.gradcam_dir))

    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"AUC     : {metrics['auc']:.4f}")
    print("Confusion Matrix:\n", metrics['cm'])
    print("\nClassification Report:\n", metrics['report'])
