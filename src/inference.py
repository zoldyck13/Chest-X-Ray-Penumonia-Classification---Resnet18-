import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ==========================
# Transforms for Inference
# ==========================
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ==========================
# Load the model
# ==========================
def load_model(model_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2) 
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

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

    def __call__(self, x, class_idx):
        logits = self.model(x)
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward()
        A = self.activations
        dA = self.gradients
        weights = dA.mean(dim=(2,3), keepdim=True)
        cam = (weights * A).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, logits.detach()

def overlay_heatmap(orig_bgr, heatmap, alpha=0.45):
    H, W = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W,H))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1-alpha, 0)
    return overlay, heatmap_resized

# ==========================
# Main inference
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .pt")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image for inference")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out", type=str, default="../gradcam_result.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)

    # Load and preprocess image
    pil = Image.open(args.image).convert("RGB")
    x = get_transform(args.img_size)(pil).unsqueeze(0).to(device)

    orig_bgr = cv2.imread(args.image)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    # Grad-CAM
    cam = GradCAM(model, model.layer4)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    heatmap, _ = cam(x, pred)
    cam.hook_a.remove()
    cam.hook_g.remove()

    overlay, heatmap_resized = overlay_heatmap(orig_bgr, heatmap)

    class_map = {0:"NORMAL", 1:"PNEUMONIA"}
    pred_name = class_map[pred]

    print(f"Prediction: {pred_name}")
    print(f"Prob NORMAL: {probs[0]:.4f}, Prob PNEUMONIA: {probs[1]:.4f}")

    # Save and show results
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(pil)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(overlay_rgb)
    plt.title(f"Grad-CAM ({pred_name})")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.title("Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved Grad-CAM result to {args.out}")

if __name__ == "__main__":
    main()
