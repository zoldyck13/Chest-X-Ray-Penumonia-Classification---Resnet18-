import torch
from models.resnet import build_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model 
model = build_resnet18(num_classes=2, pretrained=False)
model.load_state_dict(torch.load("runs/pneumonia_resnet18/best_resnet18.pt", map_location=device))
model.to(device)
model.eval()

# Save the final model
torch.save(model.state_dict(), "model_final.pt")
print("Model exported successfully as model_final.pt")
