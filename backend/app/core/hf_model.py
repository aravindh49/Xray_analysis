import torch
import torch.nn as nn
from transformers import AutoModel
import os

NUM_CLASSES = 4
MODEL_NAME = "microsoft/rad-dino"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RadDinoClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Rad-DINO is ViT-Base (768 dim)
        # Using 'fc' to match the keys in the saved .pth file
        self.fc = nn.Linear(768, NUM_CLASSES)

    def forward(self, x):
        # Backbone output: BaseHelperOutput or tuple
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state
        # Global Average Pooling
        pooled = features.mean(dim=1)
        return self.fc(pooled)

def get_model():
    print(f"Loading model backbone: {MODEL_NAME}...")
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    model = RadDinoClassifier(backbone)
    
    # Path to weights
    weight_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "rad_dino_finetuned.pth")
    
    if os.path.exists(weight_path):
        print(f"Loading finetuned weights from {weight_path}...")
        try:
            state_dict = torch.load(weight_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using initialized weights (Warning: Predictions will be random)")
    else:
        print(f"Warning: Weight file not found at {weight_path}")
        print("Using initialized weights (Warning: Predictions will be random)")
        
    model.to(DEVICE)
    model.eval()
    return model
