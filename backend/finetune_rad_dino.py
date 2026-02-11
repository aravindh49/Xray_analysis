import torch
import os
import torch.nn as nn
from transformers import AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/rad-dino"
NUM_CLASSES = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
MAX_SAMPLES = 500 # For <15 min training on CPU

def train():
    print(f"Using device: {DEVICE}")

    # 1. Load Dataset
    # Ensure consistency with preprocessing.py (RGB)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "train")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please create the structure: backend/dataset/train/[class_name]")
        return

    try:
        full_ds = datasets.ImageFolder(dataset_path, transform=transform)
        print(f"Found {len(full_ds)} images in {len(full_ds.classes)} classes: {full_ds.classes}")
        
        # Limit dataset size for speed if needed
        if len(full_ds) > MAX_SAMPLES:
            print(f"LIMITING DATASET to {MAX_SAMPLES} images for speed (CPU mode)...")
            indices = torch.randperm(len(full_ds))[:MAX_SAMPLES]
            train_ds = Subset(full_ds, indices)
        else:
            train_ds = full_ds
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Setup Model
    print("Loading backbone...")
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    
    # Classification head
    class RadDINOClassifier(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            # Rad-DINO is ViT-Base (768 dim)
            self.classifier = nn.Linear(768, NUM_CLASSES)

        def forward(self, x):
            # Backbone output: BaseHelperOutput or tuple
            outputs = self.backbone(pixel_values=x)
            features = outputs.last_hidden_state
            # Global Average Pooling
            pooled = features.mean(dim=1)
            return self.classifier(pooled)

    model = RadDINOClassifier(backbone).to(DEVICE)

    # Freeze backbone initially (Linear Probe)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 3. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # 4. Save Model
    save_path = "rad_dino_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
