import torch
from torchvision import transforms
from PIL import Image
import io

def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses the image bytes to match model requirements.
    """
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply transform
    tensor = transform(image)
    
    # Add batch dimension [1, 3, 224, 224]
    return tensor.unsqueeze(0)
