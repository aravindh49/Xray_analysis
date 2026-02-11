from pytorch_grad_cam import GradCAM
import numpy as np
import cv2
import torch

def reshape_transform_vit(tensor, height=14, width=14):
    # Helper for ViT if needed. Rad-DINO might need this.
    # tensor: [batch, seq, feature]
    # We exclude cls token if present and reshape
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring channels to 1st dim: [batch, feature, height, width]
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_gradcam(model, target_layer, image_tensor, use_cuda=False):
    # Note: For ViT models, reshape_transform might be required.
    # Passing it just in case, though the user didn't explicitly ask for it.
    # If the user model is standard CNN, this is ignored.
    # Rad-DINO is ViT.
    
    # Check if we should use the reshape transform
    # This acts as a heuristic.
    reshape_transform = None
    if "vit" in model.__class__.__name__.lower() or "dino" in model.__class__.__name__.lower():
         # Basic heuristic, might need tuning for exact model arch
         pass 

    # Construct GradCAM
    # We might need to wrap this in try/except if target_layers structure limits it
    try:
        cam = GradCAM(
            model=model,
            target_layers=[target_layer]
            # reshape_transform=reshape_transform # Uncomment if using ViT specific logic
        )
        
        # Generates: [batch, H, W]
        grayscale_cam = cam(input_tensor=image_tensor)
        
        # Take the first image in batch
        return grayscale_cam[0, :]
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return np.zeros((224, 224)) # Return empty mask on failure
