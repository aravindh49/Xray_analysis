from PIL import Image
import io

def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    """
    Validates the uploaded image.
    Returns (is_valid, message).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check if file is readable
        image.verify() 
        
        # Re-open because verify() closes the file pointer/corrupts it for subsequent ops in some versions
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check resolution
        width, height = image.size
        if width < 224 or height < 224:
            return False, f"Image resolution too low ({width}x{height}). Minimum required is 224x224."
            
        # Optional: Check if consistent with X-Ray (e.g. not too colorful)
        # Convert to HSV and check saturation? 
        # For now, we trust the user a bit but enforce resolution.
        
        return True, "Valid"
        
    except Exception as e:
        return False, "Invalid image file. Please upload a valid image."
