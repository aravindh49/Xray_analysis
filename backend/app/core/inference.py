import torch
import torch.nn.functional as F

# Alphabetical order assumed from ImageFolder
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

def predict(model, input_tensor):
    """
    Runs inference ensuring robust error handling and hallucination avoidance.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
    # Get top prediction
    confidence, predicted_idx = torch.max(probabilities, 1)
    confidence_score = confidence.item()
    predicted_label = CLASS_NAMES[predicted_idx.item()]
    
    # Probability dict
    probs_dict = {name: round(prob.item(), 4) for name, prob in zip(CLASS_NAMES, probabilities[0])}
    
    # Hallucination Avoidance & False Positive Reduction
    
    # Rule 1: Low Confidence (< 60%) -> Default to Normal (Safe)
    if confidence_score < 0.60:
        return {
            "label": "Normal",
            "confidence": round(confidence_score * 100, 2),
            "probabilities": probs_dict,
            "note": "Low confidence detected. Defaulting to Normal/Healthy as a safety precaution."
        }
        
    # Rule 2: Strict Disease Threshold
    # If model predicts a disease (COVID/Pneumonia/TB) but isn't VERY sure (< 75%), 
    # assume it's a False Positive and return Normal.
    if predicted_label != "Normal" and confidence_score < 0.75:
         return {
            "label": "Normal",
            "confidence": round(confidence_score * 100, 2),
            "probabilities": probs_dict,
            "note": f"Model leaned towards {predicted_label} ({round(confidence_score*100, 1)}%) but confidence was insufficient. Analysis interprets as Normal."
        }
    
    return {
        "label": predicted_label,
        "confidence": round(confidence_score * 100, 2),
        "probabilities": probs_dict,
        "note": "Analysis successful."
    }
