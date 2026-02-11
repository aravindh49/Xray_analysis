from pydantic import BaseModel
from typing import Dict, Optional

class PredictionResult(BaseModel):
    label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    note: Optional[str] = None

class AnalysisResponse(BaseModel):
    status: str
    diagnosis: PredictionResult
    gradcam_base64: Optional[str] = None
