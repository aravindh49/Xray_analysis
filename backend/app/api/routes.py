from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
from app.core.preprocessing import transform_image
from app.core.validators import validate_image
from app.core.inference import predict
from app.core.hf_model import get_model
from app.reports.pdf_generator import generate_pdf_report
from app.core.email_sender import send_email_with_report

router = APIRouter()

# Load model once
model = get_model()

class DiagnosisReportRequest(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    note: str

class EmailRequest(BaseModel):
    email: str
    patient_name: str
    age: str
    diagnosis: DiagnosisReportRequest

@router.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    # Read file
    image_bytes = await file.read()
    
    # Validation
    is_valid, message = validate_image(image_bytes)
    if not is_valid:
         return {"status": "error", "message": message}
    
    try:
        # Preprocessing
        tensor = transform_image(image_bytes).to(next(model.parameters()).device)
        
        # Inference
        result = predict(model, tensor)
        
        return {
            "status": "success",
            "diagnosis": result
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-report")
async def download_report(data: DiagnosisReportRequest):
    try:
        pdf_bytes = generate_pdf_report(data.dict(), None)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/email-report")
async def email_report(data: EmailRequest):
    try:
        # Generate PDF
        pdf_bytes = generate_pdf_report(
            diagnosis_data=data.diagnosis.dict(), 
            image_bytes=None,
            patient_name=data.patient_name,
            patient_age=data.age
        )
        
        # Send Email
        success, msg = send_email_with_report(
            to_email=data.email,
            subject=f"Chest X-Ray Analysis Report - {data.patient_name}",
            body=f"Dear {data.patient_name},\n\nPlease find attached your AI-generated diagnosis report.\n\nResult: {data.diagnosis.label}\nConfidence: {data.diagnosis.confidence}%",
            pdf_bytes=pdf_bytes
        )
        
        if success:
            return {"status": "success", "message": "Email sent successfully."}
        else:
            return {"status": "error", "message": f"Failed to send email: {msg}"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    return {"status": "running"}
