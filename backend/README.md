# Chest X-Ray Analysis Backend

This is the backend for the Self-Supervised Chest X-Ray Analysis project.

## Architecture
- **Framework**: FastAPI
- **Model**: `microsoft/rad-dino` (ViT backbone) + Custom Classifier Head
- **Task**: Multi-class Classification (Normal, Pneumonia, COVID-19, Tuberculosis)
- **Explainability**: Grad-CAM

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Endpoints

- `POST /api/analyze`: Upload an image for analysis.
   - Returns classification result and confidence.

## Note
The model uses `microsoft/rad-dino` as a backbone. The classification head is currently initialized randomly (unless weights are loaded), so predictions verify the pipeline but not medical accuracy without fine-tuning.
