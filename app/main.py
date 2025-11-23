from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse

# Instance FastAPI principale
app = FastAPI(
    title="Futurisys ML Deployment API",
    description="API de déploiement du modèle du projet 4",
    version="0.1.0",
)

@app.get("/")
def root():
    return {"message": "Futurisys ML API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint de prédiction (version temporaire).
    Le vrai modèle du projet 4 sera intégré plus tard.
    """
    dummy_label = "placeholder"
    dummy_proba = 0.0
    return PredictionResponse(label=dummy_label, proba=dummy_proba)
