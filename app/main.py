from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.models_ml import load_model_artifacts, predict_text, ModelNotLoadedError

app = FastAPI(
    title="Futurisys ML Deployment API",
    description="API de déploiement du modèle de classification (projet 4)",
    version="0.1.0",
)


@app.on_event("startup")
def startup_event():
    """
    Chargement du modèle au démarrage de l'application.
    """
    try:
        load_model_artifacts()
        print("Modèle du projet 4 chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")


@app.get("/")
def root():
    return {"message": "Futurisys ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint de prédiction utilisant la régression logistique optimisée.
    """
    # Validation simple de l'entrée
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Le champ 'text' ne doit pas être vide.")

    try:
        artifacts = load_model_artifacts()
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=500, detail=str(e))

    label, proba = predict_text(artifacts, request.text)

    return PredictionResponse(label=label, proba=proba)
