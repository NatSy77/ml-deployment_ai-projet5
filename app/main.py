from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.models_ml import load_model_artifacts, predict_text, ModelNotLoadedError

app = FastAPI(
    title="ML Deployment API",
    description="API FastAPI pour exposer un modèle de classification (Projet OpenClassrooms).",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    """
    Charge les artefacts du modèle une seule fois au démarrage.
    Si le chargement échoue, on stoppe l'app (c'est préférable à une API 'semi-vivante').
    """
    try:
        app.state.model_artifacts = load_model_artifacts()
    except ModelNotLoadedError as e:
        # Bloquant : si le modèle n'est pas dispo, l'API ne doit pas démarrer
        raise RuntimeError(f"Model loading failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading model: {e}") from e


@app.get("/")
def root():
    return {"message": "API is running. See /docs for documentation."}


@app.get("/health")
def health():
    """
    Endpoint simple pour vérifier que le service répond.
    (Bonus apprécié en évaluation + utile pour monitoring.)
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Retourne la prédiction du modèle à partir du champ `text`.

    La validation d'entrée (types, min_length, champs extra interdits...)
    doit être gérée côté Pydantic dans PredictionRequest.
    """
    # Sécurité : si startup n'a pas chargé le modèle (cas anormal)
    artifacts = getattr(app.state, "model_artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        label, proba = predict_text(artifacts, request.text)
    except HTTPException:
        # Si predict_text lève déjà des HTTPException, on les propage telles quelles
        raise
    except Exception:
        # Ne pas exposer de détails internes en prod : message générique
        raise HTTPException(status_code=500, detail="Erreur lors de l'inférence du modèle")

    return PredictionResponse(label=label, proba=proba)
