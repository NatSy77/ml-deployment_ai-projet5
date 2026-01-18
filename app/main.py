import logging
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse
from app.models_ml import load_model_artifacts, predict_features, ModelNotLoadedError

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="ML Deployment API",
    description="API FastAPI pour exposer un modèle de classification (Projet OpenClassrooms).",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        app.state.model_artifacts = load_model_artifacts()
    except ModelNotLoadedError as e:
        raise RuntimeError(f"Model loading failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading model: {e}") from e


@app.get("/")
def root():
    return {"message": "API is running. See /docs for documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    artifacts = getattr(app.state, "model_artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        label, proba = predict_features(artifacts, request.features)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Erreur lors de l'inférence du modèle")

    return PredictionResponse(label=label, proba=proba)
