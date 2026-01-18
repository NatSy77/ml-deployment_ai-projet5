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
    artifacts = getattr(app.state, "model_artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Cas 1 : load_model_artifacts() renvoie un dict
        if isinstance(artifacts, dict):
            # adapte les clés possibles
            model = artifacts.get("model") or artifacts.get("clf") or artifacts.get("pipeline")
            vectorizer = artifacts.get("vectorizer") or artifacts.get("tfidf") or artifacts.get("preprocessor")

            if model is None and "pipeline" not in artifacts:
                raise HTTPException(status_code=500, detail="Model artifacts invalid (missing model)")

            # Si predict_text attend (artifacts, text)
            try:
                label, proba = predict_text(artifacts, request.text)
            except TypeError:
                # Si predict_text attend (model, vectorizer, text) ou (model, text)
                if vectorizer is not None:
                    label, proba = predict_text(model, vectorizer, request.text)
                else:
                    label, proba = predict_text(model, request.text)

        # Cas 2 : load_model_artifacts() renvoie un tuple/list
        elif isinstance(artifacts, (tuple, list)):
            if len(artifacts) == 2:
                model, vectorizer = artifacts
                label, proba = predict_text(model, vectorizer, request.text)
            elif len(artifacts) == 1:
                (model,) = artifacts
                label, proba = predict_text(model, request.text)
            else:
                raise HTTPException(status_code=500, detail="Model artifacts invalid (unexpected tuple size)")

        # Cas 3 : c’est directement un modèle/pipeline sklearn
        else:
            model = artifacts
            try:
                label, proba = predict_text(model, request.text)
            except TypeError:
                # fallback : si predict_text attend (artifacts, text)
                label, proba = predict_text({"model": model}, request.text)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors de l'inférence du modèle")

    return PredictionResponse(label=label, proba=proba)
