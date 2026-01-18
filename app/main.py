import logging
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models_db import Client, PredictionRequest as DBPredictionRequest, PredictionOutput
from app.schemas import PredictFromDBRequest, PredictionResponse
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

@app.get("/clients/{client_id}")
def get_client(client_id: int, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"client_id": client.id, "n_features": len(client.features)}


@app.get("/requests/{request_id}")
def get_request(request_id: int, db: Session = Depends(get_db)):
    req = db.query(DBPredictionRequest).filter(DBPredictionRequest.id == request_id).first()
    if req is None:
        raise HTTPException(status_code=404, detail="Request not found")

    out = db.query(PredictionOutput).filter(PredictionOutput.request_id == req.id).first()

    return {
        "request_id": req.id,
        "client_id": req.client_id,
        "created_at": str(req.created_at),
        "output": None if out is None else {
            "label": out.label,
            "proba": out.proba,
            "threshold": out.threshold,
            "created_at": str(out.created_at),
        },
    }
@app.get("/predictions/latest")
def latest_predictions(limit: int = 5, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 50))

    rows = (
        db.query(DBPredictionRequest, PredictionOutput)
        .join(PredictionOutput, PredictionOutput.request_id == DBPredictionRequest.id)
        .order_by(DBPredictionRequest.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "request_id": req.id,
            "client_id": req.client_id,
            "created_at": str(req.created_at),
            "label": out.label,
            "proba": out.proba,
            "threshold": out.threshold,
        }
        for req, out in rows
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict_from_db(payload: PredictFromDBRequest, db: Session = Depends(get_db)):
    artifacts = getattr(app.state, "model_artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1) Récupérer les features du client depuis la DB
    client = db.query(Client).filter(Client.id == payload.client_id).first()
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")

    # 2) Log INPUT (snapshot) dans prediction_requests
    req_row = DBPredictionRequest(client_id=client.id, features=client.features)
    db.add(req_row)
    db.flush()  # récupère req_row.id sans commit

    try:
        # 3) Inference
        label, proba = predict_features(artifacts, client.features)
        threshold = float(artifacts.get("threshold", 0.5))
    except Exception:
        logger.exception("Inference failed")
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur lors de l'inférence du modèle")

    # 4) Log OUTPUT dans prediction_outputs
    out_row = PredictionOutput(
        request_id=req_row.id,
        label=int(label),
        proba=float(proba),
        threshold=threshold,
    )
    db.add(out_row)
    db.commit()
    db.refresh(req_row)
    db.refresh(out_row)


    return PredictionResponse(label=int(label), proba=float(proba))

