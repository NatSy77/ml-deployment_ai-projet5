from pydantic import BaseModel, Field, conlist


N_FEATURES = 61  # IMPORTANT: ton modèle attend 61 features


class PredictionRequest(BaseModel):
    features: conlist(float, min_length=N_FEATURES, max_length=N_FEATURES) = Field(
        ...,
        description=f"Vecteur de {N_FEATURES} features numériques, dans le même ordre que l'entraînement."
    )

    class Config:
        extra = "forbid"


class PredictionResponse(BaseModel):
    label: int
    proba: float

class PredictFromDBRequest(BaseModel):
    client_id: int = Field(..., ge=1)

    class Config:
        extra = "forbid"

