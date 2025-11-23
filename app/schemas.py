from pydantic import BaseModel

class PredictionRequest(BaseModel):
    text: str  # à adapter plus tard selon ton modèle réel

class PredictionResponse(BaseModel):
    label: str
    proba: float
