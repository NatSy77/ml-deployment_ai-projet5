from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à classifier")

    class Config:
        extra = "forbid"  # REFUSE les champs inconnus


class PredictionResponse(BaseModel):
    label: int
    proba: float
