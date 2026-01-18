from pathlib import Path
from functools import lru_cache
from typing import Tuple, Dict, Any

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model_project4.pkl"


class ModelNotLoadedError(Exception):
    """Exception levée si le modèle ou les artefacts ne peuvent pas être chargés."""
    pass


@lru_cache(maxsize=1)
def load_model_artifacts() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Fichier modèle introuvable à l'emplacement : {MODEL_PATH}"
        )

    artifacts = joblib.load(MODEL_PATH)

    # Fallback si on a sauvegardé un objet sklearn directement
    if not isinstance(artifacts, dict) or "model" not in artifacts:
        artifacts = {"model": artifacts, "threshold": 0.5}

    artifacts.setdefault("threshold", 0.5)
    return artifacts


def predict_features(artifacts: Dict[str, Any], features) -> Tuple[int, float]:
    """
    Prédit à partir d'un vecteur de features numériques.
    """
    model = artifacts["model"]
    threshold = float(artifacts.get("threshold", 0.5))

    X = np.array(features, dtype=float).reshape(1, -1)  # <-- FIX: 2D array
    proba_pos = float(model.predict_proba(X)[:, 1][0])
    y_hat = int(proba_pos >= threshold)
    return y_hat, proba_pos
