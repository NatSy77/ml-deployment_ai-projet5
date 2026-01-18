from pathlib import Path
from functools import lru_cache
from typing import Tuple, Dict, Any

import joblib


# Chemin vers le fichier de modèle
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model_project4.pkl"


class ModelNotLoadedError(Exception):
    """Exception levée si le modèle ou les artefacts ne peuvent pas être chargés."""
    pass


@lru_cache(maxsize=1)
def load_model_artifacts() -> Dict[str, Any]:
    """
    Charge les artefacts du modèle depuis le fichier .pkl.
    Utilise un cache pour ne le faire qu'une seule fois.

    Attendus possibles dans le .pkl :
      - un dict contenant au choix :
          {"pipeline": <sklearn Pipeline>, "threshold": 0.5}
          {"model": <sklearn estimator>, "vectorizer": <sklearn vectorizer>, "threshold": 0.5}
          {"model": <sklearn Pipeline>, "threshold": 0.5}  (cas fréquent : pipeline stocké dans 'model')
      - ou directement un objet sklearn (pipeline ou estimator)
    """
    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Fichier modèle introuvable à l'emplacement : {MODEL_PATH}"
        )

    artifacts = joblib.load(MODEL_PATH)

    # Fallback si on a sauvegardé un objet sklearn directement
    if not isinstance(artifacts, dict):
        artifacts = {"model": artifacts, "threshold": 0.5}

    # Garantir un seuil par défaut
    artifacts.setdefault("threshold", 0.5)

    return artifacts


def predict_text(artifacts: Dict[str, Any], text: str) -> Tuple[str, float]:
    """
    Prédit à partir d'un texte.

    Supporte :
      - artifacts["pipeline"] = Pipeline (vectorizer + model)
      - artifacts["model"] = Pipeline
      - artifacts["model"] + artifacts["vectorizer"] (estimator + vectorizer séparés)

    Returns:
        label (str): classe prédite (0/1 en str).
        proba (float): probabilité de la classe positive.
    """
    threshold = float(artifacts.get("threshold", 0.5))

    # 1) Pipeline explicitement fourni
    pipeline = artifacts.get("pipeline", None)
    if pipeline is not None:
        X = [text]
        proba_pos = float(pipeline.predict_proba(X)[:, 1][0])
        y_hat = int(proba_pos >= threshold)
        return str(y_hat), proba_pos

    # 2) "model" peut être un pipeline OU un estimator "nu"
    model = artifacts.get("model", None)
    if model is None:
        raise ValueError("Artefacts invalides: clé 'model' absente.")

    # 2a) Si un vectorizer est présent, on transforme avant predict_proba
    vectorizer = artifacts.get("vectorizer", None)
    if vectorizer is not None:
        X_vec = vectorizer.transform([text])  # <-- FIX: matrice 2D attendue par sklearn
        proba_pos = float(model.predict_proba(X_vec)[:, 1][0])
        y_hat = int(proba_pos >= threshold)
        return str(y_hat), proba_pos

    # 2b) Sinon, on suppose que "model" est un pipeline texte->features
    # (cas où tu as sauvegardé un Pipeline directement dans 'model')
    X = [text]
    proba_pos = float(model.predict_proba(X)[:, 1][0])
    y_hat = int(proba_pos >= threshold)
    return str(y_hat), proba_pos
