from pathlib import Path
from functools import lru_cache
from typing import Tuple, Dict, Any

import joblib
import numpy as np


# Chemin vers le fichier de modèle
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model_project4.pkl"


class ModelNotLoadedError(Exception):
    """Exception levée si le modèle ou les artefacts ne peuvent pas être chargés."""
    pass


@lru_cache(maxsize=1)
def load_model_artifacts() -> Dict[str, Any]:
    """
    Charge les artefacts du modèle (modèle + seuil) depuis le fichier .pkl.
    Utilise un cache pour ne le faire qu'une seule fois.
    """
    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Fichier modèle introuvable à l'emplacement : {MODEL_PATH}"
        )

    artifacts = joblib.load(MODEL_PATH)

    # Si on a juste sauvegardé le modèle sans dict, on adapte
    if not isinstance(artifacts, dict) or "model" not in artifacts:
        artifacts = {"model": artifacts, "threshold": 0.5}  # fallback

    return artifacts


def predict_text(artifacts: Dict[str, Any], text: str) -> Tuple[str, float]:
    """
    Applique la régression logistique sur un texte.

    Args:
        artifacts: dictionnaire contenant au moins 'model' et 'threshold'.
        text: texte à classifier.

    Returns:
        label (str): classe prédite (0/1 ou autre).
        proba (float): probabilité de la classe positive.
    """
    model = artifacts["model"]
    threshold = artifacts.get("threshold", 0.5)

    # La régression logistique scikit-learn attend une liste
    X = [text]

    # Probabilité de la classe positive (colonne 1)
    proba_pos = float(model.predict_proba(X)[:, 1][0])

    # Application du seuil t_opt
    y_hat = int(proba_pos >= threshold)

    # On renvoie y_hat en str pour rester générique
    return str(y_hat), proba_pos
