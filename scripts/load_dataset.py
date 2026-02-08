from pathlib import Path
import pandas as pd

from app.db import SessionLocal
from app.models_db import Client
from sqlalchemy import create_engine


CSV_PATH = Path("data/clients.csv")
BATCH_SIZE = 2000


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable: {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    # On prend TOUTES les colonnes dans l'ordre du CSV
    feature_cols = df.columns.tolist()

    if len(feature_cols) != 61:
        raise ValueError(f"Le modèle attend 61 features. CSV contient {len(feature_cols)} colonnes.")

    # Sécurité : convertir en numérique (si une colonne est mal typée, ça plantera ici)
    df = df.apply(pd.to_numeric, errors="raise")

    db = SessionLocal()
    try:
        total = len(df)
        inserted = 0
        buffer = []

        for _, row in df.iterrows():
            features = [float(row[c]) for c in feature_cols]
            buffer.append(Client(features=features))

            if len(buffer) >= BATCH_SIZE:
                db.bulk_save_objects(buffer)
                db.commit()
                inserted += len(buffer)
                buffer = []
                print(f"✅ Inserted {inserted}/{total}")

        if buffer:
            db.bulk_save_objects(buffer)
            db.commit()
            inserted += len(buffer)

        print(f"🎉 Done. Inserted {inserted} clients into DB.")
        print("ℹ️ Ordre des features (à garder identique pour le modèle) :")
        print(feature_cols)

    finally:
        db.close()


if __name__ == "__main__":
    main()
