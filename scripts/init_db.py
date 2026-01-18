from app.db import engine, Base
from app.models_db import Client, PredictionRequest, PredictionOutput  # noqa: F401

def main():
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created")

if __name__ == "__main__":
    main()
