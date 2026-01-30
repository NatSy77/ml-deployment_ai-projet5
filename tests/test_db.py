import os
import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError


from app.main import app
from app.db import Base, get_db
from app.models_db import Client, PredictionRequest, PredictionOutput

pytestmark = pytest.mark.integration


def _db_url():
    return os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URL")


@pytest.fixture()
def db_session():
    url = _db_url()
    if not url:
        pytest.skip("DATABASE_URL non défini : tests DB skippés")

    engine = create_engine(url)
    TestingSessionLocal = sessionmaker(bind=engine)
    
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except OperationalError:
        pytest.skip("PostgreSQL non joignable : lance docker compose up -d")


    # Reproductibilité : reset schéma
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    # Override dependency FastAPI
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Dummy modèle stable (pas dépendant du vrai modèle)
    class DummyLogisticModel:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9] for _ in range(len(X))], dtype=float)

    app.state.model_artifacts = {"model": DummyLogisticModel(), "threshold": 0.5}

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_predict_db_inserts_trace_rows(client, db_session):
    # Seed client
    db_session.add(
        Client(
            client_id=1,
            features=[0.0] * 61
        )
    )
    db_session.commit()

    # Appel endpoint DB-first
    r = client.post("/predict", json={"client_id": 1})
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["label"] in [0, 1]
    assert 0.0 <= data["proba"] <= 1.0

    # Vérification traçabilité
    requests = db_session.query(PredictionRequest).all()
    outputs = db_session.query(PredictionOutput).all()

    assert len(requests) == 1
    assert len(outputs) == 1

    req = requests[0]
    out = outputs[0]

    assert req.client_id == 1
    assert len(req.features) == 61

    assert out.request_id == req.id
    assert out.prediction in [0, 1]
    assert 0.0 <= out.probability <= 1.0
    assert out.threshold == pytest.approx(0.5)


def test_predict_db_client_not_found(client, db_session):
    r = client.post("/predict", json={"client_id": 999})
    assert r.status_code == 404

    assert db_session.query(PredictionRequest).count() == 0
    assert db_session.query(PredictionOutput).count() == 0
