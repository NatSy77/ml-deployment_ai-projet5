def test_predict_valid(client):
    payload = {"text": "Ce produit est excellent"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "label" in r.json()
    assert "proba" in r.json()


def test_predict_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_extra_field(client):
    r = client.post("/predict", json={"text": "ok", "foo": "bar"})
    assert r.status_code == 422
