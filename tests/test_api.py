def test_predict_valid(client):
    payload = {"features": [0.0] * 61}
    r = client.post("/predict_raw", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "label" in data
    assert "proba" in data
    assert 0.0 <= data["proba"] <= 1.0


def test_predict_invalid_length(client):
    payload = {"features": [0.0] * 60}
    r = client.post("/predict_raw", json=payload)
    assert r.status_code == 422


def test_predict_extra_field(client):
    payload = {"features": [0.0] * 61, "foo": "bar"}
    r = client.post("/predict_raw", json=payload)
    assert r.status_code == 422
