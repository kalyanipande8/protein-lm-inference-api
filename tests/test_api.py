def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "fake-esm2"


def test_predict_returns_embedding(client):
    resp = client.post("/predict", json={"sequence": "MKT"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sequence_length"] == 3
    assert body["embedding_dim"] == 4
    assert body["embedding"] == [3.0, 3.0, 3.0, 3.0]


def test_predict_rejects_invalid_amino_acids(client):
    resp = client.post("/predict", json={"sequence": "MK1"})
    assert resp.status_code == 422


def test_predict_rejects_empty_sequence(client):
    resp = client.post("/predict", json={"sequence": ""})
    assert resp.status_code == 422


def test_predict_concurrent_requests_are_batched(client):
    import concurrent.futures

    sequences = ["M", "MK", "MKT", "MKTA", "MKTAY"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sequences)) as pool:
        responses = list(pool.map(
            lambda s: client.post("/predict", json={"sequence": s}), sequences
        ))

    for seq, resp in zip(sequences, responses):
        assert resp.status_code == 200
        assert resp.json()["sequence_length"] == len(seq)
