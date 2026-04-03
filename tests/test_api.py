from __future__ import annotations

import base64
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from bbc_news import api
from bbc_news.storage import PredictionClassStat, PredictionLogRecord


class DummyModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> list[str]:
        self.calls.append(texts)
        return ["sport" for _ in texts]


class DummyStore:
    def __init__(self) -> None:
        self.saved_batches: list[tuple[list[str], list[str]]] = []
        self.records: list[PredictionLogRecord] = []
        self.class_stats: list[PredictionClassStat] = []
        self.health = "disabled"

    def ensure_ready(self) -> None:
        return None

    def health_status(self) -> str:
        return self.health

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        self.saved_batches.append((texts, predictions))

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        return self.records[:limit]

    def fetch_prediction_class_stats(self) -> list[PredictionClassStat]:
        return self.class_stats


def test_predict_endpoint_returns_predictions() -> None:
    store = DummyStore()
    with TestClient(api.app) as client:
        api.MODEL = DummyModel()
        api.PREDICTION_STORE = store
        response = client.post("/predict", json={"texts": ["one", "two"]})

    assert response.status_code == 200
    assert response.json() == {"predictions": ["sport", "sport"]}
    assert store.saved_batches == [(["one", "two"], ["sport", "sport"])]


def test_predict_endpoint_returns_503_when_model_not_loaded() -> None:
    with TestClient(api.app) as client:
        api.MODEL = None
        api.PREDICTION_STORE = DummyStore()
        response = client.post("/predict", json={"texts": ["one"]})

    assert response.status_code == 503


def test_health_endpoint() -> None:
    store = DummyStore()
    store.health = "ok"
    with TestClient(api.app) as client:
        api.MODEL = DummyModel()
        api.PREDICTION_STORE = store
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "ok"


def test_predict_endpoint_decodes_base64_payload() -> None:
    encoded_text = base64.b64encode("team won the final".encode("utf-8")).decode("utf-8")
    model = DummyModel()
    store = DummyStore()

    with TestClient(api.app) as client:
        api.MODEL = model
        api.PREDICTION_STORE = store
        response = client.post(
            "/predict",
            json={"texts": [encoded_text], "encoding": "base64"},
        )

    assert response.status_code == 200
    assert response.json() == {"predictions": ["sport"]}
    assert model.calls == [["team won the final"]]
    assert store.saved_batches == [(["team won the final"], ["sport"])]


def test_predict_endpoint_returns_422_for_invalid_base64() -> None:
    with TestClient(api.app) as client:
        api.MODEL = DummyModel()
        api.PREDICTION_STORE = DummyStore()
        response = client.post(
            "/predict",
            json={"texts": ["not-base64"], "encoding": "base64"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid base64 payload."


def test_predictions_endpoint_returns_recent_rows() -> None:
    store = DummyStore()
    store.records = [
        PredictionLogRecord(
            request_id="request-1",
            row_position=0,
            input_text="market rallies",
            predicted_label="business",
            created_at=datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc).isoformat(),
        )
    ]

    with TestClient(api.app) as client:
        api.MODEL = DummyModel()
        api.PREDICTION_STORE = store
        response = client.get("/predictions?limit=1")

    assert response.status_code == 200
    assert response.json() == {
        "records": [
            {
                "request_id": "request-1",
                "row_position": 0,
                "input_text": "market rallies",
                "predicted_label": "business",
                "created_at": "2026-04-02T09:00:00+00:00",
            }
        ]
    }


def test_prediction_stats_endpoint_returns_class_distribution() -> None:
    store = DummyStore()
    store.class_stats = [
        PredictionClassStat(predicted_label="sport", count=3, share=0.75),
        PredictionClassStat(predicted_label="business", count=1, share=0.25),
    ]

    with TestClient(api.app) as client:
        api.MODEL = DummyModel()
        api.PREDICTION_STORE = store
        response = client.get("/predictions/stats")

    assert response.status_code == 200
    assert response.json() == {
        "classes": [
            {"predicted_label": "sport", "count": 3, "share": 0.75},
            {"predicted_label": "business", "count": 1, "share": 0.25},
        ]
    }
