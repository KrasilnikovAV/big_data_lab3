from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .predict import DEFAULT_PREDICTION_SERVICE, PredictionModel, load_model
from .storage import (
    NullPredictionStore,
    PredictionClassStat,
    PredictionLogRecord,
    PredictionStore,
    build_prediction_store_from_env,
)

MODEL: PredictionModel | None = None
PREDICTION_STORE: PredictionStore = NullPredictionStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, PREDICTION_STORE
    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    try:
        MODEL = load_model(model_path)
    except FileNotFoundError:
        MODEL = None
    PREDICTION_STORE = build_prediction_store_from_env()
    PREDICTION_STORE.ensure_ready()
    yield


app = FastAPI(title="BBC News Classifier API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    encoding: Literal["plain", "base64"] = "plain"

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        if any(str(text).strip() == "" for text in texts):
            raise ValueError("Texts must not contain blank values.")
        return texts


class PredictResponse(BaseModel):
    predictions: list[str]


class HealthResponse(BaseModel):
    status: str
    database: str


class PredictionLogResponse(BaseModel):
    request_id: str
    row_position: int
    input_text: str
    predicted_label: str
    created_at: str


class PredictionHistoryResponse(BaseModel):
    records: list[PredictionLogResponse]


class PredictionClassStatResponse(BaseModel):
    predicted_label: str
    count: int
    share: float


class PredictionStatsResponse(BaseModel):
    classes: list[PredictionClassStatResponse]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    payload = DEFAULT_PREDICTION_SERVICE.health_status(MODEL)
    payload["database"] = PREDICTION_STORE.health_status()
    return HealthResponse(**payload)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        prepared_texts = DEFAULT_PREDICTION_SERVICE.decoder.decode(
            payload.texts,
            encoding=payload.encoding,
        )
        predictions = DEFAULT_PREDICTION_SERVICE.predict(
            MODEL,
            prepared_texts,
            encoding="plain",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        PREDICTION_STORE.save_predictions(prepared_texts, predictions)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Prediction was not saved to ClickHouse.") from exc

    return PredictResponse(predictions=predictions)


@app.get("/predictions", response_model=PredictionHistoryResponse)
def get_recent_predictions(limit: int = 10) -> PredictionHistoryResponse:
    records = PREDICTION_STORE.fetch_recent_predictions(limit=limit)
    return PredictionHistoryResponse(
        records=[PredictionLogResponse(**_serialize_record(record)) for record in records]
    )


@app.get("/predictions/stats", response_model=PredictionStatsResponse)
def get_prediction_stats() -> PredictionStatsResponse:
    class_stats = PREDICTION_STORE.fetch_prediction_class_stats()
    return PredictionStatsResponse(
        classes=[PredictionClassStatResponse(**_serialize_class_stat(stat)) for stat in class_stats]
    )


def _serialize_record(record: PredictionLogRecord) -> dict[str, str | int]:
    return {
        "request_id": record.request_id,
        "row_position": record.row_position,
        "input_text": record.input_text,
        "predicted_label": record.predicted_label,
        "created_at": record.created_at,
    }


def _serialize_class_stat(stat: PredictionClassStat) -> dict[str, str | int | float]:
    return {
        "predicted_label": stat.predicted_label,
        "count": stat.count,
        "share": stat.share,
    }
