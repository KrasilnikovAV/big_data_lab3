from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import clickhouse_connect

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ClickHouseSettings:
    host: str
    port: int
    username: str
    password: str
    database: str
    predictions_table: str = "prediction_logs"
    dataset_table: str = "dataset_rows"
    secure: bool = False


@dataclass(frozen=True)
class PredictionLogRecord:
    request_id: str
    row_position: int
    input_text: str
    predicted_label: str
    created_at: str


@dataclass(frozen=True)
class PredictionClassStat:
    predicted_label: str
    count: int
    share: float


@dataclass(frozen=True)
class DatasetRecord:
    article_id: str
    text: str
    category: str | None = None


class PredictionStore(Protocol):
    def ensure_ready(self) -> None:
        ...

    def health_status(self) -> str:
        ...

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        ...

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        ...

    def fetch_prediction_class_stats(self) -> list[PredictionClassStat]:
        ...


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Environment variable '{name}' must be set.")
    return value


def _validate_identifier(value: str, field_name: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match pattern {IDENTIFIER_PATTERN.pattern!r}, got {value!r}."
        )
    return value


def load_clickhouse_settings_from_env() -> ClickHouseSettings | None:
    if not _parse_bool(os.getenv("CLICKHOUSE_ENABLED"), default=False):
        return None

    port_value = _require_env("CLICKHOUSE_PORT")
    try:
        port = int(port_value)
    except ValueError as exc:
        raise ValueError("Environment variable 'CLICKHOUSE_PORT' must be an integer.") from exc

    return ClickHouseSettings(
        host=_require_env("CLICKHOUSE_HOST"),
        port=port,
        username=_require_env("CLICKHOUSE_USER"),
        password=_require_env("CLICKHOUSE_PASSWORD"),
        database=_validate_identifier(_require_env("CLICKHOUSE_DATABASE"), "database"),
        predictions_table=_validate_identifier(
            os.getenv("CLICKHOUSE_PREDICTIONS_TABLE", "prediction_logs").strip(),
            "predictions_table",
        ),
        dataset_table=_validate_identifier(
            os.getenv("CLICKHOUSE_DATASET_TABLE", "dataset_rows").strip(),
            "dataset_table",
        ),
        secure=_parse_bool(os.getenv("CLICKHOUSE_SECURE"), default=False),
    )


def build_prediction_store_from_env() -> PredictionStore:
    settings = load_clickhouse_settings_from_env()
    if settings is None:
        return NullPredictionStore()
    return ClickHousePredictionStore(settings)


def create_clickhouse_client(settings: ClickHouseSettings):
    return clickhouse_connect.get_client(
        host=settings.host,
        port=settings.port,
        username=settings.username,
        password=settings.password,
        database=settings.database,
        secure=settings.secure,
    )


class NullPredictionStore:
    def ensure_ready(self) -> None:
        return None

    def health_status(self) -> str:
        return "disabled"

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        return None

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        return []

    def fetch_prediction_class_stats(self) -> list[PredictionClassStat]:
        return []


class ClickHousePredictionStore:
    def __init__(self, settings: ClickHouseSettings, client=None) -> None:
        self.settings = settings
        self.client = client or create_clickhouse_client(settings)

    @property
    def _predictions_table_ref(self) -> str:
        return f"{self.settings.database}.{self.settings.predictions_table}"

    @property
    def _dataset_table_ref(self) -> str:
        return f"{self.settings.database}.{self.settings.dataset_table}"

    def ensure_ready(self) -> None:
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.settings.database}")
        self.client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._predictions_table_ref} (
                request_id UUID,
                created_at DateTime('UTC'),
                row_position UInt32,
                input_text String,
                predicted_label LowCardinality(String)
            )
            ENGINE = MergeTree
            ORDER BY (created_at, request_id, row_position)
            """
        )
        self.client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._dataset_table_ref} (
                dataset_split LowCardinality(String),
                article_id String,
                text String,
                category Nullable(String),
                loaded_at DateTime('UTC')
            )
            ENGINE = MergeTree
            ORDER BY (dataset_split, article_id)
            """
        )

    def health_status(self) -> str:
        self.client.query("SELECT 1")
        return "ok"

    def save_predictions(self, texts: list[str], predictions: list[str]) -> None:
        if len(texts) != len(predictions):
            raise ValueError("Texts and predictions must have the same length.")

        created_at = datetime.now(timezone.utc).replace(microsecond=0)
        request_id = str(uuid.uuid4())
        rows = [
            [request_id, created_at, row_position, text, prediction]
            for row_position, (text, prediction) in enumerate(zip(texts, predictions, strict=True))
        ]
        self.client.insert(
            self._predictions_table_ref,
            rows,
            column_names=[
                "request_id",
                "created_at",
                "row_position",
                "input_text",
                "predicted_label",
            ],
        )

    def fetch_recent_predictions(self, limit: int = 10) -> list[PredictionLogRecord]:
        safe_limit = max(1, min(int(limit), 100))
        query = f"""
            SELECT request_id, row_position, input_text, predicted_label, created_at
            FROM {self._predictions_table_ref}
            ORDER BY created_at DESC, row_position DESC
            LIMIT {safe_limit}
        """
        rows = self.client.query(query).result_set
        return [
            PredictionLogRecord(
                request_id=str(row[0]),
                row_position=int(row[1]),
                input_text=str(row[2]),
                predicted_label=str(row[3]),
                created_at=_format_timestamp(row[4]),
            )
            for row in rows
        ]

    def fetch_prediction_class_stats(self) -> list[PredictionClassStat]:
        query = f"""
            SELECT predicted_label, count() AS prediction_count
            FROM {self._predictions_table_ref}
            GROUP BY predicted_label
            ORDER BY prediction_count DESC, predicted_label ASC
        """
        rows = self.client.query(query).result_set
        total_predictions = sum(int(row[1]) for row in rows)
        if total_predictions == 0:
            return []

        return [
            PredictionClassStat(
                predicted_label=str(row[0]),
                count=int(row[1]),
                share=round(int(row[1]) / total_predictions, 4),
            )
            for row in rows
        ]

    def insert_dataset_rows(self, dataset_split: str, records: list[DatasetRecord]) -> int:
        normalized_split = dataset_split.strip().lower()
        if normalized_split not in {"train", "test"}:
            raise ValueError("dataset_split must be either 'train' or 'test'.")
        if not records:
            return 0

        loaded_at = datetime.now(timezone.utc).replace(microsecond=0)
        rows = [
            [normalized_split, record.article_id, record.text, record.category, loaded_at]
            for record in records
        ]
        self.client.insert(
            self._dataset_table_ref,
            rows,
            column_names=["dataset_split", "article_id", "text", "category", "loaded_at"],
        )
        return len(rows)


def _format_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)
