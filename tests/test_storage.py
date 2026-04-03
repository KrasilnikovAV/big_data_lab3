from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bbc_news.storage import (
    ClickHousePredictionStore,
    ClickHouseSettings,
    PredictionClassStat,
    PredictionLogRecord,
    load_clickhouse_settings_from_env,
)


class FakeQueryResult:
    def __init__(self, result_set):
        self.result_set = result_set


class FakeClient:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.inserts: list[tuple[str, list[list[object]], list[str]]] = []
        self.queries: list[str] = []
        self.next_result = FakeQueryResult([])

    def command(self, sql: str) -> None:
        self.commands.append(sql)

    def insert(self, table: str, data: list[list[object]], column_names: list[str]) -> None:
        self.inserts.append((table, data, column_names))

    def query(self, sql: str) -> FakeQueryResult:
        self.queries.append(sql)
        return self.next_result


def test_load_clickhouse_settings_from_env_requires_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKHOUSE_ENABLED", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse")
    monkeypatch.setenv("CLICKHOUSE_PORT", "8123")
    monkeypatch.setenv("CLICKHOUSE_USER", "app_user")
    monkeypatch.delenv("CLICKHOUSE_PASSWORD", raising=False)
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "bbc_news")

    with pytest.raises(ValueError, match="CLICKHOUSE_PASSWORD"):
        load_clickhouse_settings_from_env()


def test_clickhouse_store_creates_schema_and_inserts_predictions() -> None:
    fake_client = FakeClient()
    store = ClickHousePredictionStore(
        ClickHouseSettings(
            host="clickhouse",
            port=8123,
            username="app_user",
            password="secret",
            database="bbc_news",
        ),
        client=fake_client,
    )

    store.ensure_ready()
    store.save_predictions(["one", "two"], ["sport", "business"])

    assert len(fake_client.commands) == 3
    assert fake_client.inserts[0][0] == "bbc_news.prediction_logs"
    assert fake_client.inserts[0][2] == [
        "request_id",
        "created_at",
        "row_position",
        "input_text",
        "predicted_label",
    ]
    assert len(fake_client.inserts[0][1]) == 2


def test_clickhouse_store_reads_recent_predictions() -> None:
    fake_client = FakeClient()
    fake_client.next_result = FakeQueryResult(
        [
            (
                "request-1",
                0,
                "stock market gains",
                "business",
                datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc),
            )
        ]
    )
    store = ClickHousePredictionStore(
        ClickHouseSettings(
            host="clickhouse",
            port=8123,
            username="app_user",
            password="secret",
            database="bbc_news",
        ),
        client=fake_client,
    )

    records = store.fetch_recent_predictions(limit=1)

    assert records == [
        PredictionLogRecord(
            request_id="request-1",
            row_position=0,
            input_text="stock market gains",
            predicted_label="business",
            created_at="2026-04-02T10:00:00+00:00",
        )
    ]


def test_clickhouse_store_aggregates_prediction_class_stats() -> None:
    fake_client = FakeClient()
    fake_client.next_result = FakeQueryResult(
        [
            ("sport", 3),
            ("business", 1),
        ]
    )
    store = ClickHousePredictionStore(
        ClickHouseSettings(
            host="clickhouse",
            port=8123,
            username="app_user",
            password="secret",
            database="bbc_news",
        ),
        client=fake_client,
    )

    stats = store.fetch_prediction_class_stats()

    assert stats == [
        PredictionClassStat(predicted_label="sport", count=3, share=0.75),
        PredictionClassStat(predicted_label="business", count=1, share=0.25),
    ]
