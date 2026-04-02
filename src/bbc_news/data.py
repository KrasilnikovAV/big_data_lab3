from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import DataConfig


def _clean_text_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_training_frame(config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.train_path)
    frame = frame[[config.id_column, config.text_column, config.target_column]].dropna().copy()

    frame.loc[:, config.text_column] = _clean_text_values(frame[config.text_column])
    frame.loc[:, config.target_column] = _clean_text_values(frame[config.target_column])

    frame = frame[
        (frame[config.text_column] != "")
        & (frame[config.target_column] != "")
    ].copy()
    return frame.reset_index(drop=True)


def load_inference_frame(config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.test_path)
    frame = frame[[config.id_column, config.text_column]].dropna().copy()
    frame.loc[:, config.text_column] = _clean_text_values(frame[config.text_column])
    frame = frame[frame[config.text_column] != ""].copy()
    return frame.reset_index(drop=True)


def extract_features_targets(
    frame: pd.DataFrame, text_column: str, target_column: str
) -> tuple[list[str], list[str]]:
    texts = frame[text_column].astype(str).tolist()
    labels = frame[target_column].astype(str).tolist()
    return texts, labels


def extract_texts(frame: pd.DataFrame, text_column: str) -> list[str]:
    return frame[text_column].astype(str).tolist()


def normalize_payload_texts(texts: Iterable[str]) -> list[str]:
    return [str(text).strip() for text in texts]
