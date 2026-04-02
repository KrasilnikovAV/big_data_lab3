from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import ModelConfig, VectorizerConfig


def build_pipeline(model_config: ModelConfig, vectorizer_config: VectorizerConfig) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=vectorizer_config.lowercase,
        ngram_range=(vectorizer_config.ngram_min, vectorizer_config.ngram_max),
        min_df=vectorizer_config.min_df,
        max_df=vectorizer_config.max_df,
        sublinear_tf=vectorizer_config.sublinear_tf,
    )

    if model_config.algorithm == "linear_svc":
        classifier = LinearSVC(C=model_config.c)
    elif model_config.algorithm == "logistic_regression":
        classifier = LogisticRegression(
            C=model_config.c,
            max_iter=model_config.max_iter,
            n_jobs=model_config.n_jobs,
        )
    else:
        raise ValueError(
            "Unsupported algorithm. Use one of: linear_svc, logistic_regression."
        )

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])
