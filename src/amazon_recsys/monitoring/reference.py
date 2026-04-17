from __future__ import annotations

from pathlib import Path

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import ReferenceProfile, utcnow_iso
from amazon_recsys.ml import core
from amazon_recsys.ml.pipelines import TrainingSession
from amazon_recsys.monitoring.metrics import MONITORED_K, assign_popularity_bucket, categorical_profile, numeric_profile
from amazon_recsys.monitoring.utils import safe_float, split_candidate_sources


def _load_ranker_test_metrics(eval_dir: Path, monitored_k: int) -> dict[str, float | None]:
    metric_files = sorted(eval_dir.glob("*ranker_metrics.csv"))
    if not metric_files:
        metric_files = sorted(eval_dir.glob("*_metrics.csv"))
    for csv_path in metric_files:
        frame = pd.read_csv(csv_path)
        if "split" not in frame.columns:
            continue
        subset = frame[frame["split"].astype(str) == "test"].copy()
        if subset.empty:
            continue
        if "stage" in subset.columns:
            ranker_subset = subset[subset["stage"].astype(str) == "ranker"].copy()
            if not ranker_subset.empty:
                subset = ranker_subset
        row = subset[subset["K"].astype(int) == int(monitored_k)]
        if row.empty:
            row = subset.sort_values("K").head(1)
        if row.empty:
            continue
        metric_row = row.iloc[0]
        return {
            f"hit_rate_at_{monitored_k}": safe_float(metric_row.get("recall")),
            f"ndcg_at_{monitored_k}": safe_float(metric_row.get("ndcg")),
            f"mrr_at_{monitored_k}": safe_float(metric_row.get("mrr")),
            f"purchase_rate_at_{monitored_k}": None,
        }
    return {
        f"hit_rate_at_{monitored_k}": None,
        f"ndcg_at_{monitored_k}": None,
        f"mrr_at_{monitored_k}": None,
        f"purchase_rate_at_{monitored_k}": None,
    }


def _resolve_popularity_series(item_features: pd.DataFrame) -> pd.Series:
    if "train_positive_count" in item_features.columns:
        return pd.to_numeric(item_features["train_positive_count"], errors="coerce").fillna(0.0)
    return pd.to_numeric(item_features.get("rating_number"), errors="coerce").fillna(0.0)


def _popularity_thresholds(item_features: pd.DataFrame) -> list[float]:
    popularity = _resolve_popularity_series(item_features)
    if popularity.empty:
        return [0.0, 0.0, 0.0]
    quantiles = popularity.quantile([0.25, 0.5, 0.75]).tolist()
    return [float(value) for value in quantiles]


def _recommendation_sample_frame(session: TrainingSession, monitored_k: int, user_cap: int = 200) -> pd.DataFrame:
    prepared = session.prepared
    split_artifacts = session.split_artifacts
    item_frame = prepared.item_features[["parent_asin", "source_category", "price", "average_rating"]].copy()
    item_frame["popularity_value"] = _resolve_popularity_series(prepared.item_features).tolist()
    item_frame = item_frame.rename(columns={"parent_asin": "item_id"})

    test_examples = split_artifacts.test_examples.copy()
    if test_examples.empty:
        return pd.DataFrame()
    users = (
        test_examples[["user_id", "history_length"]]
        .drop_duplicates("user_id")
        .sort_values("history_length", ascending=False)
        .head(user_cap)
    )

    rows: list[dict[str, object]] = []
    for user_row in users.to_dict(orient="records"):
        frame = core.recommend(
            prepared,
            split_artifacts,
            session.retrievers,
            ranker=session.ranker,
            user_id=str(user_row["user_id"]),
            top_k=monitored_k,
        )
        if frame.empty:
            continue
        frame = frame.rename(columns={"parent_asin": "item_id"}).merge(item_frame[["item_id", "popularity_value"]], on="item_id", how="left")
        frame["history_length"] = int(user_row.get("history_length", 0))
        frame["is_known_user"] = True
        frame["unseen_user"] = False
        frame["unseen_history_item_rate"] = 0.0
        rows.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_reference_profile(
    settings: AppSettings,
    session: TrainingSession,
    bundle_version: str,
    monitored_k: int = MONITORED_K,
) -> ReferenceProfile:
    prepared = session.prepared
    split_artifacts = session.split_artifacts
    request_examples = (
        split_artifacts.test_examples[["user_id", "history_length"]]
        .drop_duplicates("user_id")
        .copy()
    )
    request_examples["history_length"] = pd.to_numeric(request_examples["history_length"], errors="coerce").fillna(0).astype(int)
    recommendation_frame = _recommendation_sample_frame(session, monitored_k=monitored_k)
    popularity_thresholds = _popularity_thresholds(prepared.item_features)

    numeric_features = {
        "request_history_length": numeric_profile(request_examples["history_length"]),
        "known_user_rate": numeric_profile(pd.Series([1.0] * len(request_examples), dtype="float64")),
        "unseen_user_rate": numeric_profile(pd.Series([0.0] * len(request_examples), dtype="float64")),
        "unseen_history_item_rate": numeric_profile(pd.Series([0.0] * len(request_examples), dtype="float64")),
        "served_price": numeric_profile(recommendation_frame["price"] if not recommendation_frame.empty else pd.Series(dtype="float64")),
        "served_average_rating": numeric_profile(recommendation_frame["average_rating"] if not recommendation_frame.empty else pd.Series(dtype="float64")),
        "score_distribution": numeric_profile(
            recommendation_frame["score"] if "score" in recommendation_frame.columns else recommendation_frame.get("retrieval_score", pd.Series(dtype="float64"))
        ),
    }
    categorical_features = {
        "served_category_mix": categorical_profile(
            recommendation_frame["source_category"].astype(str).tolist() if not recommendation_frame.empty else []
        ),
        "candidate_source_mix": categorical_profile(
            [token for raw in recommendation_frame.get("candidate_sources", pd.Series(dtype="string")).fillna("").astype(str).tolist() for token in split_candidate_sources(raw)]
            if not recommendation_frame.empty
            else []
        ),
        "item_popularity_bucket": categorical_profile(
            assign_popularity_bucket(
                recommendation_frame["popularity_value"] if not recommendation_frame.empty else pd.Series(dtype="float64"),
                popularity_thresholds,
            )
        ),
    }

    offline_metrics = _load_ranker_test_metrics(Path(session.pipeline_config.eval_dir), monitored_k=monitored_k)
    return ReferenceProfile(
        bundle_version=bundle_version,
        created_at=utcnow_iso(),
        monitored_k=monitored_k,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        offline_metrics=offline_metrics,
        sample_size=int(len(recommendation_frame)),
        notes={
            "request_sample_size": int(len(request_examples)),
            "served_item_sample_size": int(len(recommendation_frame)),
            "popularity_thresholds": popularity_thresholds,
            "monitoring_root": str(settings.monitoring.monitoring_root),
        },
    )
