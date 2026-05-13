from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from amazon_recsys.config.settings import MonitoringConfig
from amazon_recsys.domain.entities import ConceptDriftResult, MonitoringSummary, ReferenceProfile
from amazon_recsys.monitoring.metrics import categorical_profile, compute_concept_drift, compute_feature_drifts, numeric_profile
from amazon_recsys.monitoring import reference


def test_reference_recommendation_sample_uses_split_history_items(monkeypatch) -> None:
    prepared = SimpleNamespace(
        item_features=pd.DataFrame(
            [
                {
                    "item_idx": 1,
                    "parent_asin": "A1",
                    "source_category": "All_Beauty",
                    "price": 10.0,
                    "average_rating": 4.5,
                    "train_positive_count": 7,
                },
                {
                    "item_idx": 2,
                    "parent_asin": "A2",
                    "source_category": "Automotive",
                    "price": 20.0,
                    "average_rating": 4.0,
                    "train_positive_count": 3,
                },
            ]
        ),
        item_idx_to_id={1: "A1", 2: "A2"},
    )
    split_artifacts = SimpleNamespace(
        test_examples=pd.DataFrame(
            [
                {
                    "user_id": "user-with-brittle-history",
                    "history_length": 1,
                    "history_item_idxs": np.array([1]),
                }
            ]
        )
    )
    session = SimpleNamespace(prepared=prepared, split_artifacts=split_artifacts, retrievers={}, ranker=None)
    captured: dict[str, object] = {}

    def fake_recommend(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "parent_asin": "A2",
                    "source_category": "Automotive",
                    "price": 20.0,
                    "average_rating": 4.0,
                    "score": 0.9,
                    "candidate_sources": "content_based",
                }
            ]
        )

    monkeypatch.setattr(reference.core, "recommend", fake_recommend)

    frame = reference._recommendation_sample_frame(session, monitored_k=10)

    assert captured["history_items"] == ["A1"]
    assert captured["user_id"] == "user-with-brittle-history"
    assert frame.attrs["attempted_reference_users"] == 1
    assert frame.attrs["skipped_reference_users"] == 0
    assert frame.iloc[0]["item_id"] == "A2"


def test_reference_recommendation_sample_skips_unusable_monitoring_users(monkeypatch) -> None:
    prepared = SimpleNamespace(
        item_features=pd.DataFrame(
            [
                {
                    "item_idx": 1,
                    "parent_asin": "A1",
                    "source_category": "All_Beauty",
                    "price": 10.0,
                    "average_rating": 4.5,
                    "train_positive_count": 7,
                }
            ]
        ),
        item_idx_to_id={1: "A1"},
    )
    split_artifacts = SimpleNamespace(
        test_examples=pd.DataFrame(
            [
                {
                    "user_id": "user-with-stale-history",
                    "history_length": 1,
                    "history_item_idxs": np.array([1]),
                }
            ]
        )
    )
    session = SimpleNamespace(prepared=prepared, split_artifacts=split_artifacts, retrievers={}, ranker=None)

    def fake_recommend(*args, **kwargs):
        raise ValueError("No usable history items were found in the current item catalog.")

    monkeypatch.setattr(reference.core, "recommend", fake_recommend)

    frame = reference._recommendation_sample_frame(session, monitored_k=10)

    assert frame.empty
    assert frame.attrs["attempted_reference_users"] == 1
    assert frame.attrs["skipped_reference_users"] == 1


def test_feature_drifts_raise_alerts_for_large_distribution_shifts() -> None:
    reference_profile = ReferenceProfile(
        bundle_version="bundle-1",
        created_at="2026-04-15T00:00:00+00:00",
        monitored_k=10,
        numeric_features={
            "request_history_length": numeric_profile(pd.Series([1.0] * 40)),
            "known_user_rate": numeric_profile(pd.Series([1.0] * 40)),
            "unseen_user_rate": numeric_profile(pd.Series([0.0] * 40)),
            "unseen_history_item_rate": numeric_profile(pd.Series([0.0] * 40)),
            "served_price": numeric_profile(pd.Series([10.0] * 40)),
            "served_average_rating": numeric_profile(pd.Series([4.7] * 40)),
            "score_distribution": numeric_profile(pd.Series([0.9] * 40)),
        },
        categorical_features={
            "served_category_mix": categorical_profile(["All_Beauty"] * 40),
            "candidate_source_mix": categorical_profile(["cooccurrence"] * 40),
            "item_popularity_bucket": categorical_profile(["low"] * 40),
        },
        offline_metrics={},
        sample_size=40,
        notes={"popularity_thresholds": [1.0, 2.0, 3.0]},
    )
    monitoring = MonitoringConfig(
        enabled=True,
        monitoring_root=Path("."),
        psi_warn=0.10,
        psi_alert=0.25,
        js_warn=0.10,
        js_alert=0.20,
    )
    inference_frame = pd.DataFrame(
        [
            {
                "request_id": f"request-{index // 2}",
                "rank": (index % 2) + 1,
                "request_history_length": 25,
                "is_known_user": False,
                "unseen_user": True,
                "unseen_history_item_rate": 1.0,
                "source_category": "Industrial_and_Scientific",
                "price": 250.0,
                "average_rating": 2.0,
                "score": -1.5,
                "candidate_sources": "content_based",
                "popularity_value": 25.0,
            }
            for index in range(12)
        ]
    )

    results = compute_feature_drifts(reference_profile, inference_frame, monitoring)
    by_name = {result.feature_name: result for result in results}

    assert by_name["served_category_mix"].status == "alert"
    assert by_name["candidate_source_mix"].status == "alert"
    assert by_name["item_popularity_bucket"].status == "alert"
    assert by_name["served_price"].reference_value["proportions"]
    assert by_name["served_price"].current_value["proportions"]
    assert by_name["served_price"].reference_value["bin_labels"]
    assert by_name["served_category_mix"].reference_value["comparison_categories"]
    assert by_name["served_category_mix"].current_value["comparison_categories"]


def test_history_length_profile_counts_values_above_reference_range() -> None:
    reference_profile = ReferenceProfile(
        bundle_version="bundle-1",
        created_at="2026-04-15T00:00:00+00:00",
        monitored_k=10,
        numeric_features={
            "request_history_length": numeric_profile(
                pd.Series([3, 4, 5, 8, 10]),
                bin_edges=[0.0, 1.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0],
            )
        },
        categorical_features={},
        offline_metrics={},
        sample_size=5,
    )
    monitoring = MonitoringConfig(enabled=True, monitoring_root=Path("."), psi_warn=0.10, psi_alert=0.25)
    inference_frame = pd.DataFrame(
        [
            {
                "request_id": f"req-{index}",
                "rank": 1,
                "request_history_length": 120,
                "is_known_user": True,
                "unseen_user": False,
                "unseen_history_item_rate": 0.0,
                "source_category": "Automotive",
                "price": 10.0,
                "average_rating": 4.5,
                "score": 0.0,
                "candidate_sources": "cooccurrence",
                "popularity_value": 1.0,
            }
            for index in range(5)
        ]
    )

    results = compute_feature_drifts(reference_profile, inference_frame, monitoring)
    history = {result.feature_name: result for result in results}["request_history_length"]

    assert history.current_value["proportions"][-2] == pytest.approx(1.0)
    assert history.status == "alert"


def test_concept_drift_requires_consecutive_degraded_windows_for_alert() -> None:
    reference_profile = ReferenceProfile(
        bundle_version="bundle-1",
        created_at="2026-04-15T00:00:00+00:00",
        monitored_k=10,
        numeric_features={},
        categorical_features={},
        offline_metrics={
            "hit_rate_at_10": 1.0,
            "ndcg_at_10": 1.0,
            "mrr_at_10": 1.0,
            "purchase_rate_at_10": 1.0,
        },
        sample_size=10,
    )
    monitoring = MonitoringConfig(
        enabled=True,
        monitoring_root=Path("."),
        min_events_per_window=1,
        performance_drop_warn=0.10,
        performance_drop_alert=0.20,
    )
    inference_frame = pd.DataFrame(
        [
            {
                "request_id": "req-1",
                "user_key": "user-1",
                "item_id": "A1",
                "rank": 1,
                "requested_at": "2026-04-15T00:00:00+00:00",
                "query_mode": "known_user",
            },
            {
                "request_id": "req-2",
                "user_key": "user-2",
                "item_id": "A2",
                "rank": 1,
                "requested_at": "2026-04-15T00:05:00+00:00",
                "query_mode": "known_user",
            },
        ]
    )
    previous_summary = MonitoringSummary(
        bundle_version="bundle-1",
        reference_bundle_version="bundle-1",
        created_at="2026-04-14T00:00:00+00:00",
        window_start="2026-04-13T00:00:00+00:00",
        window_end="2026-04-14T00:00:00+00:00",
        status="alert",
        inference_count=2,
        outcome_count=0,
        concept_drift=ConceptDriftResult(
            status="alert",
            sample_size=2,
            monitored_k=10,
            metrics={
                "hit_rate_at_10": 0.0,
                "ndcg_at_10": 0.0,
                "mrr_at_10": 0.0,
                "purchase_rate_at_10": 0.0,
                "cold_start_hit_rate_at_10": None,
            },
            baseline_metrics=reference_profile.offline_metrics,
            performance_drop=1.0,
            consecutive_degraded_windows=1,
        ),
    )

    result = compute_concept_drift(
        reference_profile,
        inference_frame,
        pd.DataFrame(),
        monitoring,
        previous_summary=previous_summary,
        monitored_k=10,
    )

    assert result.status == "alert"
    assert result.sample_size == 2
    assert result.performance_drop == pytest.approx(1.0)
    assert result.consecutive_degraded_windows == 2


def test_concept_drift_marks_synthetic_outcomes_as_insufficient() -> None:
    reference_profile = ReferenceProfile(
        bundle_version="bundle-1",
        created_at="2026-04-15T00:00:00+00:00",
        monitored_k=10,
        numeric_features={},
        categorical_features={},
        offline_metrics={
            "hit_rate_at_10": 0.5,
            "ndcg_at_10": 0.5,
            "mrr_at_10": 0.5,
            "purchase_rate_at_10": None,
        },
        sample_size=10,
    )
    monitoring = MonitoringConfig(enabled=True, monitoring_root=Path("."), min_events_per_window=1)
    inference_frame = pd.DataFrame(
        [
            {
                "request_id": "req-1",
                "user_key": "user-1",
                "item_id": "A1",
                "rank": 1,
                "requested_at": "2026-04-15T00:00:00+00:00",
                "query_mode": "known_user",
            }
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "occurred_at": "2026-04-15T00:01:00+00:00",
                "user_key": "user-1",
                "item_id": "A1",
                "event_type": "purchase",
                "rating": 5.0,
                "source": "synthetic_inference_replay",
            }
        ]
    )

    result = compute_concept_drift(reference_profile, inference_frame, outcomes, monitoring, monitored_k=10)

    assert result.metrics["hit_rate_at_10"] == pytest.approx(1.0)
    assert result.status == "insufficient_data"
    assert result.notes["synthetic_outcomes"] is True
    assert result.notes["metrics_are_decisionable"] is False
