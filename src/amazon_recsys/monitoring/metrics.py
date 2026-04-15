from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from amazon_recsys.config.settings import MonitoringConfig
from amazon_recsys.domain.entities import ConceptDriftResult, FeatureDriftResult, MonitoringSummary, ReferenceProfile
from amazon_recsys.monitoring.utils import MONITORED_K, STATUS_RANK, max_status, safe_float, split_candidate_sources


EPSILON = 1e-9


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def numeric_profile(series: pd.Series, *, bin_count: int = 5) -> dict[str, Any]:
    values = _clean_numeric_series(series)
    if values.empty:
        return {
            "bin_edges": [0.0, 1.0],
            "proportions": [1.0],
            "sample_size": 0,
            "mean": None,
            "std": None,
        }
    unique_values = np.unique(values.to_numpy())
    if len(unique_values) == 1:
        anchor = float(unique_values[0])
        bin_edges = [anchor - 0.5, anchor + 0.5]
    else:
        quantiles = np.linspace(0, 1, num=min(bin_count, len(unique_values)) + 1)
        bin_edges = np.quantile(values.to_numpy(), quantiles).astype(float)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([float(values.min()) - 0.5, float(values.max()) + 0.5], dtype=float)
    histogram, resolved_edges = np.histogram(values.to_numpy(), bins=bin_edges)
    proportions = (histogram / histogram.sum()).tolist() if histogram.sum() else [1.0] * len(histogram)
    return {
        "bin_edges": [float(edge) for edge in resolved_edges.tolist()],
        "proportions": [float(value) for value in proportions],
        "sample_size": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
    }


def categorical_profile(values: list[str]) -> dict[str, Any]:
    cleaned = [str(value) for value in values if value not in {None, ""}]
    if not cleaned:
        return {"proportions": {"__empty__": 1.0}, "sample_size": 0}
    series = pd.Series(cleaned, dtype="string")
    counts = series.value_counts(dropna=False)
    total = int(counts.sum())
    return {
        "proportions": {str(key): float(value / total) for key, value in counts.items()},
        "sample_size": total,
    }


def _distribution_with_reference_bins(series: pd.Series, profile: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    values = _clean_numeric_series(series)
    bin_edges = np.asarray(profile.get("bin_edges", [0.0, 1.0]), dtype=float)
    if len(bin_edges) < 2:
        bin_edges = np.array([0.0, 1.0], dtype=float)
    if values.empty:
        counts = np.zeros(len(bin_edges) - 1, dtype=float)
    else:
        counts, _ = np.histogram(values.to_numpy(), bins=bin_edges)
    if counts.sum() <= 0:
        proportions = np.full(len(counts), 1.0 / max(len(counts), 1), dtype=float)
    else:
        proportions = counts / counts.sum()
    return (
        [float(value) for value in proportions.tolist()],
        {
            "sample_size": int(len(values)),
            "mean": float(values.mean()) if not values.empty else None,
            "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0 if len(values) == 1 else None,
            "bin_edges": [float(edge) for edge in bin_edges.tolist()],
        },
    )


def _distribution_with_reference_categories(values: list[str], profile: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    cleaned = [str(value) for value in values if value not in {None, ""}]
    reference_keys = list((profile.get("proportions") or {"__empty__": 1.0}).keys())
    counts = pd.Series(cleaned, dtype="string").value_counts(dropna=False).to_dict() if cleaned else {}
    total = float(sum(int(value) for value in counts.values()))
    if total <= 0:
        current = {key: 0.0 for key in reference_keys}
    else:
        current = {key: float(counts.get(key, 0) / total) for key in reference_keys}
        for key in counts:
            current.setdefault(str(key), float(counts[key] / total))
    return current, {"sample_size": int(total)}


def population_stability_index(reference: list[float], current: list[float]) -> float:
    ref = np.asarray(reference, dtype=float) + EPSILON
    cur = np.asarray(current, dtype=float) + EPSILON
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def jensen_shannon_distance(reference: dict[str, float], current: dict[str, float]) -> float:
    keys = sorted(set(reference.keys()).union(current.keys()))
    ref = np.asarray([float(reference.get(key, 0.0)) for key in keys], dtype=float) + EPSILON
    cur = np.asarray([float(current.get(key, 0.0)) for key in keys], dtype=float) + EPSILON
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    midpoint = 0.5 * (ref + cur)
    divergence = 0.5 * (np.sum(ref * np.log(ref / midpoint)) + np.sum(cur * np.log(cur / midpoint)))
    return float(np.sqrt(max(divergence, 0.0)))


def _status_from_metric(metric_type: str, metric_value: float, monitoring: MonitoringConfig) -> tuple[str, float, float]:
    if metric_type == "psi":
        warn_threshold = float(monitoring.psi_warn)
        alert_threshold = float(monitoring.psi_alert)
    else:
        warn_threshold = float(monitoring.js_warn)
        alert_threshold = float(monitoring.js_alert)
    if metric_value >= alert_threshold:
        return "alert", warn_threshold, alert_threshold
    if metric_value >= warn_threshold:
        return "warn", warn_threshold, alert_threshold
    return "ok", warn_threshold, alert_threshold


def _numeric_bin_labels(bin_edges: list[float] | tuple[float, ...] | np.ndarray) -> list[str]:
    resolved = np.asarray(bin_edges, dtype=float)
    if resolved.size < 2:
        return ["all"]
    labels: list[str] = []
    for left, right in zip(resolved[:-1], resolved[1:]):
        labels.append(f"{left:.1f} to {right:.1f}")
    return labels


def _series_to_tokens(series: pd.Series) -> list[str]:
    tokens: list[str] = []
    for raw_value in series.fillna("").astype(str):
        tokens.extend(split_candidate_sources(raw_value))
    return tokens


def assign_popularity_bucket(values: pd.Series, thresholds: list[float]) -> list[str]:
    cleaned = _clean_numeric_series(values)
    if cleaned.empty:
        return []
    resolved = np.asarray(thresholds, dtype=float)
    if resolved.size < 3:
        resolved = np.asarray([0.0, 0.0, 0.0], dtype=float)
    buckets: list[str] = []
    for value in cleaned.to_numpy(dtype=float):
        if value <= resolved[0]:
            buckets.append("low")
        elif value <= resolved[1]:
            buckets.append("medium")
        elif value <= resolved[2]:
            buckets.append("high")
        else:
            buckets.append("very_high")
    return buckets


def compute_feature_drifts(
    reference_profile: ReferenceProfile,
    inference_frame: pd.DataFrame,
    monitoring: MonitoringConfig,
) -> list[FeatureDriftResult]:
    if inference_frame.empty:
        return []

    request_frame = inference_frame.sort_values(["request_id", "rank"]).drop_duplicates("request_id").copy()
    feature_values: dict[str, pd.Series | list[str]] = {
        "request_history_length": request_frame["request_history_length"],
        "known_user_rate": request_frame["is_known_user"].astype(int),
        "unseen_user_rate": request_frame["unseen_user"].astype(int),
        "unseen_history_item_rate": request_frame["unseen_history_item_rate"],
        "served_category_mix": inference_frame["source_category"].astype("string").fillna("__missing__"),
        "served_price": inference_frame["price"],
        "served_average_rating": inference_frame["average_rating"],
        "score_distribution": inference_frame["score"],
        "candidate_source_mix": _series_to_tokens(inference_frame["candidate_sources"]),
        "item_popularity_bucket": assign_popularity_bucket(
            inference_frame["popularity_value"],
            list(reference_profile.notes.get("popularity_thresholds", [])),
        ),
    }

    results: list[FeatureDriftResult] = []

    for feature_name, profile in reference_profile.numeric_features.items():
        current_series = feature_values.get(feature_name)
        if not isinstance(current_series, pd.Series):
            continue
        current_distribution, current_summary = _distribution_with_reference_bins(current_series, profile)
        metric_value = population_stability_index(list(profile.get("proportions", [])), current_distribution)
        status, warn_threshold, alert_threshold = _status_from_metric("psi", metric_value, monitoring)
        results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                metric_type="psi",
                metric_value=metric_value,
                warn_threshold=warn_threshold,
                alert_threshold=alert_threshold,
                status=status,
                sample_size=int(current_summary["sample_size"]),
                reference_value={
                    "mean": profile.get("mean"),
                    "std": profile.get("std"),
                    "sample_size": profile.get("sample_size", 0),
                    "bin_edges": list(profile.get("bin_edges", [])),
                    "bin_labels": _numeric_bin_labels(list(profile.get("bin_edges", []))),
                    "proportions": list(profile.get("proportions", [])),
                },
                current_value={
                    **current_summary,
                    "bin_labels": _numeric_bin_labels(list(profile.get("bin_edges", []))),
                    "proportions": current_distribution,
                },
            )
        )

    for feature_name, profile in reference_profile.categorical_features.items():
        current_values = feature_values.get(feature_name)
        if isinstance(current_values, pd.Series):
            current_list = [str(value) for value in current_values.fillna("__missing__").astype(str).tolist()]
        else:
            current_list = list(current_values or [])
        current_distribution, current_summary = _distribution_with_reference_categories(current_list, profile)
        metric_value = jensen_shannon_distance(dict(profile.get("proportions", {})), current_distribution)
        status, warn_threshold, alert_threshold = _status_from_metric("js", metric_value, monitoring)
        reference_top = sorted(dict(profile.get("proportions", {})).items(), key=lambda item: item[1], reverse=True)[:3]
        current_top = sorted(current_distribution.items(), key=lambda item: item[1], reverse=True)[:3]
        comparison_keys = []
        for key, _ in reference_top + current_top:
            if key not in comparison_keys:
                comparison_keys.append(key)
        results.append(
            FeatureDriftResult(
                feature_name=feature_name,
                metric_type="js",
                metric_value=metric_value,
                warn_threshold=warn_threshold,
                alert_threshold=alert_threshold,
                status=status,
                sample_size=int(current_summary["sample_size"]),
                reference_value={
                    "top_categories": reference_top,
                    "proportions": dict(profile.get("proportions", {})),
                    "comparison_categories": [
                        {"label": key, "value": float(dict(profile.get("proportions", {})).get(key, 0.0))}
                        for key in comparison_keys
                    ],
                },
                current_value={
                    "top_categories": current_top,
                    "proportions": current_distribution,
                    "comparison_categories": [
                        {"label": key, "value": float(current_distribution.get(key, 0.0))}
                        for key in comparison_keys
                    ],
                },
            )
        )

    return sorted(results, key=lambda item: item.metric_value, reverse=True)


def _is_positive_outcome(frame: pd.DataFrame) -> pd.Series:
    purchase_mask = frame["event_type"].astype(str).str.lower() == "purchase"
    rating_mask = pd.to_numeric(frame.get("rating"), errors="coerce").fillna(0.0) >= 4.0
    return purchase_mask | rating_mask


def _request_level_metrics(
    request_ids: pd.Index,
    positive_ranks: dict[str, list[int]],
    purchase_ranks: dict[str, list[int]],
    request_query_modes: dict[str, str],
    monitored_k: int,
) -> dict[str, float | None]:
    hit_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    purchase_values: list[float] = []
    cold_start_hit_values: list[float] = []

    for request_id in request_ids.astype(str):
        ranks = sorted(rank for rank in positive_ranks.get(request_id, []) if rank <= monitored_k)
        purchase_hits = sorted(rank for rank in purchase_ranks.get(request_id, []) if rank <= monitored_k)
        hit_values.append(1.0 if ranks else 0.0)
        mrr_values.append(1.0 / ranks[0] if ranks else 0.0)
        if ranks:
            gains = np.asarray([1.0 / np.log2(rank + 1.0) for rank in ranks], dtype=float)
            ideal = np.asarray([1.0 / np.log2(index + 2.0) for index in range(min(len(ranks), monitored_k))], dtype=float)
            ndcg_values.append(float(gains.sum() / ideal.sum()) if ideal.sum() > 0 else 0.0)
        else:
            ndcg_values.append(0.0)
        purchase_values.append(float(len(purchase_hits) / monitored_k))
        if request_query_modes.get(request_id) != "known_user":
            cold_start_hit_values.append(1.0 if ranks else 0.0)

    return {
        f"hit_rate_at_{monitored_k}": float(np.mean(hit_values)) if hit_values else None,
        f"ndcg_at_{monitored_k}": float(np.mean(ndcg_values)) if ndcg_values else None,
        f"mrr_at_{monitored_k}": float(np.mean(mrr_values)) if mrr_values else None,
        f"purchase_rate_at_{monitored_k}": float(np.mean(purchase_values)) if purchase_values else None,
        f"cold_start_hit_rate_at_{monitored_k}": float(np.mean(cold_start_hit_values)) if cold_start_hit_values else None,
    }


def compute_concept_drift(
    reference_profile: ReferenceProfile,
    inference_frame: pd.DataFrame,
    outcomes_frame: pd.DataFrame,
    monitoring: MonitoringConfig,
    previous_summary: MonitoringSummary | None = None,
    monitored_k: int = MONITORED_K,
) -> ConceptDriftResult:
    eligible = inference_frame.dropna(subset=["user_key"]).copy()
    if eligible.empty:
        return ConceptDriftResult(status="insufficient_data", sample_size=0, monitored_k=monitored_k)

    eligible["request_id"] = eligible["request_id"].astype(str)
    eligible["requested_at_ts"] = pd.to_datetime(eligible["requested_at"], utc=True, errors="coerce")
    eligible = eligible.dropna(subset=["requested_at_ts"])
    if eligible.empty:
        return ConceptDriftResult(status="insufficient_data", sample_size=0, monitored_k=monitored_k)

    if outcomes_frame.empty:
        metrics = _request_level_metrics(eligible["request_id"].drop_duplicates(), {}, {}, dict(eligible[["request_id", "query_mode"]].drop_duplicates().values), monitored_k)
    else:
        outcomes = outcomes_frame.copy()
        outcomes["occurred_at_ts"] = pd.to_datetime(outcomes["occurred_at"], utc=True, errors="coerce")
        outcomes = outcomes.dropna(subset=["occurred_at_ts"])
        joined = eligible.merge(outcomes, on=["user_key", "item_id"], how="left", suffixes=("", "_outcome"))
        joined["is_within_horizon"] = joined["occurred_at_ts"].notna() & (
            joined["occurred_at_ts"] <= joined["requested_at_ts"] + timedelta(days=int(monitoring.attribution_horizon_days))
        ) & (joined["occurred_at_ts"] >= joined["requested_at_ts"])
        joined = joined[joined["is_within_horizon"]].copy()

        positive = joined[_is_positive_outcome(joined)].copy() if not joined.empty else joined
        purchases = joined[joined["event_type"].astype(str).str.lower() == "purchase"].copy() if not joined.empty else joined
        positive_ranks = (
            positive.groupby("request_id")["rank"].apply(lambda values: sorted({int(value) for value in values if int(value) <= monitored_k})).to_dict()
            if not positive.empty
            else {}
        )
        purchase_ranks = (
            purchases.groupby("request_id")["rank"].apply(lambda values: sorted({int(value) for value in values if int(value) <= monitored_k})).to_dict()
            if not purchases.empty
            else {}
        )
        request_modes = dict(eligible[["request_id", "query_mode"]].drop_duplicates().values)
        metrics = _request_level_metrics(eligible["request_id"].drop_duplicates(), positive_ranks, purchase_ranks, request_modes, monitored_k)

    baseline_metrics = {
        f"hit_rate_at_{monitored_k}": reference_profile.offline_metrics.get(f"hit_rate_at_{monitored_k}"),
        f"ndcg_at_{monitored_k}": reference_profile.offline_metrics.get(f"ndcg_at_{monitored_k}"),
        f"mrr_at_{monitored_k}": reference_profile.offline_metrics.get(f"mrr_at_{monitored_k}"),
        f"purchase_rate_at_{monitored_k}": reference_profile.offline_metrics.get(f"purchase_rate_at_{monitored_k}"),
        f"cold_start_hit_rate_at_{monitored_k}": None,
    }
    previous_metrics = previous_summary.concept_drift.metrics if previous_summary is not None else {}

    deltas: dict[str, float | None] = {}
    relative_drops: list[float] = []
    for metric_name, current_value in metrics.items():
        baseline_value = safe_float(baseline_metrics.get(metric_name))
        current_numeric = safe_float(current_value)
        deltas[metric_name] = None if current_numeric is None or baseline_value is None else current_numeric - baseline_value
        if baseline_value and current_numeric is not None and baseline_value > 0:
            relative_drops.append(max((baseline_value - current_numeric) / baseline_value, 0.0))

    performance_drop = float(max(relative_drops)) if relative_drops else 0.0
    sample_count = int(eligible["request_id"].nunique())

    previous_performance_drop = previous_summary.concept_drift.performance_drop if previous_summary is not None else 0.0
    previous_consecutive = previous_summary.concept_drift.consecutive_degraded_windows if previous_summary is not None else 0
    degraded_now = sample_count >= int(monitoring.min_events_per_window) and performance_drop >= float(monitoring.performance_drop_warn)
    degraded_previous = previous_performance_drop >= float(monitoring.performance_drop_warn)
    consecutive = previous_consecutive + 1 if degraded_now and degraded_previous else 1 if degraded_now else 0

    if sample_count < int(monitoring.min_events_per_window):
        status = "insufficient_data"
    elif performance_drop >= float(monitoring.performance_drop_alert) and previous_performance_drop >= float(monitoring.performance_drop_alert):
        status = "alert"
    elif performance_drop >= float(monitoring.performance_drop_warn) and previous_performance_drop >= float(monitoring.performance_drop_warn):
        status = "warn"
    else:
        status = "ok"

    return ConceptDriftResult(
        status=status,
        sample_size=sample_count,
        monitored_k=monitored_k,
        metrics={key: safe_float(value) for key, value in metrics.items()},
        baseline_metrics={key: safe_float(value) for key, value in baseline_metrics.items()},
        previous_metrics={key: safe_float(value) for key, value in previous_metrics.items()},
        deltas=deltas,
        performance_drop=performance_drop,
        consecutive_degraded_windows=consecutive,
    )


def summary_status(feature_drifts: list[FeatureDriftResult], concept_drift: ConceptDriftResult) -> str:
    statuses = [result.status for result in feature_drifts]
    statuses.append(concept_drift.status)
    return max_status(*statuses)


def feature_results_frame(results: list[FeatureDriftResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(result) for result in results])


def concept_result_frame(result: ConceptDriftResult) -> pd.DataFrame:
    payload = asdict(result)
    metrics = payload.pop("metrics", {})
    payload["status_rank"] = STATUS_RANK.get(result.status, 0)
    return pd.DataFrame([{**payload, **metrics}])
