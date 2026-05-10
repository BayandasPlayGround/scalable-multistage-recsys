from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app
from amazon_recsys.ml import bundles
from amazon_recsys.ml import core
from amazon_recsys.ml.pipelines import pipeline_config_from_settings


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.retrieval
@pytest.mark.ranking
@pytest.mark.serving
def test_training_bundle_round_trip(test_container, train_export_activate) -> None:
    bundle = train_export_activate(test_container, version="fixture-bundle")
    test_container.recommendation_service.refresh()

    recommendations = test_container.recommendation_service.recommend(user_id="u1", top_k=3)
    history = test_container.recommendation_service.get_user_history("u1")
    available_users = test_container.recommendation_service.list_available_users(limit=10, min_history=3)
    model = test_container.recommendation_service.get_active_model()
    summary = test_container.recommendation_service.get_evaluation_summary()
    bundle_dir = Path(bundle.manifest.bundle_dir)
    loaded_bundle = test_container.artifact_store.load_bundle(bundle.manifest)

    assert bundle.pointer is not None
    assert bundle.pointer.version == "fixture-bundle"
    assert bundle.manifest.bundle_format == "onnx"
    assert Path(bundle.manifest.runtime_bundle_path).name == "runtime_bundle.json"
    assert loaded_bundle.serving_index is not None
    assert not loaded_bundle.serving_index.user_summary.empty
    assert not loaded_bundle.serving_index.user_history.empty
    assert loaded_bundle.prepared.interactions.empty
    assert loaded_bundle.prepared.hard_negatives.empty
    assert (bundle_dir / "models" / "ranker.onnx").exists()
    assert (bundle_dir / "data" / "serving_user_summary.parquet").exists()
    assert (bundle_dir / "data" / "serving_user_history.parquet").exists()
    assert (bundle_dir / "data" / "serving_state.json").exists()
    assert not list(bundle_dir.rglob("*.pkl"))
    assert bundle.session.pipeline_config.__class__.__module__ == "amazon_recsys.ml.core"
    assert bundle.session.prepared.__class__.__module__ == "amazon_recsys.ml.core"
    assert 1 <= len(recommendations) <= 3
    assert history
    assert available_users
    assert available_users[0].interaction_count >= 1
    assert model.version == "fixture-bundle"
    assert summary.metric_files
    metric_file_names = {metric.file for metric in summary.metric_files}
    assert "candidate_recall_by_category.csv" in metric_file_names
    assert "candidate_recall_by_history_bucket.csv" in metric_file_names
    assert "candidate_recall_by_source.csv" in metric_file_names
    assert "candidate_recall_by_cold_start_type.csv" in metric_file_names
    assert "candidate_recall_worst_slices.csv" in metric_file_names
    assert "candidate_union_recall_by_category.csv" in metric_file_names
    assert "candidate_union_recall_by_source.csv" in metric_file_names
    assert "candidate_union_recall_by_history_bucket.csv" in metric_file_names
    assert "served_distribution_by_category_price.csv" in metric_file_names

    client = TestClient(create_app(test_container.settings))
    api_response = client.post("/recommend", json={"user_id": "u1", "top_k": 2})
    active_response = client.get("/models/active")
    users_response = client.get("/users", params={"limit": 5, "min_history": 3})
    page_response = client.get("/?user_id=u1&top_k=2")

    assert api_response.status_code == 200
    assert active_response.status_code == 200
    assert users_response.status_code == 200
    assert users_response.json()["items"]
    assert users_response.json()["items"][0]["user_id"] in {"u1", "u2", "u3", "u4"}
    assert active_response.json()["version"] == "fixture-bundle"
    assert "available-users" in page_response.text
    assert 'data-filter-table="qa-results"' in page_response.text
    assert 'data-filter-table="analysis-users"' in page_response.text


@pytest.mark.retrieval
def test_candidate_recall_diagnostics_break_down_categories_and_sources() -> None:
    candidates = pd.DataFrame(
        [
            {"example_id": 1, "label": 1, "rank": 1, "target_source_category": "Automotive", "history_length": 4, "target_price_bucket": "price_low", "from_cooccurrence": 1, "from_latent_cf": 0},
            {"example_id": 1, "label": 0, "rank": 2, "target_source_category": "Automotive", "history_length": 4, "target_price_bucket": "price_low", "from_cooccurrence": 0, "from_latent_cf": 1},
            {"example_id": 2, "label": 0, "rank": 1, "target_source_category": "All_Beauty", "history_length": 18, "target_price_bucket": "price_high", "from_cooccurrence": 1, "from_latent_cf": 0},
        ]
    )
    candidates["cold_start_user_type"] = ["known_user_full_history", "known_user_full_history", "anonymous_no_history"]

    diagnostics = core.candidate_recall_diagnostics(
        candidates,
        split="test",
        variant="hybrid_union",
        stage="candidate_union",
    )

    by_scope = {(row["scope"], row["name"]): row for row in diagnostics.to_dict(orient="records")}
    assert by_scope[("overall", "all")]["hit_rate"] == pytest.approx(0.5)
    assert by_scope[("target_category", "Automotive")]["hit_rate"] == pytest.approx(1.0)
    assert by_scope[("target_category", "All_Beauty")]["hit_rate"] == pytest.approx(0.0)
    assert by_scope[("history_length_bucket", "03-05")]["hit_rate"] == pytest.approx(1.0)
    assert by_scope[("history_length_bucket", "11-25")]["hit_rate"] == pytest.approx(0.0)
    assert by_scope[("target_price_bucket", "price_low")]["positive_recoveries"] == 1
    assert by_scope[("candidate_source", "cooccurrence")]["positive_recoveries"] == 1
    assert by_scope[("cold_start_user_type", "anonymous_no_history")]["hit_rate"] == pytest.approx(0.0)


@pytest.mark.retrieval
def test_history_normalization_accepts_common_sequence_shapes() -> None:
    item_text_matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    expected = np.asarray([0.5, 0.5], dtype=np.float32)

    np.testing.assert_allclose(core._mean_text_profile([1, 2], item_text_matrix), expected)
    np.testing.assert_allclose(core._mean_text_profile((1, 2), item_text_matrix), expected)
    np.testing.assert_allclose(core._mean_text_profile(np.asarray([1, 2]), item_text_matrix), expected)
    np.testing.assert_allclose(core._mean_text_profile([], item_text_matrix), np.zeros((2,), dtype=np.float32))
    np.testing.assert_array_equal(core._pad_history(np.asarray([1, 2, 3]), 2), np.asarray([2, 3], dtype=np.int32))


@pytest.mark.retrieval
def test_category_aware_popularity_backfill_reserves_global_fallback() -> None:
    config = core.PipelineConfig(
        categories=("Automotive", "Industrial_and_Scientific"),
        category_backfill_enabled=True,
        popularity_backfill_k=5,
    )
    split_artifacts = SimpleNamespace(
        config=config,
        train_item_popularity=Counter({101: 100, 102: 90, 201: 80, 301: 70, 302: 60}),
        category_item_popularity={
            "Automotive": Counter({101: 100, 102: 90}),
            "Industrial_and_Scientific": Counter({201: 80}),
        },
    )
    examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "test",
                "user_id": "u1",
                "history_item_idxs": [],
                "target_item_idx": 999,
                "pref_Automotive": 0.5,
                "pref_Industrial_and_Scientific": 0.5,
            }
        ]
    )

    candidates = core.popularity_by_category_candidates(split_artifacts, examples, top_k=5)

    assert set(candidates["item_idx"]).issuperset({101, 201})
    assert len(candidates) == 5


@pytest.mark.retrieval
def test_recency_weighted_cooccurrence_prefers_recent_history() -> None:
    config = core.PipelineConfig(recency_cooccurrence_enabled=True)
    split_artifacts = SimpleNamespace(
        config=config,
        cooccurrence={
            1: Counter({101: 100}),
            2: Counter({202: 60}),
        },
        train_item_popularity=Counter(),
    )
    examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "test",
                "user_id": "u1",
                "history_item_idxs": [1, 2],
                "target_item_idx": 999,
            }
        ]
    )

    candidates = core.item_item_cooccurrence_candidates(split_artifacts, examples, top_k=2)

    assert candidates.iloc[0]["item_idx"] == 202


@pytest.mark.retrieval
def test_source_balanced_union_reserves_popularity_candidates(monkeypatch) -> None:
    config = core.PipelineConfig(
        categories=("Automotive",),
        candidate_union_top_k=10,
        candidate_source_balance_enabled=True,
    )
    prepared = SimpleNamespace(config=config)
    split_artifacts = SimpleNamespace()
    examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "test",
                "user_id": "u1",
                "user_idx": 1,
                "history_item_idxs": [1],
                "target_item_idx": 999,
                "target_parent_asin": "target",
                "target_timestamp": pd.Timestamp("2026-01-01"),
                "target_source_category": "Automotive",
                "history_length": 1,
                "user_interaction_count": 1.0,
                "user_mean_rating": 5.0,
                "user_verified_rate": 1.0,
                "days_since_last": 1.0,
                "avg_days_between": 1.0,
                "pref_Automotive": 1.0,
            }
        ]
    )

    def fake_source_frames(*_args, **_kwargs):
        cooccurrence = pd.DataFrame(
            [
                {"example_id": 1, "split": "test", "user_id": "u1", "item_idx": item_idx, "retrieval_score": 100.0 - rank, "rank": rank, "label": 0, "target_item_idx": 999}
                for rank, item_idx in enumerate(range(100, 120), start=1)
            ]
        )
        popularity = pd.DataFrame(
            [
                {"example_id": 1, "split": "test", "user_id": "u1", "item_idx": 900 + rank, "retrieval_score": 1.0, "rank": 100 + rank, "label": 0, "target_item_idx": 999}
                for rank in range(3)
            ]
        )
        return {
            "cooccurrence": core._normalize_candidate_frame(cooccurrence, "cooccurrence"),
            "latent_cf": core._normalize_candidate_frame(None, "latent_cf"),
            "content_based": core._normalize_candidate_frame(None, "content_based"),
            "two_tower": core._normalize_candidate_frame(None, "two_tower"),
            "popularity": core._normalize_candidate_frame(popularity, "popularity"),
        }

    monkeypatch.setattr(core, "_source_candidate_frames", fake_source_frames)

    union = core.generate_candidate_union(prepared, split_artifacts, {}, examples, top_k=10, inject_target_if_missing=False)

    assert len(union) == 10
    assert int(union["from_popularity"].sum()) >= 1


@pytest.mark.retrieval
def test_source_balanced_union_applies_reserved_latent_quota(monkeypatch) -> None:
    config = core.PipelineConfig(
        categories=("Automotive",),
        candidate_union_top_k=10,
        candidate_source_balance_enabled=True,
    )
    prepared = SimpleNamespace(config=config)
    split_artifacts = SimpleNamespace()
    examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "test",
                "user_id": "u1",
                "user_idx": 1,
                "history_item_idxs": [1],
                "target_item_idx": 999,
                "target_parent_asin": "target",
                "target_timestamp": pd.Timestamp("2026-01-01"),
                "target_source_category": "Automotive",
                "history_length": 1,
                "user_interaction_count": 1.0,
                "user_mean_rating": 5.0,
                "user_verified_rate": 1.0,
                "days_since_last": 1.0,
                "avg_days_between": 1.0,
                "pref_Automotive": 1.0,
            }
        ]
    )

    def frame_for(source: str, start: int, count: int, rank_offset: int = 0) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {"example_id": 1, "split": "test", "user_id": "u1", "item_idx": start + rank, "retrieval_score": 1.0, "rank": rank + rank_offset, "label": 0, "target_item_idx": 999}
                for rank in range(1, count + 1)
            ]
        )
        return core._normalize_candidate_frame(frame, source)

    def fake_source_frames(*_args, **_kwargs):
        return {
            "cooccurrence": frame_for("cooccurrence", 100, 20),
            "latent_cf": frame_for("latent_cf", 300, 5, rank_offset=100),
            "content_based": core._normalize_candidate_frame(None, "content_based"),
            "two_tower": core._normalize_candidate_frame(None, "two_tower"),
            "popularity": core._normalize_candidate_frame(None, "popularity"),
        }

    monkeypatch.setattr(core, "_source_candidate_frames", fake_source_frames)

    union = core.generate_candidate_union(prepared, split_artifacts, {}, examples, top_k=10, inject_target_if_missing=False)

    assert int(union["from_latent_cf"].sum()) >= 2


@pytest.mark.retrieval
def test_ranker_candidate_pruning_preserves_reserved_latent_candidates() -> None:
    config = core.PipelineConfig(
        categories=("Automotive",),
        candidate_source_balance_enabled=True,
        ranker_candidate_top_k=5,
    )
    candidates = pd.DataFrame(
        [
            {
                "example_id": 1,
                "item_idx": item_idx,
                "retrieval_score": 1.0 / rank,
                "union_score": 1.0 / rank,
                "source_count": 1,
                "from_cooccurrence": 1,
                "rank_cooccurrence": rank,
                "from_latent_cf": 0,
                "rank_latent_cf": 99,
                "from_content_based": 0,
                "rank_content_based": 99,
                "from_popularity": 0,
                "rank_popularity": 99,
            }
            for rank, item_idx in enumerate(range(100, 110), start=1)
        ]
        + [
            {
                "example_id": 1,
                "item_idx": 300,
                "retrieval_score": 0.001,
                "union_score": 0.001,
                "source_count": 1,
                "from_cooccurrence": 0,
                "rank_cooccurrence": 99,
                "from_latent_cf": 1,
                "rank_latent_cf": 1,
                "from_content_based": 0,
                "rank_content_based": 99,
                "from_popularity": 0,
                "rank_popularity": 99,
            }
        ]
    )

    pruned = core._prune_ranker_candidates(candidates, config)

    assert len(pruned) == 5
    assert 300 in set(pruned["item_idx"])


@pytest.mark.retrieval
def test_multi_trigger_content_retrieval_uses_recent_item_neighbors(workspace_dir: Path) -> None:
    config = core.PipelineConfig(
        base_dir=workspace_dir,
        categories=("Automotive",),
        vector_retriever_trigger_count=2,
        ann_trees=2,
    )
    item_embeddings = core._normalize_rows(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
                [0.0, 0.98],
            ],
            dtype=np.float32,
        )
    )
    ann_index_path = core.build_ann_index(config, item_embeddings, workspace_dir / "content.ann")
    retriever = SimpleNamespace(
        variant="content_based",
        retriever_kind="vector",
        item_embeddings=item_embeddings,
        ann_index_path=ann_index_path,
        ann_index=core._load_ann_index(item_embeddings.shape[1], ann_index_path),
        metadata={},
    )
    prepared = SimpleNamespace(config=config, item_text_matrix=item_embeddings)
    split_artifacts = SimpleNamespace(config=config, train_seen_map={}, val_seen_map={}, test_seen_map={})
    examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "test",
                "user_id": "u1",
                "user_idx": 1,
                "history_item_idxs": [1, 2],
                "target_item_idx": 4,
                "target_parent_asin": "target",
                "target_timestamp": pd.Timestamp("2026-01-01"),
                "target_source_category": "Automotive",
                "history_length": 2,
                "user_interaction_count": 2.0,
                "user_mean_rating": 5.0,
                "user_verified_rate": 1.0,
                "days_since_last": 1.0,
                "avg_days_between": 1.0,
                "pref_Automotive": 1.0,
            }
        ]
    )

    candidates = core.generate_candidates(prepared, split_artifacts, retriever, examples, top_k=1, inject_target_if_missing=False)

    assert candidates.iloc[0]["item_idx"] == 4


@pytest.mark.retrieval
def test_latent_vector_query_specs_include_known_user_and_recent_triggers() -> None:
    config = core.PipelineConfig(categories=("Automotive",), vector_retriever_trigger_count=1)
    retriever = SimpleNamespace(
        variant="latent_cf",
        item_embeddings=core._normalize_rows(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
        metadata={"user_vectors": core._normalize_rows(np.asarray([[0.5, 0.5]], dtype=np.float32))},
    )
    prepared = SimpleNamespace(config=config, item_text_matrix=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    example = pd.Series({"user_idx": 1, "history_item_idxs": [2]})

    specs = core._weighted_vector_query_specs(prepared, retriever, example)

    assert len(specs) == 3
    np.testing.assert_allclose(specs[0][0], retriever.metadata["user_vectors"][0])
    np.testing.assert_allclose(specs[1][0], retriever.item_embeddings[1])


@pytest.mark.retrieval
def test_candidate_recovery_diagnostics_do_not_inject_targets(monkeypatch) -> None:
    config = core.PipelineConfig(categories=("Automotive",), candidate_union_top_k=5, ranker_candidate_top_k=3)
    prepared = SimpleNamespace(
        config=config,
        item_features=pd.DataFrame([{"item_idx": 1, "source_category": "Automotive", "price": 10.0}]),
    )
    split_artifacts = SimpleNamespace()
    examples = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "test",
                "user_id": "u1",
                "user_idx": 1,
                "history_item_idxs": [1],
                "target_item_idx": 1,
                "target_parent_asin": "A1",
                "target_timestamp": pd.Timestamp("2026-01-01"),
                "target_source_category": "Automotive",
                "history_length": 1,
                "user_interaction_count": 1.0,
                "user_mean_rating": 5.0,
                "user_verified_rate": 1.0,
                "days_since_last": 1.0,
                "avg_days_between": 1.0,
                "pref_Automotive": 1.0,
            }
        ]
    )
    captured: list[bool] = []

    def fake_generate_candidate_union(*_args, **kwargs):
        captured.append(bool(kwargs["inject_target_if_missing"]))
        return core._empty_candidate_frame()

    monkeypatch.setattr(core, "generate_candidate_union", fake_generate_candidate_union)

    core.run_candidate_recovery_diagnostics(
        prepared,
        split_artifacts,
        {},
        examples,
        split="test",
        scenarios=("known_user_full_history",),
    )

    assert captured == [False]


@pytest.mark.retrieval
def test_two_tower_vector_serving_query_uses_history_embedding_mean() -> None:
    retriever = SimpleNamespace(
        variant="two_tower",
        item_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        metadata={},
    )
    examples = pd.DataFrame([{"history_item_idxs": [1, 3]}])

    query = core._vector_retriever_queries(SimpleNamespace(), retriever, examples)
    expected = np.asarray([[1.0, 0.5]], dtype=np.float32)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)

    np.testing.assert_allclose(query, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.retrieval
def test_neural_retriever_is_sanitized_for_portable_export(workspace_dir: Path) -> None:
    ann_index_path = workspace_dir / "two_tower.ann"
    ann_index_path.write_text("placeholder", encoding="utf-8")
    retriever = SimpleNamespace(
        variant="two_tower",
        retriever_kind="neural",
        ann_index=None,
        ann_index_path=ann_index_path,
        item_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        item_encoder=object(),
        user_encoder=object(),
        model=object(),
        metadata={},
    )
    session = SimpleNamespace(
        prepared=SimpleNamespace(item_text_matrix=np.asarray([[1.0]], dtype=np.float32)),
        split_artifacts=object(),
        retrievers={"two_tower": retriever},
        ranker=SimpleNamespace(backend="xgboost"),
    )

    _, _, sanitized_retrievers, _ = bundles._sanitize_runtime_objects(session)
    sanitized = sanitized_retrievers["two_tower"]

    assert sanitized.retriever_kind == "vector"
    assert sanitized.user_encoder is None
    assert sanitized.item_encoder is None
    assert sanitized.model["serving_query"] == "history_item_embedding_mean"
    assert sanitized.metadata["exported_from_retriever_kind"] == "neural"


@pytest.mark.retrieval
def test_quality_profile_raises_debug_sized_candidate_budgets(test_settings, caplog) -> None:
    settings = test_settings.model_copy(update={"run_profile": "quality"})

    with caplog.at_level("WARNING"):
        config = pipeline_config_from_settings(settings)

    assert config.candidate_union_top_k == 500
    assert config.ranker_candidate_top_k == 200
    assert config.cooccurrence_candidate_k == 250
    assert config.latent_cf_candidate_k == 250
    assert config.content_candidate_k == 250
    assert config.popularity_backfill_k == 100
    assert "Candidate budget settings were below the quality profile floor" in caplog.text


@pytest.mark.retrieval
def test_neural_retriever_failures_raise_when_enabled(monkeypatch) -> None:
    prepared = SimpleNamespace(config=SimpleNamespace(enable_neural_retriever=True))
    split_artifacts = object()

    monkeypatch.setattr(core, "train_content_retriever", lambda *_args, **_kwargs: "content")
    monkeypatch.setattr(core, "train_latent_cf_retriever", lambda *_args, **_kwargs: "latent")

    def _fail_neural(*_args, **_kwargs):
        raise RuntimeError("two_tower retriever exploded")

    monkeypatch.setattr(core, "train_retriever", _fail_neural)

    with pytest.raises(RuntimeError, match="two_tower retriever exploded"):
        core.train_retrievers(prepared, split_artifacts)


@pytest.mark.retrieval
def test_latent_cf_retriever_handles_users_removed_by_split_caps(synthetic_workspace: Path) -> None:
    config = core.PipelineConfig(
        base_dir=synthetic_workspace,
        categories=("All_Beauty",),
        run_name="latent-capped-users",
        run_profile="debug",
        seed=42,
        k_core=2,
        show_progress=False,
        train_positive_cap=1,
        split_eval_example_cap=1,
        eval_user_cap=2,
        retrieval_top_k=3,
        latent_cf_components=2,
        ann_trees=2,
        text_max_features=32,
        text_svd_dim=4,
        metadata_download_if_missing=False,
        memory_map_item_text=False,
    )
    config = core.apply_run_profile(config)
    prepared = core.prepare_corpus(config, force_rebuild=True)
    split_artifacts = core.make_splits(prepared)

    training_users = set(core._get_training_interactions(prepared)["user_id"].astype(str).unique())
    split_users = set(split_artifacts.user_id_to_idx)

    assert training_users - split_users

    retriever = core.train_latent_cf_retriever(prepared, split_artifacts)

    assert retriever.variant == "latent_cf"
    assert retriever.metadata["user_vectors"].shape[0] == len(split_artifacts.user_id_to_idx)
