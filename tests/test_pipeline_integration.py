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

    assert config.candidate_union_top_k == 200
    assert config.ranker_candidate_top_k == 100
    assert config.cooccurrence_candidate_k == 100
    assert config.latent_cf_candidate_k == 150
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
