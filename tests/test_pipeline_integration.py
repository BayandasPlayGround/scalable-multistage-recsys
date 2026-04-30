from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app
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
            {"example_id": 1, "label": 1, "rank": 1, "target_source_category": "Automotive", "from_cooccurrence": 1, "from_latent_cf": 0},
            {"example_id": 1, "label": 0, "rank": 2, "target_source_category": "Automotive", "from_cooccurrence": 0, "from_latent_cf": 1},
            {"example_id": 2, "label": 0, "rank": 1, "target_source_category": "All_Beauty", "from_cooccurrence": 1, "from_latent_cf": 0},
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
    assert by_scope[("candidate_source", "cooccurrence")]["positive_recoveries"] == 1


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
