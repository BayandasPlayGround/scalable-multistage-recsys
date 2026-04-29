from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app
from amazon_recsys.ml import core


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

    assert bundle.pointer is not None
    assert bundle.pointer.version == "fixture-bundle"
    assert bundle.manifest.bundle_format == "onnx"
    assert Path(bundle.manifest.runtime_bundle_path).name == "runtime_bundle.json"
    assert (bundle_dir / "models" / "ranker.onnx").exists()
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
