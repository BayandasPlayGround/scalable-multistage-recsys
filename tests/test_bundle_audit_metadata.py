"""Round-trip test for the audit-metadata fields persisted alongside each retriever.

Phase 2 added `encoder_name` and `item_text_source_counts` to the in-memory RetrieverArtifacts.
Phase C (this file's subject) makes those fields survive `_write_retriever_artifacts` ->
`_load_retriever_artifacts` so operators can audit a saved bundle without scraping training logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amazon_recsys.ml import bundles, core


def _make_retriever(
    config: core.PipelineConfig,
    *,
    variant: str,
    model_extra: dict[str, object] | None = None,
    metadata_extra: dict[str, object] | None = None,
) -> core.RetrieverArtifacts:
    base_model = {"retriever": variant}
    if model_extra:
        base_model.update(model_extra)
    base_metadata = {"source_alias": "two_tower"}
    if metadata_extra:
        base_metadata.update(metadata_extra)
    return core.RetrieverArtifacts(
        config=config,
        variant=variant,
        model=base_model,
        item_encoder=None,
        user_encoder=None,
        item_embeddings=np.ones((3, 4), dtype=np.float32),
        ann_index_path=None,
        ann_index=None,
        metrics=pd.DataFrame([{"K": 10, "recall": 0.5, "split": "val", "variant": variant}]),
        history={},
        retriever_kind="vector",
        metadata=base_metadata,
    )


def test_blair_audit_metadata_survives_bundle_roundtrip(workspace_dir: Path) -> None:
    config = core.PipelineConfig(base_dir=workspace_dir, run_name="pytest-audit")
    core.ensure_directories(config)

    blair_retriever = _make_retriever(
        config,
        variant="blair_text",
        model_extra={"encoder_name": "hyp1231/blair-roberta-base"},
        metadata_extra={
            "encoder_name": "hyp1231/blair-roberta-base",
            "configured_model_name": "hyp1231/blair-roberta-base",
            "fallback_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 768,
            "batch_size": 64,
            "max_seq_length": 256,
            "serving_query": "history_item_embedding_mean",
            "item_text_columns": ["meta_title", "store", "categories_text", "description_text", "features_text"],
            "item_text_source_counts": {"rich": 2108, "fallback": 0},
        },
    )

    bundle_dir = workspace_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    written = bundles._write_retriever_artifacts({"two_tower": blair_retriever}, bundle_dir)
    metadata_path = bundle_dir / str(written["two_tower"]["metadata"])  # type: ignore[index]
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Persisted JSON contains the curated audit dict and the encoder_name in model.
    assert raw["model"]["encoder_name"] == "hyp1231/blair-roberta-base"
    assert raw["audit_metadata"]["encoder_name"] == "hyp1231/blair-roberta-base"
    assert raw["audit_metadata"]["item_text_source_counts"] == {"rich": 2108, "fallback": 0}
    assert raw["audit_metadata"]["item_text_columns"] == ["meta_title", "store", "categories_text", "description_text", "features_text"]
    assert raw["audit_metadata"]["serving_query"] == "history_item_embedding_mean"
    assert raw["audit_metadata"]["source_alias"] == "two_tower"

    # Round-trip restores them into RetrieverArtifacts.metadata.
    restored = bundles._load_retriever_artifacts(bundle_dir, written, config)
    loaded = restored["two_tower"]
    assert loaded.variant == "blair_text"
    assert loaded.model.get("encoder_name") == "hyp1231/blair-roberta-base"
    assert loaded.metadata["encoder_name"] == "hyp1231/blair-roberta-base"
    assert loaded.metadata["item_text_source_counts"] == {"rich": 2108, "fallback": 0}
    assert loaded.metadata["embedding_dim"] == 768
    assert loaded.metadata["serving_query"] == "history_item_embedding_mean"


def test_legacy_bundle_without_audit_metadata_still_loads(workspace_dir: Path) -> None:
    """A bundle whose retriever metadata.json predates Phase C must still load (no KeyError)."""
    config = core.PipelineConfig(base_dir=workspace_dir, run_name="pytest-legacy")
    core.ensure_directories(config)
    bundle_dir = workspace_dir / "legacy-bundle"
    retriever_dir = bundle_dir / "retrievers" / "content_based"
    retriever_dir.mkdir(parents=True, exist_ok=True)

    item_embeddings_path = retriever_dir / "item_embeddings.npy"
    np.save(item_embeddings_path, np.ones((3, 4), dtype=np.float32))
    metrics_path = retriever_dir / "metrics.parquet"
    pd.DataFrame([{"K": 10, "recall": 0.0, "split": "val", "variant": "content_based"}]).to_parquet(metrics_path, index=False)
    legacy_metadata_path = retriever_dir / "metadata.json"
    legacy_metadata_path.write_text(json.dumps({
        "variant": "content_based",
        "retriever_kind": "vector",
        "history": {},
        "model": {"retriever": "content_based"},
        "item_embeddings": "retrievers/content_based/item_embeddings.npy",
        "metrics": "retrievers/content_based/metrics.parquet",
        "ann_index": None,
        "user_vectors": None,
        # No "audit_metadata" key — simulates a bundle written by an older codebase.
    }))

    payload = {"content_based": {"metadata": "retrievers/content_based/metadata.json"}}
    restored = bundles._load_retriever_artifacts(bundle_dir, payload, config)
    loaded = restored["content_based"]
    assert loaded.variant == "content_based"
    assert "encoder_name" not in loaded.metadata
    assert "item_text_source_counts" not in loaded.metadata


def test_non_blair_retriever_omits_blair_specific_keys(workspace_dir: Path) -> None:
    """latent_cf / content_based don't set BLAIR fields, so the persisted audit dict is empty."""
    config = core.PipelineConfig(base_dir=workspace_dir, run_name="pytest-other")
    core.ensure_directories(config)
    latent = _make_retriever(config, variant="latent_cf")
    bundle_dir = workspace_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    written = bundles._write_retriever_artifacts({"latent_cf": latent}, bundle_dir)
    raw = json.loads((bundle_dir / str(written["latent_cf"]["metadata"])).read_text())  # type: ignore[index]
    # source_alias was the only audit-relevant field on this retriever; the BLAIR-specific
    # fields must not appear because the retriever never set them.
    assert raw["audit_metadata"] == {"source_alias": "two_tower"}
    assert "encoder_name" not in raw["audit_metadata"]
