"""Phase-1 BLAIR retriever tests.

Scope: prove that `train_blair_retriever` integrates with the existing pipeline shape with a
mocked encoder. Does NOT depend on sentence-transformers being installed, does NOT depend on
network access, and does NOT exercise the full training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _import_core():
    return pytest.importorskip("amazon_recsys.ml.core")


def _import_blair():
    return pytest.importorskip("amazon_recsys.ml.retrievers.blair")


def _build_prepared(workspace: Path):
    core = _import_core()
    item_features = pd.DataFrame(
        [
            {"item_idx": 1, "parent_asin": "P1", "title": "Lipstick gloss", "source_category": "All_Beauty"},
            {"item_idx": 2, "parent_asin": "P2", "title": "Car battery charger", "source_category": "Automotive"},
            {"item_idx": 3, "parent_asin": "P3", "title": "Soldering iron kit", "source_category": "Industrial_and_Scientific"},
        ]
    )
    config = core.PipelineConfig(
        base_dir=workspace,
        run_name="pytest-blair-phase1",
        enable_neural_retriever=True,
        neural_retriever_variant="blair_text",
        blair_batch_size=2,
        blair_max_seq_length=32,
    )
    core.ensure_directories(config)
    prepared = core.PreparedArtifacts(
        config=config,
        raw_review_stats=pd.DataFrame(),
        kcore_stats=pd.DataFrame(),
        interactions=pd.DataFrame(),
        hard_negatives=pd.DataFrame(),
        item_features=item_features,
        item_text_matrix=np.zeros((3, 4), dtype=np.float32),
        vectorizer=None,  # type: ignore[arg-type]
        svd=None,  # type: ignore[arg-type]
        item_id_to_idx={"a": 1, "b": 2, "c": 3},
        item_idx_to_id={1: "a", 2: "b", 3: "c"},
        category_to_idx={"All_Beauty": 1, "Automotive": 2, "Industrial_and_Scientific": 3},
    )
    return core, prepared


def test_build_item_texts_falls_back_to_title_and_category_when_no_metadata() -> None:
    blair = _import_blair()
    item_features = pd.DataFrame(
        [
            {"item_idx": 1, "parent_asin": "A", "title": "Lipstick", "source_category": "All_Beauty"},
            {"item_idx": 2, "parent_asin": "B", "title": "Wrench", "source_category": "Automotive"},
            {"item_idx": 3, "parent_asin": "C", "title": None, "source_category": "Automotive"},
        ]
    )
    texts, counts = blair._build_item_texts(item_features, metadata=None)
    assert texts == ["Lipstick All_Beauty", "Wrench Automotive", "Automotive"]
    assert counts == {"rich": 0, "fallback": 3}


def test_build_item_texts_uses_rich_metadata_when_available() -> None:
    blair = _import_blair()
    item_features = pd.DataFrame(
        [
            {"item_idx": 1, "parent_asin": "A", "title": "Lipstick", "source_category": "All_Beauty"},
            {"item_idx": 2, "parent_asin": "B", "title": "Wrench", "source_category": "Automotive"},
            {"item_idx": 3, "parent_asin": "C", "title": "Goggles", "source_category": "Industrial_and_Scientific"},
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "parent_asin": "A",
                "meta_title": "Glossy lipstick",
                "store": "Beauty Co",
                "categories_text": "Beauty | Makeup",
                "description_text": "Long-lasting moisturising lipstick",
                "features_text": "vegan, gluten free",
            },
            # B is intentionally missing from metadata → must fall back to title+category.
            {
                "parent_asin": "C",
                "meta_title": "Safety goggles",
                "store": "",
                "categories_text": "PPE",
                "description_text": "",
                "features_text": "anti-fog",
            },
        ]
    )
    texts, counts = blair._build_item_texts(item_features, metadata=metadata)
    assert texts[0] == "Glossy lipstick Beauty Co Beauty | Makeup Long-lasting moisturising lipstick vegan, gluten free"
    assert texts[1] == "Wrench Automotive", "items missing from metadata fall back to Phase-1 text per row"
    assert texts[2] == "Safety goggles PPE anti-fog"
    assert counts == {"rich": 2, "fallback": 1}


def test_vector_retriever_queries_treats_blair_text_like_two_tower() -> None:
    core = _import_core()
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    examples = pd.DataFrame(
        [
            {"history_item_idxs": [1, 2]},
            {"history_item_idxs": [3]},
            {"history_item_idxs": []},
        ]
    )

    def _make(variant: str) -> core.RetrieverArtifacts:
        return core.RetrieverArtifacts(
            config=None,  # type: ignore[arg-type]
            variant=variant,
            model={"retriever": variant},
            item_encoder=None,
            user_encoder=None,
            item_embeddings=embeddings.copy(),
            ann_index_path=None,
            ann_index=None,
            metrics=pd.DataFrame(),
            history={},
            retriever_kind="vector",
            metadata={},
        )

    two_tower = core._vector_retriever_queries(None, _make("two_tower"), examples)  # type: ignore[arg-type]
    blair_text = core._vector_retriever_queries(None, _make("blair_text"), examples)  # type: ignore[arg-type]
    np.testing.assert_allclose(blair_text, two_tower)


def test_train_blair_retriever_with_mock_encoder(monkeypatch: pytest.MonkeyPatch, workspace_dir: Path) -> None:
    core, prepared = _build_prepared(workspace_dir)
    blair = _import_blair()

    captured_texts: list[list[str]] = []

    class _FakeEncoder:
        def __init__(self) -> None:
            self.max_seq_length = 0

        def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
            captured_texts.append(list(texts))
            assert batch_size == 2
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            return np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]][: len(texts)], dtype=np.float32)

    monkeypatch.setattr(blair, "_load_sentence_transformer", lambda *_a, **_k: (_FakeEncoder(), "fake-encoder", "cpu"))
    monkeypatch.setattr(core, "build_ann_index", lambda *_a, **_k: workspace_dir / "fake.ann")
    monkeypatch.setattr(core, "_load_ann_index", lambda *_a, **_k: object())
    monkeypatch.setattr(
        core,
        "evaluate_retriever",
        lambda *_a, **_k: pd.DataFrame([{"K": 100, "recall": 0.1, "split": "val", "variant": "blair_text"}]),
    )
    # No metadata files exist in this test workspace, so _load_rich_metadata returns None and
    # text construction falls back per row to title+source_category — same as Phase 1 behaviour.
    monkeypatch.setattr(blair, "_load_rich_metadata", lambda *_a, **_k: None)

    artifacts = blair.train_blair_retriever(prepared, object())  # type: ignore[arg-type]

    assert artifacts.variant == "blair_text"
    assert artifacts.retriever_kind == "vector"
    assert artifacts.item_embeddings.shape == (3, 2)
    assert artifacts.item_encoder is None and artifacts.user_encoder is None
    assert artifacts.metadata["encoder_name"] == "fake-encoder"
    assert artifacts.metadata["serving_query"] == "history_item_embedding_mean"
    assert artifacts.metadata["item_text_columns"] == ["title", "source_category"]
    assert artifacts.metadata["item_text_source_counts"] == {"rich": 0, "fallback": 3}
    assert captured_texts and captured_texts[0] == [
        "Lipstick gloss All_Beauty",
        "Car battery charger Automotive",
        "Soldering iron kit Industrial_and_Scientific",
    ]
    metrics_path = prepared.config.eval_dir / "blair_text_retriever_metrics.csv"
    assert metrics_path.exists(), "Phase-1 success criterion: blair_text_retriever_metrics.csv must be written"


def test_train_blair_retriever_passes_rich_text_when_metadata_present(monkeypatch: pytest.MonkeyPatch, workspace_dir: Path) -> None:
    core, prepared = _build_prepared(workspace_dir)
    blair = _import_blair()

    captured_texts: list[list[str]] = []

    class _FakeEncoder:
        def __init__(self) -> None:
            self.max_seq_length = 0

        def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
            captured_texts.append(list(texts))
            return np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]][: len(texts)], dtype=np.float32)

    rich_metadata = pd.DataFrame(
        [
            {
                "parent_asin": "P1",
                "meta_title": "Premium lipstick gloss",
                "store": "Beauty Brand",
                "categories_text": "All Beauty | Makeup",
                "description_text": "Long-lasting glossy formula",
                "features_text": "moisturising",
            },
            {
                "parent_asin": "P2",
                "meta_title": "Car battery charger 12V",
                "store": "AutoCo",
                "categories_text": "Automotive | Electrical",
                "description_text": "Fully automatic charger",
                "features_text": "trickle mode",
            },
            {
                "parent_asin": "P3",
                "meta_title": "Soldering iron 60W",
                "store": "ElectroSupply",
                "categories_text": "Industrial | Tools",
                "description_text": "Adjustable temperature",
                "features_text": "ESD safe",
            },
        ]
    )

    monkeypatch.setattr(blair, "_load_sentence_transformer", lambda *_a, **_k: (_FakeEncoder(), "fake-encoder", "cpu"))
    monkeypatch.setattr(blair, "_load_rich_metadata", lambda *_a, **_k: rich_metadata)
    monkeypatch.setattr(core, "build_ann_index", lambda *_a, **_k: workspace_dir / "fake.ann")
    monkeypatch.setattr(core, "_load_ann_index", lambda *_a, **_k: object())
    monkeypatch.setattr(
        core,
        "evaluate_retriever",
        lambda *_a, **_k: pd.DataFrame([{"K": 100, "recall": 0.1, "split": "val", "variant": "blair_text"}]),
    )

    artifacts = blair.train_blair_retriever(prepared, object())  # type: ignore[arg-type]

    assert artifacts.metadata["item_text_columns"] == list(blair._RICH_METADATA_COLUMNS)
    assert artifacts.metadata["item_text_source_counts"] == {"rich": 3, "fallback": 0}
    assert captured_texts[0][0].startswith("Premium lipstick gloss Beauty Brand"), captured_texts[0][0]
    assert "Long-lasting glossy formula" in captured_texts[0][0]
    assert "moisturising" in captured_texts[0][0]
    assert "ESD safe" in captured_texts[0][2]


def test_train_blair_retriever_respects_item_cap(monkeypatch: pytest.MonkeyPatch, workspace_dir: Path) -> None:
    core, prepared = _build_prepared(workspace_dir)
    prepared.config.blair_item_cap = 2
    blair = _import_blair()
    captured_texts: list[list[str]] = []

    class _FakeEncoder:
        def encode(self, texts, **_kwargs):
            captured_texts.append(list(texts))
            return np.asarray([[1.0, 0.0], [0.0, 1.0]][: len(texts)], dtype=np.float32)

    monkeypatch.setattr(blair, "_load_sentence_transformer", lambda *_a, **_k: (_FakeEncoder(), "fake-encoder", "cpu"))
    monkeypatch.setattr(blair, "_load_rich_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "build_ann_index", lambda *_a, **_k: workspace_dir / "fake.ann")
    monkeypatch.setattr(core, "_load_ann_index", lambda *_a, **_k: object())
    monkeypatch.setattr(core, "evaluate_retriever", lambda *_a, **_k: pd.DataFrame())

    artifacts = blair.train_blair_retriever(prepared, object())  # type: ignore[arg-type]

    assert artifacts.item_embeddings.shape == (2, 2)
    assert artifacts.metadata["item_cap"] == 2
    assert artifacts.metadata["encoded_item_count"] == 2
    assert captured_texts == [["Lipstick gloss All_Beauty", "Car battery charger Automotive"]]


def test_train_blair_retriever_resumes_from_completed_rows(monkeypatch: pytest.MonkeyPatch, workspace_dir: Path) -> None:
    core, prepared = _build_prepared(workspace_dir)
    prepared.config.blair_chunk_rows = 1
    blair = _import_blair()
    embedding_path = prepared.config.model_dir / "blair_text_item_embeddings.npy"
    state_path = prepared.config.model_dir / "blair_text_item_embeddings_state.json"
    existing = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(3, 2))
    existing[0] = np.asarray([1.0, 0.0], dtype=np.float32)
    existing.flush()
    mmap_handle = getattr(existing, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()
    del existing
    state_path.write_text(
        json.dumps(
            {
                "complete": False,
                "completed_rows": 1,
                "item_count": 3,
                "full_item_count": 3,
                "item_cap": None,
                "configured_model_name": prepared.config.blair_model_name,
                "fallback_model_name": prepared.config.blair_fallback_model,
                "encoder_name": "fake-encoder",
                "device": "cpu",
                "max_seq_length": prepared.config.blair_max_seq_length,
                "projection_dim": prepared.config.blair_projection_dim,
                "raw_embedding_dim": 2,
                "embedding_dim": 2,
                "chunk_rows": prepared.config.blair_chunk_rows,
                "seed": prepared.config.seed,
                "item_text_columns": ["title", "source_category"],
                "item_text_source_counts": {"rich": 0, "fallback": 1},
            }
        ),
        encoding="utf-8",
    )
    captured_texts: list[list[str]] = []

    class _FakeEncoder:
        def encode(self, texts, **_kwargs):
            captured_texts.append(list(texts))
            return np.asarray([[0.0, 1.0]][: len(texts)], dtype=np.float32)

    monkeypatch.setattr(blair, "_load_sentence_transformer", lambda *_a, **_k: (_FakeEncoder(), "fake-encoder", "cpu"))
    monkeypatch.setattr(blair, "_load_rich_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "build_ann_index", lambda *_a, **_k: workspace_dir / "fake.ann")
    monkeypatch.setattr(core, "_load_ann_index", lambda *_a, **_k: object())
    monkeypatch.setattr(core, "evaluate_retriever", lambda *_a, **_k: pd.DataFrame())

    artifacts = blair.train_blair_retriever(prepared, object())  # type: ignore[arg-type]

    assert artifacts.item_embeddings.shape == (3, 2)
    assert captured_texts == [["Car battery charger Automotive"], ["Soldering iron kit Industrial_and_Scientific"]]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["complete"] is True
    assert state["completed_rows"] == 3


def test_train_blair_retriever_repairs_missing_state_from_partial_memmap(monkeypatch: pytest.MonkeyPatch, workspace_dir: Path) -> None:
    core, prepared = _build_prepared(workspace_dir)
    prepared.config.blair_chunk_rows = 1
    blair = _import_blair()
    embedding_path = prepared.config.model_dir / "blair_text_item_embeddings.npy"
    state_path = prepared.config.model_dir / "blair_text_item_embeddings_state.json"
    existing = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(3, 2))
    existing[0] = np.asarray([1.0, 0.0], dtype=np.float32)
    existing.flush()
    mmap_handle = getattr(existing, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()
    del existing
    assert not state_path.exists()
    captured_texts: list[list[str]] = []

    class _FakeEncoder:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **_kwargs):
            captured_texts.append(list(texts))
            return np.asarray([[0.0, 1.0]][: len(texts)], dtype=np.float32)

    monkeypatch.setattr(blair, "_load_sentence_transformer", lambda *_a, **_k: (_FakeEncoder(), "fake-encoder", "cpu"))
    monkeypatch.setattr(blair, "_load_rich_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "build_ann_index", lambda *_a, **_k: workspace_dir / "fake.ann")
    monkeypatch.setattr(core, "_load_ann_index", lambda *_a, **_k: object())
    monkeypatch.setattr(core, "evaluate_retriever", lambda *_a, **_k: pd.DataFrame())

    artifacts = blair.train_blair_retriever(prepared, object())  # type: ignore[arg-type]

    assert artifacts.item_embeddings.shape == (3, 2)
    assert captured_texts == [["Car battery charger Automotive"], ["Soldering iron kit Industrial_and_Scientific"]]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["complete"] is True
    assert state["completed_rows"] == 3


def test_load_embedding_state_repairs_stale_completed_rows(workspace_dir: Path) -> None:
    core, prepared = _build_prepared(workspace_dir)
    blair = _import_blair()
    embedding_path = prepared.config.model_dir / "blair_text_item_embeddings.npy"
    state_path = prepared.config.model_dir / "blair_text_item_embeddings_state.json"
    existing = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(3, 2))
    existing[0] = np.asarray([1.0, 0.0], dtype=np.float32)
    existing.flush()
    mmap_handle = getattr(existing, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()
    del existing
    state_path.write_text(
        json.dumps(
            {
                "complete": False,
                "completed_rows": 2,
                "item_count": 3,
                "item_cap": None,
                "configured_model_name": prepared.config.blair_model_name,
                "fallback_model_name": prepared.config.blair_fallback_model,
                "max_seq_length": prepared.config.blair_max_seq_length,
                "projection_dim": prepared.config.blair_projection_dim,
                "raw_embedding_dim": 2,
                "embedding_dim": 2,
                "chunk_rows": prepared.config.blair_chunk_rows,
                "seed": prepared.config.seed,
            }
        ),
        encoding="utf-8",
    )

    state = blair._load_embedding_state(
        state_path,
        embedding_path,
        item_count=3,
        configured_model_name=prepared.config.blair_model_name,
        fallback_model_name=prepared.config.blair_fallback_model,
        max_seq_length=prepared.config.blair_max_seq_length,
        projection_dim=prepared.config.blair_projection_dim,
        chunk_rows=prepared.config.blair_chunk_rows,
        seed=prepared.config.seed,
        item_cap=None,
        repair_enabled=True,
    )

    assert state is not None
    assert state["completed_rows"] == 1
    assert state["complete"] is False


def test_load_sentence_transformer_falls_back_when_primary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    blair = _import_blair()

    fake_module = pytest.importorskip("types").ModuleType("sentence_transformers")  # type: ignore[attr-defined]

    class _Loader:
        instances: list[str] = []

        def __init__(self, name: str, device: str | None = None) -> None:
            _Loader.instances.append(name)
            if name == "boom/fail":
                raise RuntimeError("simulated download failure")
            self.max_seq_length = 0

    fake_module.SentenceTransformer = _Loader  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    encoder, name, device = blair._load_sentence_transformer("boom/fail", "ok/fallback", max_seq_length=128, device="cpu")
    assert name == "ok/fallback"
    assert device == "cpu"
    assert encoder.max_seq_length == 128
    assert _Loader.instances == ["boom/fail", "ok/fallback"]
