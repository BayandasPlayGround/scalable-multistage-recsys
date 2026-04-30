from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest, EvaluationSummary, RuntimeBundle
from amazon_recsys.ml import core
from amazon_recsys.ml.onnx import ONNXRankerPredictor, export_xgboost_ranker_to_onnx

if TYPE_CHECKING:
    from amazon_recsys.ml.pipelines import TrainingSession


ONNX_BUNDLE_FORMAT = "onnx"
ONNX_BUNDLE_SCHEMA_VERSION = 2


def generate_bundle_version(run_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_name}-{stamp}"


def build_bundle_manifest(
    settings: AppSettings,
    session: TrainingSession,
    version: str,
    bundle_dir: Path,
) -> BundleManifest:
    manifest_path = (bundle_dir / "manifest.json").resolve()
    runtime_bundle_path = (bundle_dir / "runtime_bundle.json").resolve()
    evaluation_summary_path = (bundle_dir / "evaluation_summary.json").resolve()
    return BundleManifest(
        version=version,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        manifest_path=str(manifest_path),
        bundle_dir=str(bundle_dir.resolve()),
        runtime_bundle_path=str(runtime_bundle_path),
        evaluation_summary_path=str(evaluation_summary_path),
        run_name=settings.training.run_name,
        run_profile=settings.training.run_profile,
        model_backend=settings.ranking.backend,
        bundle_format=ONNX_BUNDLE_FORMAT,
        retriever_variants=sorted(session.retrievers.keys()),
        notes={
            "workspace_root": str(settings.workspace_root),
            "legacy_workspace_root": str(settings.legacy_workspace_root),
            "legacy_artifact_root": str(settings.legacy_artifact_root),
            "mlflow_tracking_enabled": bool(session.mlflow is not None),
            "mlflow_tracking_uri": session.mlflow.tracking_uri if session.mlflow is not None else None,
            "mlflow_experiment_name": session.mlflow.experiment_name if session.mlflow is not None else None,
            "mlflow_run_id": session.mlflow.run_id if session.mlflow is not None else None,
        },
    )


def _sanitize_runtime_objects(
    session: TrainingSession,
) -> tuple[core.PreparedArtifacts, core.SplitArtifacts, dict[str, core.RetrieverArtifacts], core.RankerArtifacts]:
    prepared = session.prepared
    if hasattr(prepared, "item_text_matrix"):
        prepared.item_text_matrix = np.asarray(prepared.item_text_matrix)

    retrievers = dict(session.retrievers)
    for retriever in retrievers.values():
        retriever.ann_index = None
        if getattr(retriever, "retriever_kind", "vector") != "vector":
            raise ValueError("ONNX bundle export currently supports classical/vector retrievers only.")

    ranker = session.ranker
    if getattr(ranker, "backend", "xgboost") != "xgboost":
        raise ValueError("ONNX bundle export currently supports backend='xgboost' only.")

    return prepared, session.split_artifacts, retrievers, ranker


def build_runtime_bundle(session: TrainingSession, manifest: BundleManifest) -> RuntimeBundle:
    prepared, split_artifacts, retrievers, ranker = _sanitize_runtime_objects(session)
    return RuntimeBundle(
        manifest=manifest,
        prepared=prepared,
        split_artifacts=split_artifacts,
        retrievers=retrievers,
        ranker=ranker,
        serving_index=core.build_serving_index(prepared, split_artifacts),
        evaluation_summary=session.evaluation_summary,
        is_mock=False,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def _read_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Counter):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _pipeline_config_payload(config: core.PipelineConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["base_dir"] = str(config.base_dir)
    payload["categories"] = list(config.categories)
    payload["retriever_hidden_dims"] = list(config.retriever_hidden_dims)
    payload["ranker_hidden_dims"] = list(config.ranker_hidden_dims)
    return payload


def _load_pipeline_config(payload: dict[str, object]) -> core.PipelineConfig:
    normalized = dict(payload)
    for key in ("categories", "retriever_hidden_dims", "ranker_hidden_dims"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return core.PipelineConfig(**normalized)


def _counter_payload(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _counter_from_payload(payload: object, *, key_type: type = int) -> Counter:
    if not isinstance(payload, dict):
        return Counter()
    return Counter({key_type(key): int(value) for key, value in payload.items()})


def _seen_map_payload(value: dict[str, set[int]]) -> dict[str, list[int]]:
    return {str(key): sorted(int(item) for item in items) for key, items in value.items()}


def _seen_map_from_payload(value: object) -> dict[str, set[int]]:
    if not isinstance(value, dict):
        return {}
    return {str(key): {int(item) for item in items} for key, items in value.items()}


def _hard_negative_payload(value: dict[str, list[tuple[int, int]]]) -> dict[str, list[list[int]]]:
    return {
        str(key): [[int(timestamp), int(item_idx)] for timestamp, item_idx in rows]
        for key, rows in value.items()
    }


def _hard_negative_from_payload(value: object) -> dict[str, list[tuple[int, int]]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): [(int(row[0]), int(row[1])) for row in rows]
        for key, rows in value.items()
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _empty_interactions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "parent_asin",
            "timestamp",
            "timestamp_dt",
            "rating",
            "verified_purchase",
            "item_idx",
            "source_category",
        ]
    )


def _empty_hard_negatives_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["user_id", "parent_asin", "timestamp", "timestamp_dt", "item_idx"])


def _empty_examples_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "example_id",
            "split",
            "user_id",
            "target_item_idx",
            "target_parent_asin",
            "target_source_category",
            "target_timestamp",
            "history_item_idxs",
            "history_length",
            "user_interaction_count",
            "user_mean_rating",
            "user_verified_rate",
            "days_since_last",
            "avg_days_between",
            "user_idx",
        ]
    )


def _write_serving_index(serving_index: core.ServingIndex | None, bundle_dir: Path) -> dict[str, object] | None:
    if serving_index is None:
        return None
    data_dir = bundle_dir / "data"
    paths = {
        "user_summary": data_dir / "serving_user_summary.parquet",
        "user_history": data_dir / "serving_user_history.parquet",
    }
    _write_frame(paths["user_summary"], serving_index.user_summary)
    _write_frame(paths["user_history"], serving_index.user_history)
    return {"paths": {key: _relative(path, bundle_dir) for key, path in paths.items()}}


def _load_serving_index(bundle_dir: Path, payload: dict[str, object] | None) -> core.ServingIndex | None:
    if not payload:
        return None
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return None
    user_summary = _read_frame(bundle_dir / str(paths["user_summary"]))
    user_history = _read_frame(bundle_dir / str(paths["user_history"]))
    if "user_id" in user_summary.columns:
        user_summary["user_id"] = user_summary["user_id"].astype(str)
    if "user_id" in user_history.columns:
        user_history["user_id"] = user_history["user_id"].astype(str)
    return core.ServingIndex(user_summary=user_summary, user_history=user_history)


def _write_prepared_artifacts(prepared: core.PreparedArtifacts, bundle_dir: Path) -> dict[str, object]:
    data_dir = bundle_dir / "data"
    paths = {
        "config": data_dir / "pipeline_config.json",
        "raw_review_stats": data_dir / "raw_review_stats.parquet",
        "kcore_stats": data_dir / "kcore_stats.parquet",
        "interactions": data_dir / "interactions.parquet",
        "hard_negatives": data_dir / "hard_negatives.parquet",
        "item_features": data_dir / "item_features.parquet",
        "item_text_matrix": data_dir / "item_text_matrix.npy",
    }
    _write_json(paths["config"], _pipeline_config_payload(prepared.config))
    _write_frame(paths["raw_review_stats"], prepared.raw_review_stats)
    _write_frame(paths["kcore_stats"], prepared.kcore_stats)
    _write_frame(paths["interactions"], prepared.interactions)
    _write_frame(paths["hard_negatives"], prepared.hard_negatives)
    _write_frame(paths["item_features"], prepared.item_features)
    np.save(paths["item_text_matrix"], np.asarray(prepared.item_text_matrix, dtype=np.float32))
    return {
        "paths": {key: _relative(path, bundle_dir) for key, path in paths.items()},
        "item_id_to_idx": {str(key): int(value) for key, value in prepared.item_id_to_idx.items()},
        "item_idx_to_id": {str(key): str(value) for key, value in prepared.item_idx_to_id.items()},
        "category_to_idx": {str(key): int(value) for key, value in prepared.category_to_idx.items()},
    }


def _load_prepared_artifacts(
    bundle_dir: Path,
    payload: dict[str, object],
    *,
    serving_mode: bool = False,
) -> core.PreparedArtifacts:
    paths = payload["paths"]
    assert isinstance(paths, dict)
    config = _load_pipeline_config(_read_json(bundle_dir / str(paths["config"])))
    item_id_to_idx_payload = payload.get("item_id_to_idx", {})
    item_idx_to_id_payload = payload.get("item_idx_to_id", {})
    category_to_idx_payload = payload.get("category_to_idx", {})
    return core.PreparedArtifacts(
        config=config,
        raw_review_stats=_read_frame(bundle_dir / str(paths["raw_review_stats"])),
        kcore_stats=_read_frame(bundle_dir / str(paths["kcore_stats"])),
        interactions=_empty_interactions_frame() if serving_mode else _read_frame(bundle_dir / str(paths["interactions"])),
        hard_negatives=_empty_hard_negatives_frame() if serving_mode else _read_frame(bundle_dir / str(paths["hard_negatives"])),
        item_features=_read_frame(bundle_dir / str(paths["item_features"])),
        item_text_matrix=np.load(bundle_dir / str(paths["item_text_matrix"]), mmap_mode="r" if config.memory_map_item_text else None),
        vectorizer=None,
        svd=None,
        item_id_to_idx={str(key): int(value) for key, value in dict(item_id_to_idx_payload).items()},
        item_idx_to_id={int(key): str(value) for key, value in dict(item_idx_to_id_payload).items()},
        category_to_idx={str(key): int(value) for key, value in dict(category_to_idx_payload).items()},
    )


def _compact_counter(counter: Counter, limit: int | None = None) -> dict[str, object]:
    items = counter.most_common(limit) if limit is not None else counter.items()
    return _counter_payload(Counter(dict(items)))


def _write_split_artifacts(
    split_artifacts: core.SplitArtifacts,
    bundle_dir: Path,
    serving_index: core.ServingIndex | None = None,
) -> dict[str, object]:
    data_dir = bundle_dir / "data"
    paths = {
        "train_examples": data_dir / "train_examples.parquet",
        "val_examples": data_dir / "val_examples.parquet",
        "test_examples": data_dir / "test_examples.parquet",
        "split_state": bundle_dir / "split_state.json",
        "serving_state": data_dir / "serving_state.json",
    }
    _write_frame(paths["train_examples"], split_artifacts.train_examples)
    _write_frame(paths["val_examples"], split_artifacts.val_examples)
    _write_frame(paths["test_examples"], split_artifacts.test_examples)
    state = {
        "user_id_to_idx": {str(key): int(value) for key, value in split_artifacts.user_id_to_idx.items()},
        "user_idx_to_id": {str(key): str(value) for key, value in split_artifacts.user_idx_to_id.items()},
        "train_seen_map": _seen_map_payload(split_artifacts.train_seen_map),
        "val_seen_map": _seen_map_payload(split_artifacts.val_seen_map),
        "test_seen_map": _seen_map_payload(split_artifacts.test_seen_map),
        "train_item_popularity": _counter_payload(split_artifacts.train_item_popularity),
        "category_item_popularity": {
            str(category): _counter_payload(counter)
            for category, counter in split_artifacts.category_item_popularity.items()
        },
        "cooccurrence": {
            str(item_idx): _counter_payload(counter)
            for item_idx, counter in split_artifacts.cooccurrence.items()
        },
        "hard_negative_history": _hard_negative_payload(split_artifacts.hard_negative_history),
    }
    _write_json(paths["split_state"], state)
    serving_user_ids: set[str] = set()
    serving_history_items: set[int] = set()
    if serving_index is not None:
        if "user_id" in serving_index.user_summary.columns:
            serving_user_ids = set(serving_index.user_summary["user_id"].astype(str))
        if "item_idx" in serving_index.user_history.columns:
            serving_history_items = {
                int(value)
                for value in pd.to_numeric(serving_index.user_history["item_idx"], errors="coerce").dropna().tolist()
            }
    compact_cooccurrence_k = max(
        int(split_artifacts.config.cooccurrence_candidate_k),
        int(split_artifacts.config.candidate_union_top_k),
        50,
    )
    serving_state = {
        "user_id_to_idx": {
            str(user_id): int(user_idx)
            for user_id, user_idx in split_artifacts.user_id_to_idx.items()
            if not serving_user_ids or str(user_id) in serving_user_ids
        },
        "user_idx_to_id": {
            str(user_idx): str(user_id)
            for user_idx, user_id in split_artifacts.user_idx_to_id.items()
            if not serving_user_ids or str(user_id) in serving_user_ids
        },
        "train_seen_map": {},
        "val_seen_map": {},
        "test_seen_map": {},
        "train_item_popularity": _counter_payload(split_artifacts.train_item_popularity),
        "category_item_popularity": {
            str(category): _counter_payload(counter)
            for category, counter in split_artifacts.category_item_popularity.items()
        },
        "cooccurrence": {
            str(item_idx): _compact_counter(counter, compact_cooccurrence_k)
            for item_idx, counter in split_artifacts.cooccurrence.items()
            if int(item_idx) in serving_history_items
        },
        "hard_negative_history": {},
    }
    _write_json(paths["serving_state"], serving_state)
    return {"paths": {key: _relative(path, bundle_dir) for key, path in paths.items()}}


def _load_split_artifacts(
    bundle_dir: Path,
    payload: dict[str, object],
    config: core.PipelineConfig,
    *,
    serving_mode: bool = False,
) -> core.SplitArtifacts:
    paths = payload["paths"]
    assert isinstance(paths, dict)
    state_key = "serving_state" if serving_mode and "serving_state" in paths else "split_state"
    state = _read_json(bundle_dir / str(paths[state_key]))
    user_id_to_idx = {str(key): int(value) for key, value in dict(state["user_id_to_idx"]).items()}
    user_idx_to_id = {int(key): str(value) for key, value in dict(state["user_idx_to_id"]).items()}
    return core.SplitArtifacts(
        config=config,
        train_examples=_empty_examples_frame() if serving_mode else _read_frame(bundle_dir / str(paths["train_examples"])),
        val_examples=_read_frame(bundle_dir / str(paths["val_examples"])),
        test_examples=_read_frame(bundle_dir / str(paths["test_examples"])),
        user_id_to_idx=user_id_to_idx,
        user_idx_to_id=user_idx_to_id,
        train_seen_map=_seen_map_from_payload(state.get("train_seen_map", {})),
        val_seen_map=_seen_map_from_payload(state.get("val_seen_map", {})),
        test_seen_map=_seen_map_from_payload(state.get("test_seen_map", {})),
        train_item_popularity=_counter_from_payload(state.get("train_item_popularity", {})),
        category_item_popularity={
            str(category): _counter_from_payload(counter_payload)
            for category, counter_payload in dict(state.get("category_item_popularity", {})).items()
        },
        cooccurrence={
            int(item_idx): _counter_from_payload(counter_payload)
            for item_idx, counter_payload in dict(state.get("cooccurrence", {})).items()
        },
        hard_negative_history=_hard_negative_from_payload(state.get("hard_negative_history", {})),
    )


def _write_retriever_artifacts(
    retrievers: dict[str, core.RetrieverArtifacts],
    bundle_dir: Path,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, retriever in retrievers.items():
        retriever_dir = bundle_dir / "retrievers" / name
        retriever_dir.mkdir(parents=True, exist_ok=True)
        item_embeddings_path = retriever_dir / "item_embeddings.npy"
        metrics_path = retriever_dir / "metrics.parquet"
        metadata_path = retriever_dir / "metadata.json"
        np.save(item_embeddings_path, np.asarray(retriever.item_embeddings, dtype=np.float32))
        _write_frame(metrics_path, retriever.metrics)

        ann_index_path: Path | None = None
        if retriever.ann_index_path is not None:
            source_index = Path(retriever.ann_index_path)
            if source_index.exists():
                ann_index_path = retriever_dir / "item_index.ann"
                shutil.copyfile(source_index, ann_index_path)

        user_vectors_path: Path | None = None
        user_vectors = retriever.metadata.get("user_vectors")
        if user_vectors is not None:
            user_vectors_path = retriever_dir / "user_vectors.npy"
            np.save(user_vectors_path, np.asarray(user_vectors, dtype=np.float32))

        metadata = {
            "variant": retriever.variant,
            "retriever_kind": retriever.retriever_kind,
            "history": retriever.history,
            "model": {"retriever": retriever.variant},
            "item_embeddings": _relative(item_embeddings_path, bundle_dir),
            "metrics": _relative(metrics_path, bundle_dir),
            "ann_index": _relative(ann_index_path, bundle_dir) if ann_index_path is not None else None,
            "user_vectors": _relative(user_vectors_path, bundle_dir) if user_vectors_path is not None else None,
        }
        _write_json(metadata_path, metadata)
        payload[name] = {"metadata": _relative(metadata_path, bundle_dir)}
    return payload


def _load_retriever_artifacts(
    bundle_dir: Path,
    payload: dict[str, object],
    config: core.PipelineConfig,
) -> dict[str, core.RetrieverArtifacts]:
    retrievers: dict[str, core.RetrieverArtifacts] = {}
    for name, item in payload.items():
        item_payload = dict(item)
        metadata = _read_json(bundle_dir / str(item_payload["metadata"]))
        user_vectors_path = metadata.get("user_vectors")
        metadata_values: dict[str, object] = {}
        if user_vectors_path is not None:
            metadata_values["user_vectors"] = np.load(bundle_dir / str(user_vectors_path))
        retrievers[str(name)] = core.RetrieverArtifacts(
            config=config,
            variant=str(metadata["variant"]),
            model=dict(metadata.get("model", {"retriever": name})),
            item_encoder=None,
            user_encoder=None,
            item_embeddings=np.load(bundle_dir / str(metadata["item_embeddings"])),
            ann_index_path=(bundle_dir / str(metadata["ann_index"])) if metadata.get("ann_index") is not None else None,
            ann_index=None,
            metrics=_read_frame(bundle_dir / str(metadata["metrics"])),
            history=dict(metadata.get("history", {})),
            retriever_kind=str(metadata.get("retriever_kind", "vector")),
            metadata=metadata_values,
        )
    return retrievers


def _ranker_feature_names(config: core.PipelineConfig) -> list[str]:
    return [*core._ranker_dense_feature_columns(config), "item_category_idx", "rank"]


def _write_ranker_artifacts(ranker: core.RankerArtifacts, bundle_dir: Path) -> dict[str, object]:
    models_dir = bundle_dir / "models"
    model_path = models_dir / "ranker.onnx"
    features_path = models_dir / "ranker_features.json"
    metrics_path = models_dir / "ranker_metrics.parquet"
    feature_names = _ranker_feature_names(ranker.config)
    export_xgboost_ranker_to_onnx(ranker.model, model_path, n_features=len(feature_names))
    _write_json(
        features_path,
        {
            "input_name": "input",
            "feature_names": feature_names,
            "target_opset": 15,
        },
    )
    _write_frame(metrics_path, ranker.metrics)
    return {
        "backend": ranker.backend,
        "selected_retriever_variant": ranker.selected_retriever_variant,
        "history": ranker.history,
        "model": _relative(model_path, bundle_dir),
        "features": _relative(features_path, bundle_dir),
        "metrics": _relative(metrics_path, bundle_dir),
    }


def _load_ranker_artifacts(
    bundle_dir: Path,
    payload: dict[str, object],
    config: core.PipelineConfig,
) -> core.RankerArtifacts:
    feature_payload = _read_json(bundle_dir / str(payload["features"]))
    feature_names = [str(name) for name in feature_payload["feature_names"]]
    predictor = ONNXRankerPredictor(bundle_dir / str(payload["model"]), feature_names)
    return core.RankerArtifacts(
        config=config,
        model=predictor,
        metrics=_read_frame(bundle_dir / str(payload["metrics"])),
        history=dict(payload.get("history", {})),
        selected_retriever_variant=str(payload["selected_retriever_variant"]),
        backend=str(payload.get("backend", "xgboost")),
    )


def save_runtime_bundle(bundle: RuntimeBundle) -> None:
    manifest = bundle.manifest
    if bundle.prepared is None or bundle.split_artifacts is None or bundle.ranker is None:
        raise ValueError("Cannot save an incomplete runtime bundle.")

    bundle_dir = Path(manifest.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stale_pickle = bundle_dir / "runtime_bundle.pkl"
    if stale_pickle.exists():
        stale_pickle.unlink()

    prepared_payload = _write_prepared_artifacts(bundle.prepared, bundle_dir)
    split_payload = _write_split_artifacts(bundle.split_artifacts, bundle_dir, serving_index=bundle.serving_index)
    serving_payload = _write_serving_index(bundle.serving_index, bundle_dir)
    retriever_payload = _write_retriever_artifacts(bundle.retrievers, bundle_dir)
    ranker_payload = _write_ranker_artifacts(bundle.ranker, bundle_dir)
    evaluation_path = Path(manifest.evaluation_summary_path) if manifest.evaluation_summary_path is not None else None
    if evaluation_path is not None:
        _write_json(evaluation_path, bundle.evaluation_summary.to_dict())

    payload = {
        "schema_version": ONNX_BUNDLE_SCHEMA_VERSION,
        "bundle_format": ONNX_BUNDLE_FORMAT,
        "created_at": manifest.created_at,
        "manifest": _relative(manifest.manifest_file, bundle_dir),
        "prepared": prepared_payload,
        "split_artifacts": split_payload,
        "serving_index": serving_payload,
        "retrievers": retriever_payload,
        "ranker": ranker_payload,
        "evaluation_summary": _relative(evaluation_path, bundle_dir) if evaluation_path is not None else None,
    }
    _write_json(manifest.runtime_bundle_file, payload)


def load_runtime_bundle(manifest: BundleManifest, *, serving_mode: bool = True) -> RuntimeBundle:
    bundle_dir = Path(manifest.bundle_dir)
    payload = _read_json(manifest.runtime_bundle_file)
    if payload.get("bundle_format") != ONNX_BUNDLE_FORMAT:
        raise ValueError(f"Unsupported portable bundle format: {payload.get('bundle_format')!r}")

    use_serving_mode = bool(serving_mode and payload.get("serving_index"))
    prepared = _load_prepared_artifacts(bundle_dir, dict(payload["prepared"]), serving_mode=use_serving_mode)
    split_artifacts = _load_split_artifacts(
        bundle_dir,
        dict(payload["split_artifacts"]),
        prepared.config,
        serving_mode=use_serving_mode,
    )
    serving_index_payload = payload.get("serving_index")
    serving_index = _load_serving_index(bundle_dir, dict(serving_index_payload)) if isinstance(serving_index_payload, dict) else None
    retrievers = _load_retriever_artifacts(bundle_dir, dict(payload["retrievers"]), prepared.config)
    ranker = _load_ranker_artifacts(bundle_dir, dict(payload["ranker"]), prepared.config)
    evaluation_summary_path = payload.get("evaluation_summary")
    if evaluation_summary_path is not None:
        evaluation_summary = EvaluationSummary.from_dict(_read_json(bundle_dir / str(evaluation_summary_path)))
    else:
        evaluation_summary = EvaluationSummary()
    return RuntimeBundle(
        manifest=manifest,
        prepared=prepared,
        split_artifacts=split_artifacts,
        retrievers=retrievers,
        ranker=ranker,
        serving_index=serving_index,
        evaluation_summary=evaluation_summary,
        is_mock=False,
    )
