from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest, RuntimeBundle


def generate_bundle_version(run_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_name}-{stamp}"


def build_bundle_manifest(
    settings: AppSettings,
    session: Any,
    version: str,
    bundle_dir: Path,
) -> BundleManifest:
    manifest_path = (bundle_dir / "manifest.json").resolve()
    runtime_bundle_path = (bundle_dir / "runtime_bundle.pkl").resolve()
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
        retriever_variants=sorted(session.retrievers.keys()),
        notes={
            "workspace_root": str(settings.workspace_root),
            "legacy_workspace_root": str(settings.legacy_workspace_root),
            "legacy_artifact_root": str(settings.legacy_artifact_root),
            "mlflow_tracking_enabled": bool(getattr(session, "mlflow_run_id", None)),
            "mlflow_tracking_uri": getattr(session, "mlflow_tracking_uri", None),
            "mlflow_experiment_name": getattr(session, "mlflow_experiment_name", None),
            "mlflow_run_id": getattr(session, "mlflow_run_id", None),
        },
    )


def _sanitize_runtime_objects(session: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    prepared = session.prepared
    if hasattr(prepared, "item_text_matrix"):
        prepared.item_text_matrix = np.asarray(prepared.item_text_matrix)

    retrievers = dict(session.retrievers)
    for retriever in retrievers.values():
        retriever.ann_index = None
        if getattr(retriever, "retriever_kind", "vector") == "neural":
            raise ValueError("Bundle export currently supports classical/vector retrievers only.")

    ranker = session.ranker
    if getattr(ranker, "backend", "xgboost") != "xgboost":
        raise ValueError("Bundle export currently supports backend='xgboost' only.")

    return prepared, session.split_artifacts, retrievers, ranker


def build_runtime_bundle(session: Any, manifest: BundleManifest) -> RuntimeBundle:
    prepared, split_artifacts, retrievers, ranker = _sanitize_runtime_objects(session)
    return RuntimeBundle(
        manifest=manifest,
        prepared=prepared,
        split_artifacts=split_artifacts,
        retrievers=retrievers,
        ranker=ranker,
        evaluation_summary=session.evaluation_summary,
        is_mock=False,
    )
