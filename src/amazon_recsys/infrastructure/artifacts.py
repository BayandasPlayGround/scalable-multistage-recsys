from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import ActiveBundlePointer, BundleManifest, RuntimeBundle, utcnow_iso
from amazon_recsys.ml.bundles import build_bundle_manifest, build_runtime_bundle, generate_bundle_version
from amazon_recsys.ml import core  # noqa: F401
from amazon_recsys.ml.legacy import load_legacy_pipeline
from amazon_recsys.monitoring.reference import build_reference_profile
from amazon_recsys.monitoring.store import LocalMonitoringStore
from amazon_recsys.observability.mlflow import MLflowTracker


class LocalArtifactStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings.ensure_runtime_directories()
        self.mlflow_tracker = MLflowTracker(settings)
        self.monitoring_store = LocalMonitoringStore(settings)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _read_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_bundles(self) -> list[BundleManifest]:
        manifests: list[BundleManifest] = []
        for manifest_path in sorted(self.settings.resolved_bundle_root.glob("*/manifest.json")):
            manifests.append(BundleManifest.from_dict(self._read_json(manifest_path)))
        return manifests

    def load_manifest(self, version: str) -> BundleManifest:
        manifest_path = self.settings.resolved_bundle_root / version / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found for version {version!r}.")
        return BundleManifest.from_dict(self._read_json(manifest_path))

    def save_bundle(self, session: Any, version: str | None = None) -> BundleManifest:
        bundle_version = version or generate_bundle_version(self.settings.training.run_name)
        bundle_dir = self.settings.resolved_bundle_root / bundle_version
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_bundle_manifest(self.settings, session, bundle_version, bundle_dir)
        runtime_bundle = build_runtime_bundle(session, manifest)
        with open(manifest.runtime_bundle_file, "wb") as handle:
            pickle.dump(runtime_bundle, handle)
        self._write_json(Path(manifest.manifest_path), manifest.to_dict())
        if manifest.evaluation_summary_path is not None:
            self._write_json(Path(manifest.evaluation_summary_path), runtime_bundle.evaluation_summary)
        reference_profile = build_reference_profile(self.settings, session, bundle_version=manifest.version)
        reference_profile_path = self.monitoring_store.save_reference_profile(reference_profile)
        manifest.notes["reference_profile_path"] = str(reference_profile_path)
        manifest.notes["reference_bundle_version"] = manifest.version
        self._write_json(Path(manifest.manifest_path), manifest.to_dict())
        self.mlflow_tracker.log_bundle_export(session, manifest, extra_artifacts=[reference_profile_path])
        return manifest

    def activate_bundle(self, version: str) -> ActiveBundlePointer:
        manifest = self.load_manifest(version)
        pointer = ActiveBundlePointer(
            version=manifest.version,
            manifest_path=manifest.manifest_path,
            activated_at=utcnow_iso(),
        )
        self._write_json(self.settings.resolved_active_bundle_path, pointer.to_dict())
        return pointer

    def read_active_pointer(self) -> ActiveBundlePointer | None:
        path = self.settings.resolved_active_bundle_path
        if not path.exists():
            return None
        payload = self._read_json(path)
        if "manifest_path" in payload and "activated_at" in payload:
            return ActiveBundlePointer.from_dict(payload)
        if "runtime_bundle_path" in payload:
            return ActiveBundlePointer(
                version=str(payload.get("version", "unknown")),
                manifest_path=str(path),
                activated_at=str(payload.get("created_at", utcnow_iso())),
            )
        raise ValueError(f"Unsupported active bundle payload at {path}")

    def read_active_manifest(self) -> BundleManifest | None:
        pointer = self.read_active_pointer()
        if pointer is None:
            return None
        manifest_path = Path(pointer.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Active manifest file is missing: {manifest_path}")
        return BundleManifest.from_dict(self._read_json(manifest_path))

    def load_bundle(self, manifest: BundleManifest) -> RuntimeBundle:
        # Import both module paths so bundles created before and after the migration unpickle cleanly.
        load_legacy_pipeline()
        with open(manifest.runtime_bundle_file, "rb") as handle:
            return pickle.load(handle)

    def load_active_bundle(self) -> RuntimeBundle | None:
        manifest = self.read_active_manifest()
        if manifest is None:
            return None
        return self.load_bundle(manifest)
