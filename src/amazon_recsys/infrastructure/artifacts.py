from __future__ import annotations

import json
import pickle
from pathlib import Path

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import ActiveBundlePointer, BundleManifest, RuntimeBundle, utcnow_iso
from amazon_recsys.ml.io_utils import atomic_write_json
from amazon_recsys.ml.bundles import (
    ONNX_BUNDLE_FORMAT,
    build_bundle_manifest,
    build_runtime_bundle,
    generate_bundle_version,
    load_runtime_bundle,
    save_runtime_bundle,
)
from amazon_recsys.ml.legacy import load_legacy_pipeline
from amazon_recsys.ml.pipelines import TrainingSession


class LocalArtifactStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings.ensure_runtime_directories()

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        atomic_write_json(path, payload)

    def _read_json(self, path: Path) -> dict[str, object]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_manifest(self, manifest: BundleManifest) -> None:
        self._write_json(manifest.manifest_file, manifest.to_dict())

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

    def save_bundle(self, session: TrainingSession, version: str | None = None) -> BundleManifest:
        bundle_version = version or generate_bundle_version(self.settings.training.run_name)
        bundle_dir = self.settings.resolved_bundle_root / bundle_version
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_bundle_manifest(self.settings, session, bundle_version, bundle_dir)
        runtime_bundle = build_runtime_bundle(session, manifest)
        save_runtime_bundle(runtime_bundle)
        self.write_manifest(manifest)
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
            return self._legacy_active_pointer(path, payload)
        raise ValueError(f"Unsupported active bundle payload at {path}")

    def _legacy_active_pointer(self, path: Path, payload: dict[str, object]) -> ActiveBundlePointer:
        return ActiveBundlePointer(
            version=str(payload.get("version", "unknown")),
            manifest_path=str(path),
            activated_at=str(payload.get("created_at", utcnow_iso())),
        )

    def read_active_manifest(self) -> BundleManifest | None:
        pointer = self.read_active_pointer()
        if pointer is None:
            return None
        manifest_path = Path(pointer.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Active manifest file is missing: {manifest_path}")
        return BundleManifest.from_dict(self._read_json(manifest_path))

    def load_bundle(self, manifest: BundleManifest) -> RuntimeBundle:
        if manifest.bundle_format == ONNX_BUNDLE_FORMAT:
            return load_runtime_bundle(manifest)
        if manifest.bundle_format != "pickle":
            raise ValueError(f"Unsupported bundle format: {manifest.bundle_format!r}")

        # Import both module paths so legacy pickle bundles created before the migration unpickle cleanly.
        load_legacy_pipeline()
        with open(manifest.runtime_bundle_file, "rb") as handle:
            return pickle.load(handle)

    def load_active_bundle(self) -> RuntimeBundle | None:
        manifest = self.read_active_manifest()
        if manifest is None:
            return None
        return self.load_bundle(manifest)
