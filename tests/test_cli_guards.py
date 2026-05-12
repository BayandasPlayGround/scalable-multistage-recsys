from __future__ import annotations

import json

import pytest

from amazon_recsys.cli.main import _assert_activation_allowed
from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest


def _manifest(workspace_dir, *, run_profile: str = "debug", version: str = "debug-bundle") -> BundleManifest:
    bundle_dir = workspace_dir / "bundle"
    data_dir = bundle_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "pipeline_config.json").write_text(json.dumps({"categories": ["All_Beauty"]}), encoding="utf-8")
    return BundleManifest(
        version=version,
        created_at="2026-05-12T00:00:00+00:00",
        manifest_path=str(bundle_dir / "manifest.json"),
        bundle_dir=str(bundle_dir),
        runtime_bundle_path=str(bundle_dir / "runtime_bundle.json"),
        evaluation_summary_path=None,
        run_name=version,
        run_profile=run_profile,
        model_backend="xgboost",
        bundle_format="onnx",
        retriever_variants=["content_based"],
    )


def test_production_activation_refuses_debug_or_single_category_bundle(workspace_dir) -> None:
    settings = AppSettings(workspace_root=workspace_dir, environment="production")

    with pytest.raises(RuntimeError, match="Production activation refused"):
        _assert_activation_allowed(settings, _manifest(workspace_dir), allow_non_prod_activation=False)


def test_production_activation_allows_explicit_override(workspace_dir) -> None:
    settings = AppSettings(workspace_root=workspace_dir, environment="production")

    _assert_activation_allowed(settings, _manifest(workspace_dir), allow_non_prod_activation=True)
