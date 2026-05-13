from __future__ import annotations

import json

import pytest

from amazon_recsys.cli.main import _assert_acceptance_gates_allowed, _assert_activation_allowed, _manifest_eval_dir
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


def test_acceptance_gate_refuses_activation_for_below_floor_metrics(workspace_dir) -> None:
    eval_dir = workspace_dir / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "hybrid_union_xgboost_ranker_metrics.csv").write_text(
        "split,K,recall\n"
        "test,100,0.021\n",
        encoding="utf-8",
    )
    settings = AppSettings(workspace_root=workspace_dir, gate_profile="recovery-v1")

    with pytest.raises(RuntimeError) as info:
        _assert_acceptance_gates_allowed(settings, eval_dir, bundle_version="weak-bundle")

    message = str(info.value)
    assert "Activation refused" in message
    assert "weak-bundle" in message
    assert "observed 0.0210 < floor 0.0520" in message


def test_acceptance_gate_off_does_not_require_eval_dir(workspace_dir) -> None:
    settings = AppSettings(workspace_root=workspace_dir, gate_profile="off")

    _assert_acceptance_gates_allowed(settings, None, bundle_version="manual-test")


def test_manifest_eval_dir_reads_bundle_evaluation_summary(workspace_dir) -> None:
    eval_dir = workspace_dir / "evaluation"
    eval_dir.mkdir()
    manifest = _manifest(workspace_dir, run_profile="quality", version="candidate")
    summary_path = workspace_dir / "bundle" / "evaluation_summary.json"
    summary_path.write_text(json.dumps({"eval_dir": str(eval_dir), "metric_files": []}), encoding="utf-8")
    manifest.evaluation_summary_path = str(summary_path)

    assert _manifest_eval_dir(manifest) == eval_dir
