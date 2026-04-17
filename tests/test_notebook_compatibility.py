from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
import sys

import pytest

from amazon_recsys.ml import core
from amazon_recsys.ml.legacy import load_legacy_pipeline


def _load_notebook_pipeline(spec_name: str) -> object:
    module_path = Path(__file__).resolve().parents[1] / "notebooks" / "amazon_recsys_pipeline.py"
    spec = importlib.util.spec_from_file_location(spec_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.foundation
@pytest.mark.data
def test_notebook_pipeline_reexports_package_core() -> None:
    module = _load_notebook_pipeline("notebook_pipeline_compat")

    assert module.PipelineConfig is core.PipelineConfig
    assert module.prepare_corpus is core.prepare_corpus
    assert module.make_splits is core.make_splits
    assert module.train_retrievers is core.train_retrievers
    assert module.train_ranker is core.train_ranker
    assert module.recommend is core.recommend
    assert module.xgb is core.xgb
    assert module.tfrs is core.tfrs
    assert module._normalize_metrics_frame is core._normalize_metrics_frame
    assert "PipelineConfig" in module.__all__


@pytest.mark.foundation
@pytest.mark.data
def test_legacy_loader_aliases_notebook_module_for_pickle_compatibility() -> None:
    aliases = ("amazon_recsys_pipeline", "amazon_recsys_legacy_pipeline")
    saved_modules = {
        alias: sys.modules[alias]
        for alias in aliases
        if alias in sys.modules
    }

    for alias in aliases:
        sys.modules.pop(alias, None)
    load_legacy_pipeline.cache_clear()

    try:
        module = load_legacy_pipeline()

        assert sys.modules["amazon_recsys_pipeline"] is module
        assert sys.modules["amazon_recsys_legacy_pipeline"] is module
        assert module.PipelineConfig is core.PipelineConfig

        legacy_reference = pickle.dumps(core.PipelineConfig, protocol=0).replace(
            b"amazon_recsys.ml.core\nPipelineConfig\n",
            b"amazon_recsys_pipeline\nPipelineConfig\n",
        )

        assert pickle.loads(legacy_reference) is core.PipelineConfig
    finally:
        load_legacy_pipeline.cache_clear()
        for alias in aliases:
            sys.modules.pop(alias, None)
        sys.modules.update(saved_modules)
