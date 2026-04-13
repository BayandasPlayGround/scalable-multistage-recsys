from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from amazon_recsys.ml import core


@pytest.mark.foundation
@pytest.mark.data
def test_notebook_pipeline_reexports_package_core() -> None:
    module_path = Path(__file__).resolve().parents[1] / "notebooks" / "amazon_recsys_pipeline.py"
    spec = importlib.util.spec_from_file_location("notebook_pipeline_compat", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.PipelineConfig is core.PipelineConfig
    assert module.prepare_corpus is core.prepare_corpus
    assert module.make_splits is core.make_splits
    assert module.train_retrievers is core.train_retrievers
    assert module.train_ranker is core.train_ranker
    assert module.recommend is core.recommend
