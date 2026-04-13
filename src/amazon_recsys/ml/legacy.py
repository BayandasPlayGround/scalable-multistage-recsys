from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


LEGACY_MODULE_NAME = "amazon_recsys_legacy_pipeline"


def legacy_pipeline_path() -> Path:
    return Path(__file__).resolve().parents[3] / "notebooks" / "amazon_recsys_pipeline.py"


@lru_cache
def load_legacy_pipeline():
    module_path = legacy_pipeline_path()
    if not module_path.exists():
        raise FileNotFoundError(f"Legacy pipeline file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(LEGACY_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
