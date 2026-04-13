from __future__ import annotations

from amazon_recsys.ml.legacy import load_legacy_pipeline


def prepare_corpus(legacy_config, force_rebuild: bool = False):
    return load_legacy_pipeline().prepare_corpus(legacy_config, force_rebuild=force_rebuild)


def make_splits(prepared):
    return load_legacy_pipeline().make_splits(prepared)


__all__ = ["make_splits", "prepare_corpus"]
