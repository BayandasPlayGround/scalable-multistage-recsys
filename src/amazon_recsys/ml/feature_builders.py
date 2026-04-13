from __future__ import annotations

from amazon_recsys.ml import core


def prepare_corpus(pipeline_config, force_rebuild: bool = False):
    return core.prepare_corpus(pipeline_config, force_rebuild=force_rebuild)


def make_splits(prepared):
    return core.make_splits(prepared)


__all__ = ["make_splits", "prepare_corpus"]
