from amazon_recsys.ml.legacy import load_legacy_pipeline


SUPPORTED_RANKERS = ("xgboost", "dlrm")


def train_ranker(prepared, split_artifacts, retrievers, backend: str = "xgboost"):
    return load_legacy_pipeline().train_ranker(prepared, split_artifacts, retrievers, backend=backend)


__all__ = ["SUPPORTED_RANKERS", "train_ranker"]
