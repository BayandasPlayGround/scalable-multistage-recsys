from amazon_recsys.ml import core


SUPPORTED_RANKERS = ("xgboost", "dlrm")


def train_ranker(prepared, split_artifacts, retrievers, backend: str = "xgboost"):
    return core.train_ranker(prepared, split_artifacts, retrievers, backend=backend)


__all__ = ["SUPPORTED_RANKERS", "train_ranker"]
