from amazon_recsys.ml import core


SUPPORTED_RETRIEVERS = ("cooccurrence", "popularity", "content_based", "latent_cf", "two_tower")


def train_retrievers(prepared, split_artifacts):
    return core.train_retrievers(prepared, split_artifacts)


__all__ = ["SUPPORTED_RETRIEVERS", "train_retrievers"]
