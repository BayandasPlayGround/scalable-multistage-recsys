from amazon_recsys.ml.legacy import load_legacy_pipeline


SUPPORTED_RETRIEVERS = ("cooccurrence", "popularity", "content_based", "latent_cf", "two_tower")


def train_retrievers(prepared, split_artifacts):
    return load_legacy_pipeline().train_retrievers(prepared, split_artifacts)


__all__ = ["SUPPORTED_RETRIEVERS", "train_retrievers"]
