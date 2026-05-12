"""Tests for Phase D: source-biased ranker negative sampling.

Three scopes:
- ``_parse_hardneg_mix``: input validation + normalisation.
- ``_sample_negatives_by_source_mix``: pool selection respects the mix weights.
- ``_rebalance_ranker_candidates``: end-to-end behaviour, with the default mix preserving
  exact legacy uniform-random behaviour and the recommended ``0.6,0.3,0.1`` mix shifting
  the per-group source composition as expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amazon_recsys.ml import core


# --- parser ---------------------------------------------------------------------------------


def test_parse_hardneg_mix_normalises_to_unit_sum() -> None:
    weights = core._parse_hardneg_mix("0.6,0.3,0.1")
    assert weights == pytest.approx((0.6, 0.3, 0.1))
    weights = core._parse_hardneg_mix("3,1,1")
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.6)


def test_parse_hardneg_mix_default_is_pure_random() -> None:
    weights = core._parse_hardneg_mix("0,0,1.0")
    assert weights == pytest.approx((0.0, 0.0, 1.0))


def test_parse_hardneg_mix_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="three"):
        core._parse_hardneg_mix("0.6,0.4")
    with pytest.raises(ValueError, match="three"):
        core._parse_hardneg_mix("0.5,0.3,0.1,0.1")


def test_parse_hardneg_mix_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="numeric"):
        core._parse_hardneg_mix("a,b,c")


def test_parse_hardneg_mix_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        core._parse_hardneg_mix("0.6,0.3,-0.1")


def test_parse_hardneg_mix_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="positive value"):
        core._parse_hardneg_mix("0,0,0")


def test_pipeline_config_post_init_rejects_bad_mix() -> None:
    with pytest.raises(ValueError):
        core.PipelineConfig(ranker_hardneg_mix="abc,def,ghi")


# --- sampler --------------------------------------------------------------------------------


def _build_negatives(per_source: dict[str, int]) -> pd.DataFrame:
    rows = []
    item_idx = 1
    for source, count in per_source.items():
        for _ in range(count):
            rows.append({"item_idx": item_idx, "source": source, "label": 0, "rank": item_idx})
            item_idx += 1
    return pd.DataFrame(rows)


def test_sample_with_zero_pop_zero_cooc_falls_back_to_uniform() -> None:
    negatives = _build_negatives({"popularity": 5, "cooccurrence": 5, "latent_cf": 10})
    rng = np.random.default_rng(0)
    drawn = core._sample_negatives_by_source_mix(negatives, target_count=8, mix_weights=(0.0, 0.0, 1.0), rng=rng)
    assert len(drawn) == 8


def test_sample_respects_mix_weights_when_pools_are_ample() -> None:
    negatives = _build_negatives({"popularity": 50, "cooccurrence": 50, "latent_cf": 50})
    rng = np.random.default_rng(42)
    drawn = core._sample_negatives_by_source_mix(negatives, target_count=10, mix_weights=(0.6, 0.3, 0.1), rng=rng)
    counts = drawn["source"].value_counts().to_dict()
    assert counts.get("popularity") == 6  # 10 * 0.6 = 6
    assert counts.get("cooccurrence") == 3  # 10 * 0.3 = 3
    # remaining 1 from "latent_cf" — labeled as random pool because not in {pop, cooc}
    assert counts.get("latent_cf") == 1


def test_sample_tops_up_from_remaining_when_pool_short() -> None:
    negatives = _build_negatives({"popularity": 2, "cooccurrence": 1, "latent_cf": 20})
    rng = np.random.default_rng(7)
    drawn = core._sample_negatives_by_source_mix(negatives, target_count=10, mix_weights=(0.6, 0.3, 0.1), rng=rng)
    assert len(drawn) == 10
    # popularity pool only had 2 → fully consumed; deficit comes from random pool.
    counts = drawn["source"].value_counts().to_dict()
    assert counts.get("popularity") == 2
    assert counts.get("cooccurrence") == 1
    # random pool fills the rest (10 - 2 - 1 = 7).
    assert counts.get("latent_cf") == 7


def test_sample_returns_input_when_smaller_than_target() -> None:
    negatives = _build_negatives({"popularity": 2, "latent_cf": 3})
    rng = np.random.default_rng(0)
    drawn = core._sample_negatives_by_source_mix(negatives, target_count=10, mix_weights=(0.6, 0.3, 0.1), rng=rng)
    assert len(drawn) == 5  # everything, untouched
    assert drawn.equals(negatives)


# --- end-to-end rebalancer ------------------------------------------------------------------


def _build_candidates_one_group(per_source: dict[str, int], positives: int = 1) -> pd.DataFrame:
    rows = []
    item_idx = 1
    for _ in range(positives):
        rows.append({"example_id": 0, "item_idx": item_idx, "source": "cooccurrence", "label": 1, "rank": item_idx})
        item_idx += 1
    for source, count in per_source.items():
        for _ in range(count):
            rows.append({"example_id": 0, "item_idx": item_idx, "source": source, "label": 0, "rank": item_idx})
            item_idx += 1
    return pd.DataFrame(rows)


def test_default_mix_matches_legacy_pure_random_behaviour() -> None:
    """``0,0,1.0`` must be bit-for-bit identical to the pre-Phase-D uniform sampler.

    We can't easily compare against the literal old function (it's gone) but we can verify
    that the sampler reduces to ``pool.sample(...)`` when pop+cooc weight is zero — that's
    the same call the legacy code made.
    """
    negatives = _build_negatives({"popularity": 10, "cooccurrence": 10, "latent_cf": 30})

    def _legacy_sample(rng_seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(rng_seed)
        return negatives.sample(n=15, random_state=int(rng.integers(0, 1_000_000)))

    def _new_sample(rng_seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(rng_seed)
        return core._sample_negatives_by_source_mix(negatives, 15, (0.0, 0.0, 1.0), rng)

    pd.testing.assert_frame_equal(_legacy_sample(123).sort_index(), _new_sample(123).sort_index())


def test_rebalance_with_hardneg_mix_changes_source_distribution() -> None:
    candidates = _build_candidates_one_group({"popularity": 30, "cooccurrence": 30, "latent_cf": 30})

    pure_random = core._rebalance_ranker_candidates(candidates.copy(), negatives_per_positive=10, seed=0, hardneg_mix="0,0,1.0")
    biased = core._rebalance_ranker_candidates(candidates.copy(), negatives_per_positive=10, seed=0, hardneg_mix="0.6,0.3,0.1")

    pure_neg_sources = pure_random[pure_random["label"] == 0]["source"].value_counts().to_dict()
    biased_neg_sources = biased[biased["label"] == 0]["source"].value_counts().to_dict()

    # Pure random should be roughly uniform; biased should be heavily skewed to popularity.
    assert biased_neg_sources.get("popularity", 0) >= 5, biased_neg_sources
    assert biased_neg_sources.get("popularity", 0) > pure_neg_sources.get("popularity", 0)


def test_rebalance_preserves_positives() -> None:
    """No matter what mix, every positive must survive the rebalancer."""
    candidates = _build_candidates_one_group({"popularity": 10, "latent_cf": 20}, positives=2)
    out = core._rebalance_ranker_candidates(candidates, negatives_per_positive=3, seed=0, hardneg_mix="0.6,0.3,0.1")
    assert int((out["label"] == 1).sum()) == 2


def test_rebalance_handles_group_with_no_positives() -> None:
    """Groups with zero positives should still produce some rows (legacy behaviour: top-K by rank)."""
    candidates = pd.DataFrame(
        [
            {"example_id": 0, "item_idx": 1, "source": "popularity", "label": 0, "rank": 1},
            {"example_id": 0, "item_idx": 2, "source": "latent_cf", "label": 0, "rank": 2},
            {"example_id": 0, "item_idx": 3, "source": "cooccurrence", "label": 0, "rank": 3},
        ]
    )
    out = core._rebalance_ranker_candidates(candidates, negatives_per_positive=2, seed=0, hardneg_mix="0.6,0.3,0.1")
    assert len(out) == 2  # max(1, negatives_per_positive) = 2 head-by-rank rows
