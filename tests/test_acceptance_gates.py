"""Tests for ``validate_acceptance_gates``.

Drives the validator with synthesised eval CSVs so we don't need a full training run. Two
scenarios are anchored in real history:

* ``_populate_dat_v2_eval_dir`` writes the exact test-split numbers from the failed
  ``prod-2026-05-10-dat-v2`` bundle (the run that triggered this whole work). We assert that
  ``blair-v1`` would have caught that regression, while ``recovery-v1`` (the wider, baseline
  profile) would have passed it.

* ``_populate_passing_eval_dir`` writes synthesised numbers that clear every ``blair-v1``
  floor — confirming the validator approves a good run, not just blocks bad ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amazon_recsys.ml.bundles import GATE_PROFILES, GateValidationError, validate_acceptance_gates


def _write_csv(directory: Path, name: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / name, index=False)


def _populate_dat_v2_eval_dir(eval_dir: Path) -> None:
    """Real test-split numbers from the ``prod-2026-05-10-dat-v2`` bundle (the failed run)."""
    _write_csv(eval_dir, "blair_text_retriever_metrics.csv", [{"split": "test", "K": 100, "recall": 0.0}])
    _write_csv(
        eval_dir,
        "candidate_recall_diagnostics.csv",
        [
            {"split": "test", "stage": "candidate_union", "scope": "overall", "name": "all", "hit_rate": 0.108},
            {"split": "test", "stage": "ranker_candidates", "scope": "overall", "name": "all", "hit_rate": 0.064},
        ],
    )
    _write_csv(
        eval_dir,
        "candidate_recall_by_cold_start_type.csv",
        [{"split": "test", "stage": "candidate_union", "name": "anonymous_no_history", "hit_rate": 0.052}],
    )
    _write_csv(
        eval_dir,
        "candidate_recall_by_category.csv",
        [{"split": "test", "stage": "candidate_union", "name": "Industrial_and_Scientific", "hit_rate": 0.058}],
    )
    _write_csv(
        eval_dir,
        "hybrid_union_xgboost_ranker_metrics.csv",
        [
            {"split": "test", "K": 10, "recall": 0.028},
            {"split": "test", "K": 100, "recall": 0.052},
        ],
    )


def _populate_passing_eval_dir(eval_dir: Path) -> None:
    """Synthesised numbers that clear every ``blair-v1`` floor."""
    _write_csv(eval_dir, "blair_text_retriever_metrics.csv", [{"split": "test", "K": 100, "recall": 0.08}])
    _write_csv(
        eval_dir,
        "candidate_recall_diagnostics.csv",
        [
            {"split": "test", "stage": "candidate_union", "scope": "overall", "name": "all", "hit_rate": 0.20},
            {"split": "test", "stage": "ranker_candidates", "scope": "overall", "name": "all", "hit_rate": 0.15},
        ],
    )
    _write_csv(
        eval_dir,
        "candidate_recall_by_cold_start_type.csv",
        [{"split": "test", "stage": "candidate_union", "name": "anonymous_no_history", "hit_rate": 0.07}],
    )
    _write_csv(
        eval_dir,
        "candidate_recall_by_category.csv",
        [{"split": "test", "stage": "candidate_union", "name": "Industrial_and_Scientific", "hit_rate": 0.09}],
    )
    _write_csv(
        eval_dir,
        "hybrid_union_xgboost_ranker_metrics.csv",
        [
            {"split": "test", "K": 10, "recall": 0.030},
            {"split": "test", "K": 100, "recall": 0.075},
        ],
    )


def test_off_profile_is_a_noop(workspace_dir: Path) -> None:
    assert validate_acceptance_gates(workspace_dir, "off") == []


def test_recovery_v1_passes_dat_v2_numbers(workspace_dir: Path) -> None:
    _populate_dat_v2_eval_dir(workspace_dir)
    passes = validate_acceptance_gates(workspace_dir, "recovery-v1")
    assert any("recall@100" in line for line in passes), passes


def test_blair_v1_catches_dat_v2_regression(workspace_dir: Path) -> None:
    """The failed dat-v2 bundle should trip the ``blair-v1`` profile so it never ships again."""
    _populate_dat_v2_eval_dir(workspace_dir)
    with pytest.raises(GateValidationError) as info:
        validate_acceptance_gates(workspace_dir, "blair-v1")
    message = str(info.value)
    assert "blair_text retriever recall@100" in message
    assert "0.052" in message  # final recall@100 baseline shows in the failure trail


def test_blair_v1_passes_when_metrics_clear_every_floor(workspace_dir: Path) -> None:
    _populate_passing_eval_dir(workspace_dir)
    passes = validate_acceptance_gates(workspace_dir, "blair-v1")
    assert len(passes) == len(GATE_PROFILES["blair-v1"])


def test_unknown_profile_raises_value_error(workspace_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unknown gate profile"):
        validate_acceptance_gates(workspace_dir, "not-a-real-profile")


def test_missing_csv_surfaces_filename_clearly(workspace_dir: Path) -> None:
    """``recovery-v1`` reads only the ranker metrics CSV; if it's missing, error must name it."""
    with pytest.raises(GateValidationError) as info:
        validate_acceptance_gates(workspace_dir, "recovery-v1")
    message = str(info.value)
    assert "missing CSV" in message
    assert "hybrid_union_xgboost_ranker_metrics.csv" in message


def test_settings_gate_profile_validator_rejects_garbage() -> None:
    """The pydantic-side validator should catch typos before the pipeline even runs."""
    from amazon_recsys.config.settings import AppSettings

    with pytest.raises(ValueError, match="gate_profile"):
        AppSettings(gate_profile="blair-v999")  # type: ignore[call-arg]
