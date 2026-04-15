from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amazon_recsys.config.settings import AppSettings


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_synthetic_workspace(root: Path) -> Path:
    data_dir = root / "amazon_review_data"
    metadata_dir = data_dir / "metadata"
    item_ids = ["A1", "A2", "A3", "A4", "A5", "A6"]

    reviews: list[dict] = []
    sequences = {
        "u1": ["A1", "A2", "A3", "A4", "A5"],
        "u2": ["A1", "A2", "A3", "A5", "A6"],
        "u3": ["A2", "A3", "A4", "A5", "A6"],
        "u4": ["A1", "A3", "A4", "A5", "A6"],
    }
    base_timestamp = 1_700_000_000_000
    for user_index, (user_id, item_sequence) in enumerate(sequences.items(), start=1):
        for step, item_id in enumerate(item_sequence):
            reviews.append(
                {
                    "user_id": user_id,
                    "asin": item_id,
                    "parent_asin": item_id,
                    "rating": 5,
                    "timestamp": base_timestamp + (user_index * 10_000) + (step * 1_000),
                    "verified_purchase": True,
                    "helpful_vote": step,
                    "title": f"Review {user_id}-{item_id}",
                    "text": f"{user_id} enjoyed {item_id}",
                }
            )
        reviews.append(
            {
                "user_id": user_id,
                "asin": item_ids[(user_index + 1) % len(item_ids)],
                "parent_asin": item_ids[(user_index + 1) % len(item_ids)],
                "rating": 1,
                "timestamp": base_timestamp + (user_index * 10_000) + 9_999,
                "verified_purchase": False,
                "helpful_vote": 0,
                "title": f"Negative {user_id}",
                "text": f"{user_id} disliked this item",
            }
        )

    metadata_rows = [
        {
            "parent_asin": item_id,
            "title": f"Product {item_id}",
            "store": "Synthetic Store",
            "categories": ["Beauty", "Synthetic"],
            "description": [f"Description for {item_id}"],
            "features": [f"Feature {item_id}"],
            "bought_together": [],
            "price": f"{10 + index}.99",
            "average_rating": 4.0 + (index * 0.1),
            "rating_number": 50 + index,
        }
        for index, item_id in enumerate(item_ids, start=1)
    ]

    _write_jsonl(data_dir / "All_Beauty.jsonl", reviews)
    _write_jsonl_gz(metadata_dir / "meta_All_Beauty.jsonl.gz", metadata_rows)
    return root


@pytest.fixture
def workspace_dir(request) -> Path:
    root = Path(__file__).resolve().parent / ".tmp" / f"{request.node.name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def synthetic_workspace(workspace_dir: Path) -> Path:
    return build_synthetic_workspace(workspace_dir)


@pytest.fixture
def test_settings(synthetic_workspace: Path) -> AppSettings:
    settings = AppSettings(
        workspace_root=synthetic_workspace,
        data_dir=Path("amazon_review_data"),
        categories=("All_Beauty",),
        metadata_download_if_missing=False,
        use_mock_bundle_if_missing=False,
        mlflow_enabled=False,
        monitoring_enabled=False,
        run_name="pytest",
        run_profile="debug",
        show_progress=False,
        dev_mode=False,
        k_core=2,
        eval_user_cap=10,
        retrieval_top_k=10,
        candidate_union_top_k=12,
        candidate_union_batch_size=20,
        cooccurrence_candidate_k=10,
        latent_cf_candidate_k=10,
        content_candidate_k=10,
        neural_candidate_k=10,
        ranker_candidate_top_k=10,
        ranker_train_example_cap=50,
        ranker_val_example_cap=10,
        ranker_negatives_per_positive=3,
        xgb_n_estimators=10,
    )
    settings.ensure_runtime_directories()
    return settings


@pytest.fixture
def mock_settings(workspace_dir: Path) -> AppSettings:
    settings = AppSettings(
        workspace_root=workspace_dir,
        environment="local",
        use_mock_bundle_if_missing=True,
        mlflow_enabled=False,
        monitoring_enabled=False,
        show_progress=False,
    )
    settings.ensure_runtime_directories()
    return settings


@pytest.fixture
def production_settings(synthetic_workspace: Path) -> AppSettings:
    settings = AppSettings(
        workspace_root=synthetic_workspace,
        environment="production",
        debug=False,
        reload=False,
        data_dir=Path("amazon_review_data"),
        categories=("All_Beauty",),
        metadata_download_if_missing=False,
        use_mock_bundle_if_missing=False,
        mlflow_enabled=False,
        monitoring_enabled=False,
        run_name="prod-pytest",
        run_profile="debug",
        show_progress=False,
        dev_mode=False,
        k_core=2,
        eval_user_cap=10,
        retrieval_top_k=10,
        candidate_union_top_k=12,
        candidate_union_batch_size=20,
        cooccurrence_candidate_k=10,
        latent_cf_candidate_k=10,
        content_candidate_k=10,
        neural_candidate_k=10,
        ranker_candidate_top_k=10,
        ranker_train_example_cap=50,
        ranker_val_example_cap=10,
        ranker_negatives_per_positive=3,
        xgb_n_estimators=10,
    )
    settings.ensure_runtime_directories()
    return settings


@pytest.fixture
def test_container(test_settings: AppSettings):
    from amazon_recsys.config.container import build_container

    return build_container(test_settings)
