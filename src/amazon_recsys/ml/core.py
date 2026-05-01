from __future__ import annotations

import json
import logging
import math
import pickle
import random
import shutil
import urllib.request
import hashlib
import gc
from collections import Counter, defaultdict
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, TypeAlias

import numpy as np
import pandas as pd
import tensorflow as tf
from annoy import AnnoyIndex
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import xgboost as xgb
except ModuleNotFoundError:
    xgb = None

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:
    _tqdm = None

# Legacy notebook compatibility alias. The package no longer imports TFRS directly.
tfrs = None


LOGGER = logging.getLogger(__name__)


Record: TypeAlias = dict[str, object]
RetrieverState: TypeAlias = dict[str, object] | tf.keras.Model
MetricsHistory: TypeAlias = dict[str, object]


class RankerPredictor(Protocol):
    def predict(self, *args: object, **kwargs: object) -> object: ...


REVIEW_FILE_NAMES = {
    "All_Beauty": "All_Beauty.jsonl",
    "Automotive": "Automotive.jsonl",
    "Industrial_and_Scientific": "Industrial_and_Scientific.jsonl",
}

METADATA_URLS = {
    "All_Beauty": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz",
    "Automotive": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Automotive.jsonl.gz",
    "Industrial_and_Scientific": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz",
}


@dataclass
class PipelineConfig:
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    categories: tuple[str, ...] = ("All_Beauty", "Automotive", "Industrial_and_Scientific")
    run_name: str = "default"
    cache_version: int = 5
    run_profile: str = "quality"
    seed: int = 42
    k_core: int = 5
    dev_mode: bool = False
    dev_fraction: float = 0.05
    dev_sampling_strategy: str = "stratified_user"
    dev_hard_negative_multiplier: float = 2.5
    dev_neutral_multiplier: float = 1.5
    show_progress: bool = True
    history_len: int = 10
    max_rows_per_category: int | None = None
    train_positive_cap: int = 2_000_000
    split_eval_example_cap: int | None = None
    review_chunk_rows: int = 250_000
    text_max_features: int = 12_000
    text_svd_dim: int = 64
    retriever_embedding_dim: int = 64
    retriever_hidden_dims: tuple[int, ...] = (256, 128, 64)
    retriever_batch_size: int = 256
    retriever_epochs: int = 3
    retriever_train_example_cap: int | None = 50_000
    retriever_shuffle_buffer: int = 20_000
    negatives_per_positive: int = 3
    retriever_validation_negatives_per_positive: int = 10
    retriever_quality_min_history: int = 2
    retriever_logit_scale: float = 8.0
    persist_encoder_models: bool = False
    enable_neural_retriever: bool = False
    in_batch_weight: float = 0.15
    dat_mimic_weight: float = 0.10
    dat_category_alignment_weight: float = 0.05
    ann_trees: int = 50
    retrieval_top_k: int = 100
    eval_user_cap: int | None = 1_000
    cooccurrence_candidate_k: int = 100
    latent_cf_candidate_k: int = 150
    content_candidate_k: int = 100
    neural_candidate_k: int = 150
    popularity_backfill_k: int = 50
    category_backfill_enabled: bool = True
    recency_cooccurrence_enabled: bool = True
    candidate_union_top_k: int = 300
    ranker_candidate_top_k: int = 200
    ranker_train_example_cap: int = 2_000
    ranker_val_example_cap: int | None = 1_000
    ranker_negatives_per_positive: int = 10
    ranker_batch_size: int = 512
    ranker_epochs: int = 3
    candidate_union_batch_size: int = 500
    ranker_embedding_dim: int = 16
    ranker_hidden_dims: tuple[int, ...] = (128, 64, 32)
    ranker_backend: str = "xgboost"
    latent_cf_components: int = 64
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    metadata_download_if_missing: bool = True
    training_verbose: int = 2
    tf_prefetch_batches: int = 1
    memory_map_item_text: bool = True

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        if not 0 < float(self.dev_fraction) <= 1:
            raise ValueError("dev_fraction must be in the interval (0, 1].")
        valid_run_profiles = {"debug", "quality", "quality-neural", "full"}
        if self.run_profile not in valid_run_profiles:
            raise ValueError(f"run_profile must be one of {sorted(valid_run_profiles)}.")
        valid_sampling_strategies = {"user", "stratified_user", "category_balanced_user"}
        if self.dev_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"dev_sampling_strategy must be one of {sorted(valid_sampling_strategies)}; "
                f"received {self.dev_sampling_strategy!r}."
            )
        valid_ranker_backends = {"xgboost", "dlrm"}
        if self.ranker_backend not in valid_ranker_backends:
            raise ValueError(f"ranker_backend must be one of {sorted(valid_ranker_backends)}.")
        if float(self.dev_hard_negative_multiplier) <= 0 or float(self.dev_neutral_multiplier) <= 0:
            raise ValueError("Dev sampling multipliers must be positive.")
        if int(self.retriever_validation_negatives_per_positive) <= 0:
            raise ValueError("retriever_validation_negatives_per_positive must be positive.")
        if int(self.retriever_quality_min_history) < 1:
            raise ValueError("retriever_quality_min_history must be at least 1.")
        if float(self.retriever_logit_scale) <= 0:
            raise ValueError("retriever_logit_scale must be positive.")

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "amazon_review_data"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def artifact_root(self) -> Path:
        return self.base_dir / "artifacts" / "amazon_recsys" / self.run_name

    @property
    def cache_dir(self) -> Path:
        return self.artifact_root / "cache"

    @property
    def model_dir(self) -> Path:
        return self.artifact_root / "models"

    @property
    def eval_dir(self) -> Path:
        return self.artifact_root / "evaluation"


@dataclass
class PreparedArtifacts:
    config: PipelineConfig
    raw_review_stats: pd.DataFrame
    kcore_stats: pd.DataFrame
    interactions: pd.DataFrame
    hard_negatives: pd.DataFrame
    item_features: pd.DataFrame
    item_text_matrix: np.ndarray
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    item_id_to_idx: dict[str, int]
    item_idx_to_id: dict[int, str]
    category_to_idx: dict[str, int]


@dataclass
class SplitArtifacts:
    config: PipelineConfig
    train_examples: pd.DataFrame
    val_examples: pd.DataFrame
    test_examples: pd.DataFrame
    user_id_to_idx: dict[str, int]
    user_idx_to_id: dict[int, str]
    train_seen_map: dict[str, set[int]]
    val_seen_map: dict[str, set[int]]
    test_seen_map: dict[str, set[int]]
    train_item_popularity: Counter
    category_item_popularity: dict[str, Counter]
    cooccurrence: dict[int, Counter]
    hard_negative_history: dict[str, list[tuple[int, int]]]


@dataclass
class ServingIndex:
    user_summary: pd.DataFrame
    user_history: pd.DataFrame


@dataclass
class RetrieverArtifacts:
    config: PipelineConfig
    variant: str
    model: RetrieverState
    item_encoder: tf.keras.Model | None
    user_encoder: tf.keras.Model | None
    item_embeddings: np.ndarray
    ann_index_path: Path | None
    ann_index: AnnoyIndex | None
    metrics: pd.DataFrame
    history: dict[str, list[float]]
    retriever_kind: str = "neural"
    metadata: MetricsHistory = field(default_factory=dict)


@dataclass
class RankerArtifacts:
    config: PipelineConfig
    model: RankerPredictor
    metrics: pd.DataFrame
    history: MetricsHistory
    selected_retriever_variant: str
    backend: str = "xgboost"


@dataclass(slots=True)
class InferenceRequestContext:
    user_identifier: str
    history_item_idxs: list[int]
    history_item_set: set[int]
    inference_example: pd.DataFrame


class _NullProgress:
    def __init__(self, iterable: Iterable[object] | None = None):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable) if self._iterable is not None else iter(())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def _progress(
    iterable: Iterable[object] | None = None,
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    disable: bool = False,
    leave: bool = False,
):
    if disable or _tqdm is None:
        return _NullProgress(iterable)
    return _tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories(config: PipelineConfig) -> None:
    for path in [
        config.data_dir,
        config.metadata_dir,
        config.artifact_root,
        config.cache_dir,
        config.model_dir,
        config.eval_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def apply_run_profile(config: PipelineConfig) -> PipelineConfig:
    defaults = {
        "debug": {
            "max_rows_per_category": 100_000,
            "retriever_train_example_cap": 40_000,
            "retriever_quality_min_history": 2,
            "enable_neural_retriever": False,
            "category_backfill_enabled": True,
            "recency_cooccurrence_enabled": True,
            "eval_user_cap": 1_000,
            "candidate_union_top_k": 200,
            "candidate_union_batch_size": 300,
            "ranker_candidate_top_k": 75,
            "ranker_train_example_cap": 2_000,
            "ranker_val_example_cap": 500,
            "split_eval_example_cap": 1_000,
        },
        "quality": {
            "max_rows_per_category": None,
            "retriever_train_example_cap": 100_000,
            "retriever_quality_min_history": 3,
            "enable_neural_retriever": False,
            "category_backfill_enabled": True,
            "recency_cooccurrence_enabled": True,
            "eval_user_cap": 2_000,
            "candidate_union_top_k": 200,
            "candidate_union_batch_size": 500,
            "ranker_candidate_top_k": 100,
            "ranker_train_example_cap": 5_000,
            "ranker_val_example_cap": 1_000,
            "split_eval_example_cap": 2_000,
        },
        "quality-neural": {
            "max_rows_per_category": None,
            "retriever_train_example_cap": 100_000,
            "retriever_quality_min_history": 3,
            "enable_neural_retriever": True,
            "category_backfill_enabled": True,
            "recency_cooccurrence_enabled": True,
            "eval_user_cap": 2_000,
            "candidate_union_top_k": 200,
            "candidate_union_batch_size": 500,
            "ranker_candidate_top_k": 100,
            "ranker_train_example_cap": 5_000,
            "ranker_val_example_cap": 1_000,
            "split_eval_example_cap": 2_000,
        },
        "full": {
            "max_rows_per_category": None,
            "retriever_train_example_cap": None,
            "retriever_quality_min_history": 3,
            "enable_neural_retriever": True,
            "category_backfill_enabled": True,
            "recency_cooccurrence_enabled": True,
            "eval_user_cap": None,
            "candidate_union_top_k": 300,
            "candidate_union_batch_size": 1_000,
            "ranker_candidate_top_k": 200,
            "ranker_train_example_cap": 50_000,
            "ranker_val_example_cap": 5_000,
            "split_eval_example_cap": None,
        },
    }
    field_defaults = {
        key: value.default
        for key, value in PipelineConfig.__dataclass_fields__.items()
        if value.default is not MISSING
    }
    profile_defaults = defaults[config.run_profile]
    for key, value in profile_defaults.items():
        if getattr(config, key) == field_defaults.get(key):
            setattr(config, key, value)
    return config


def _cache_sensitive_config(config: PipelineConfig) -> dict[str, object]:
    keys = {
        "cache_version",
        "run_profile",
        "categories",
        "seed",
        "k_core",
        "dev_mode",
        "dev_fraction",
        "dev_sampling_strategy",
        "dev_hard_negative_multiplier",
        "dev_neutral_multiplier",
        "history_len",
        "max_rows_per_category",
        "train_positive_cap",
        "split_eval_example_cap",
        "review_chunk_rows",
        "text_max_features",
        "text_svd_dim",
    }
    payload = asdict(config)
    payload["base_dir"] = str(config.base_dir)
    return {key: payload[key] for key in keys}


def _should_force_rebuild_for_config(config: PipelineConfig) -> bool:
    config_path = config.artifact_root / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return True
    current = _cache_sensitive_config(config)
    comparable_stored = {key: stored.get(key) for key in current}
    return comparable_stored != current


def _assert_within_workspace(target: Path, workspace: Path) -> None:
    resolved_target = target.resolve()
    resolved_workspace = workspace.resolve()
    if resolved_workspace not in resolved_target.parents and resolved_target != resolved_workspace:
        raise ValueError(f"Refusing to delete path outside workspace: {resolved_target}")


def _safe_rmtree(target: Path, workspace: Path) -> None:
    if target.exists():
        _assert_within_workspace(target, workspace)
        try:
            shutil.rmtree(target)
        except PermissionError as exc:
            raise RuntimeError(
                f"Could not rebuild cached artifacts because {target} is locked by another process. "
                "Close any open notebook/file handles that are using this run, or change CONFIG.run_name and retry."
            ) from exc


def _iter_jsonl(path: Path, max_rows: int | None = None) -> Iterable[Record]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_listlike(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items() if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]


def _parse_price(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    allowed = "".join(ch for ch in text if ch.isdigit() or ch in {".", "-"})
    if not allowed or allowed in {".", "-", "-."}:
        return np.nan
    try:
        return float(allowed)
    except ValueError:
        return np.nan


def _coerce_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return np.nan
    if math.isnan(coerced):
        return np.nan
    return coerced


def _keep_review_record(config: PipelineConfig, category: str, row: Record) -> bool:
    return _keep_review_record_with_fraction(config, category, row, base_keep_fraction=None)


def _count_jsonl_rows(path: Path, max_rows: int | None = None) -> int:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        kept_rows = 0
        for row_number, line in enumerate(handle, start=1):
            if max_rows is not None and row_number > max_rows:
                break
            if line.strip():
                kept_rows += 1
    return kept_rows


def _load_raw_category_row_counts(config: PipelineConfig, force: bool = False) -> dict[str, int]:
    ensure_directories(config)
    counts_path = config.cache_dir / "raw_category_row_counts.json"
    if not force and counts_path.exists():
        try:
            with open(counts_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            counts = {category: int(payload.get(category, 0)) for category in config.categories}
            if set(counts) == set(config.categories):
                return counts
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            counts_path.unlink(missing_ok=True)
    counts: dict[str, int] = {}
    category_bar = _progress(
        config.categories,
        total=len(config.categories),
        desc="Count raw review rows",
        unit="category",
        disable=not config.show_progress,
    )
    for category in category_bar:
        source_path = config.data_dir / REVIEW_FILE_NAMES[category]
        if not source_path.exists():
            raise FileNotFoundError(f"Review file not found: {source_path}")
        counts[category] = _count_jsonl_rows(source_path, max_rows=config.max_rows_per_category)
        category_bar.set_postfix(rows=counts[category])
    with open(counts_path, "w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2)
    return counts


def _category_keep_fraction_plan(
    config: PipelineConfig,
    force: bool = False,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    base_fraction = float(config.dev_fraction)
    if not config.dev_mode:
        return ({category: 1.0 for category in config.categories}, {}, {})
    if config.dev_sampling_strategy != "category_balanced_user":
        return ({category: base_fraction for category in config.categories}, {}, {})
    raw_counts = _load_raw_category_row_counts(config, force=force)
    non_zero_counts = {category: count for category, count in raw_counts.items() if count > 0}
    if not non_zero_counts:
        return ({category: base_fraction for category in config.categories}, {}, raw_counts)
    balanced_target_rows = max(1, int(math.ceil(sum(non_zero_counts.values()) * base_fraction / len(non_zero_counts))))
    keep_fractions: dict[str, float] = {}
    target_rows: dict[str, int] = {}
    for category in config.categories:
        raw_count = int(raw_counts.get(category, 0))
        if raw_count <= 0:
            keep_fractions[category] = 0.0
            target_rows[category] = 0
            continue
        keep_fractions[category] = min(1.0, balanced_target_rows / float(raw_count))
        target_rows[category] = min(raw_count, balanced_target_rows)
    return keep_fractions, target_rows, raw_counts


def _keep_review_record_with_fraction(
    config: PipelineConfig,
    category: str,
    row: Record,
    base_keep_fraction: float | None,
) -> bool:
    if not config.dev_mode:
        return True
    resolved_base_fraction = float(config.dev_fraction if base_keep_fraction is None else base_keep_fraction)
    rating = _coerce_float(row.get("rating"))
    if pd.notna(rating) and rating <= 2.0:
        rating_bucket = "hard_negative"
        keep_fraction = min(1.0, resolved_base_fraction * float(config.dev_hard_negative_multiplier))
    elif pd.notna(rating) and rating == 3.0:
        rating_bucket = "neutral"
        keep_fraction = min(1.0, resolved_base_fraction * float(config.dev_neutral_multiplier))
    else:
        rating_bucket = "positive"
        keep_fraction = resolved_base_fraction

    user_id = str(row.get("user_id") or "").strip()
    if user_id and config.dev_sampling_strategy in {"stratified_user", "category_balanced_user"}:
        key = f"{config.seed}|{config.dev_sampling_strategy}|{category}|{rating_bucket}|{user_id}"
    elif user_id:
        key = f"{config.seed}|user|{user_id}"
    else:
        parent_asin = row.get("parent_asin") or row.get("asin") or ""
        key = f"{config.seed}|fallback|{category}|{rating_bucket}|{parent_asin}|{row.get('timestamp', '')}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value < keep_fraction


def load_reviews(
    config: PipelineConfig,
    categories: Iterable[str] | None = None,
    max_rows_per_category: int | None = 2_000,
) -> pd.DataFrame:
    ensure_directories(config)
    categories = tuple(categories or config.categories)
    rows: list[Record] = []
    category_bar = _progress(categories, total=len(categories), desc="Audit review files", unit="category", disable=not config.show_progress)
    for category in category_bar:
        file_name = REVIEW_FILE_NAMES[category]
        path = config.data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Review file not found: {path}")
        with _progress(desc=f"{category} rows", unit="rows", disable=not config.show_progress) as row_bar:
            batch_count = 0
            for row in _iter_jsonl(path, max_rows=max_rows_per_category):
                rows.append(
                    {
                        "source_category": category,
                        "user_id": row.get("user_id"),
                        "asin": row.get("asin"),
                        "parent_asin": row.get("parent_asin") or row.get("asin"),
                        "rating": row.get("rating"),
                        "timestamp": row.get("timestamp"),
                        "verified_purchase": bool(row.get("verified_purchase", False)),
                        "helpful_vote": row.get("helpful_vote", 0),
                        "title": row.get("title", ""),
                        "text": row.get("text", ""),
                    }
                )
                batch_count += 1
                if batch_count >= 1000:
                    row_bar.update(batch_count)
                    batch_count = 0
            if batch_count:
                row_bar.update(batch_count)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    return df


def download_metadata_files(
    config: PipelineConfig,
    categories: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    ensure_directories(config)
    categories = tuple(categories or config.categories)
    downloaded: dict[str, Path] = {}
    category_bar = _progress(categories, total=len(categories), desc="Download metadata", unit="category", disable=not config.show_progress)
    for category in category_bar:
        if category not in METADATA_URLS:
            raise KeyError(f"No metadata URL configured for category: {category}")
        url = METADATA_URLS[category]
        target = config.metadata_dir / f"meta_{category}.jsonl.gz"
        if force and target.exists():
            target.unlink()
        if not target.exists():
            temp_target = target.with_suffix(target.suffix + ".part")
            if temp_target.exists():
                temp_target.unlink()
            last_error: Exception | None = None
            for _ in range(3):
                try:
                    with urllib.request.urlopen(url) as response, open(temp_target, "wb") as handle:
                        total_bytes = int(response.headers.get("Content-Length") or 0)
                        with _progress(
                            total=total_bytes if total_bytes > 0 else None,
                            desc=f"{category} bytes",
                            unit="B",
                            disable=not config.show_progress,
                            leave=False,
                        ) as download_bar:
                            while True:
                                chunk = response.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                                download_bar.update(len(chunk))
                    temp_target.replace(target)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    temp_target.unlink(missing_ok=True)
            if last_error is not None:
                raise last_error
        downloaded[category] = target
    return downloaded


def load_metadata(
    config: PipelineConfig,
    categories: Iterable[str] | None = None,
    download_if_missing: bool | None = None,
    max_rows_per_category: int | None = None,
    valid_parent_asins: set[str] | None = None,
) -> pd.DataFrame:
    ensure_directories(config)
    categories = tuple(categories or config.categories)
    if download_if_missing is None:
        download_if_missing = config.metadata_download_if_missing
    missing = [c for c in categories if not (config.metadata_dir / f"meta_{c}.jsonl.gz").exists()]
    if missing and download_if_missing:
        download_metadata_files(config, categories=missing, force=False)
    rows: list[Record] = []
    category_bar = _progress(categories, total=len(categories), desc="Load metadata", unit="category", disable=not config.show_progress)
    for category in category_bar:
        path = config.metadata_dir / f"meta_{category}.jsonl.gz"
        if not path.exists():
            continue
        with _progress(desc=f"{category} metadata rows", unit="rows", disable=not config.show_progress) as row_bar:
            batch_count = 0
            for row in _iter_jsonl(path, max_rows=max_rows_per_category):
                parent_asin = row.get("parent_asin")
                if valid_parent_asins is not None and parent_asin not in valid_parent_asins:
                    continue
                categories_list = _normalize_listlike(row.get("categories"))
                description_list = _normalize_listlike(row.get("description"))
                features_list = _normalize_listlike(row.get("features"))
                bought_together_list = _normalize_listlike(row.get("bought_together"))
                rows.append(
                    {
                        "source_category": category,
                        "parent_asin": parent_asin,
                        "meta_title": row.get("title") or "",
                        "store": row.get("store") or "",
                        "categories_text": " | ".join(categories_list),
                        "description_text": " ".join(description_list),
                        "features_text": " ".join(features_list),
                        "bought_together_text": " | ".join(bought_together_list),
                        "price": _parse_price(row.get("price")),
                        "average_rating": row.get("average_rating"),
                        "rating_number": row.get("rating_number"),
                    }
                )
                batch_count += 1
                if batch_count >= 1000:
                    row_bar.update(batch_count)
                    batch_count = 0
            if batch_count:
                row_bar.update(batch_count)
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        metadata = pd.DataFrame(
            columns=[
                "source_category",
                "parent_asin",
                "meta_title",
                "store",
                "categories_text",
                "description_text",
                "features_text",
                "bought_together_text",
                "price",
                "average_rating",
                "rating_number",
            ]
        )
    metadata = metadata.dropna(subset=["parent_asin"]).drop_duplicates(subset=["parent_asin"])
    return metadata


def _flush_records_to_parquet(records: list[Record], output_path: Path) -> None:
    if records:
        pd.DataFrame(records).to_parquet(output_path, index=False)


def _extract_review_signals(config: PipelineConfig, force: bool = False) -> tuple[pd.DataFrame, Path, Path]:
    ensure_directories(config)
    positive_dir = config.cache_dir / "positive_chunks"
    hard_negative_dir = config.cache_dir / "hard_negative_chunks"
    raw_stats_path = config.cache_dir / "raw_review_stats.csv"
    if force:
        _safe_rmtree(positive_dir, config.base_dir)
        _safe_rmtree(hard_negative_dir, config.base_dir)
        raw_stats_path.unlink(missing_ok=True)
    if positive_dir.exists() and hard_negative_dir.exists() and raw_stats_path.exists():
        LOGGER.info("Using cached review signal chunks: %s", config.cache_dir)
        return pd.read_csv(raw_stats_path), positive_dir, hard_negative_dir
    LOGGER.info(
        "Extracting review signals: categories=%s max_rows_per_category=%s dev_mode=%s dev_fraction=%s",
        ",".join(config.categories),
        config.max_rows_per_category,
        config.dev_mode,
        config.dev_fraction,
    )
    positive_dir.mkdir(parents=True, exist_ok=True)
    hard_negative_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[Record] = []
    category_keep_fractions, category_target_rows, raw_category_counts = _category_keep_fraction_plan(config, force=force)
    category_bar = _progress(config.categories, total=len(config.categories), desc="Extract review signals", unit="category", disable=not config.show_progress)
    for category in category_bar:
        source_path = config.data_dir / REVIEW_FILE_NAMES[category]
        if not source_path.exists():
            raise FileNotFoundError(f"Review file not found: {source_path}")
        LOGGER.info("Extracting review signals for category=%s source=%s", category, source_path)
        base_keep_fraction = float(category_keep_fractions.get(category, config.dev_fraction))
        positive_records: list[Record] = []
        negative_records: list[Record] = []
        part_positive = 0
        part_negative = 0
        rating_counter: Counter = Counter()
        verified_count = 0
        helpful_nonzero_count = 0
        text_word_sum = 0
        raw_rows_seen = 0
        kept_rows = 0
        min_timestamp: int | None = None
        max_timestamp: int | None = None
        with _progress(desc=f"{category} review rows", unit="rows", disable=not config.show_progress) as row_bar:
            batch_count = 0
            for row in _iter_jsonl(source_path, max_rows=config.max_rows_per_category):
                parent_asin = row.get("parent_asin") or row.get("asin")
                user_id = row.get("user_id")
                rating = row.get("rating")
                timestamp = row.get("timestamp")
                if not parent_asin or not user_id or rating is None or timestamp is None:
                    continue
                raw_rows_seen += 1
                if not _keep_review_record_with_fraction(config, category, row, base_keep_fraction):
                    continue
                kept_rows += 1
                rating_counter[float(rating)] += 1
                verified_count += int(bool(row.get("verified_purchase", False)))
                helpful_nonzero_count += int((row.get("helpful_vote") or 0) > 0)
                text_word_sum += len(str(row.get("text") or "").split())
                min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
                max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)
                record = {
                    "source_category": category,
                    "user_id": str(user_id),
                    "asin": str(row.get("asin") or ""),
                    "parent_asin": str(parent_asin),
                    "rating": float(rating),
                    "timestamp": int(timestamp),
                    "verified_purchase": int(bool(row.get("verified_purchase", False))),
                    "helpful_vote": int(row.get("helpful_vote") or 0),
                    "review_title": str(row.get("title") or ""),
                    "review_text": str(row.get("text") or ""),
                }
                if float(rating) >= 4.0:
                    positive_records.append(record)
                    if len(positive_records) >= config.review_chunk_rows:
                        _flush_records_to_parquet(positive_records, positive_dir / f"{category}_part_{part_positive:04d}.parquet")
                        positive_records.clear()
                        part_positive += 1
                elif float(rating) <= 2.0:
                    negative_records.append(record)
                    if len(negative_records) >= config.review_chunk_rows:
                        _flush_records_to_parquet(negative_records, hard_negative_dir / f"{category}_part_{part_negative:04d}.parquet")
                        negative_records.clear()
                        part_negative += 1
                batch_count += 1
                if batch_count >= 5000:
                    row_bar.update(batch_count)
                    batch_count = 0
                    row_bar.set_postfix(
                        kept=kept_rows,
                        pos=int(sum(count for value, count in rating_counter.items() if value >= 4.0)),
                        neg=int(sum(count for value, count in rating_counter.items() if value <= 2.0)),
                    )
            if batch_count:
                row_bar.update(batch_count)
        if positive_records:
            _flush_records_to_parquet(positive_records, positive_dir / f"{category}_part_{part_positive:04d}.parquet")
        if negative_records:
            _flush_records_to_parquet(negative_records, hard_negative_dir / f"{category}_part_{part_negative:04d}.parquet")
        LOGGER.info(
            "Category extraction complete: category=%s raw_rows=%s kept_rows=%s positives=%s hard_negatives=%s keep_rate=%.3f",
            category,
            f"{raw_rows_seen:,}",
            f"{kept_rows:,}",
            f"{int(sum(count for value, count in rating_counter.items() if value >= 4.0)):,}",
            f"{int(sum(count for value, count in rating_counter.items() if value <= 2.0)):,}",
            kept_rows / max(raw_rows_seen, 1),
        )
        stats_rows.append(
            {
                "source_category": category,
                "sampling_strategy": config.dev_sampling_strategy if config.dev_mode else "full_corpus",
                "category_raw_rows": int(raw_category_counts.get(category, raw_rows_seen)),
                "sampling_target_rows": int(category_target_rows.get(category, 0)),
                "base_keep_fraction": float(base_keep_fraction if config.dev_mode else 1.0),
                "raw_rows_seen": raw_rows_seen,
                "kept_rows": kept_rows,
                "effective_keep_rate": kept_rows / max(raw_rows_seen, 1),
                "rows": kept_rows,
                "avg_rating": sum(float(value) * count for value, count in rating_counter.items()) / max(kept_rows, 1),
                "verified_rate": verified_count / max(kept_rows, 1),
                "helpful_nonzero_rate": helpful_nonzero_count / max(kept_rows, 1),
                "avg_text_words": text_word_sum / max(kept_rows, 1),
                "positive_rows": int(sum(count for value, count in rating_counter.items() if value >= 4.0)),
                "neutral_rows": int(rating_counter.get(3.0, 0)),
                "hard_negative_rows": int(sum(count for value, count in rating_counter.items() if value <= 2.0)),
                "min_timestamp": min_timestamp,
                "max_timestamp": max_timestamp,
                "rating_histogram": json.dumps({str(k): int(v) for k, v in sorted(rating_counter.items())}),
            }
        )
    raw_stats = pd.DataFrame(stats_rows)
    raw_stats.to_csv(raw_stats_path, index=False)
    LOGGER.info("Review signal extraction complete: stats_path=%s", raw_stats_path)
    return raw_stats, positive_dir, hard_negative_dir


def _count_entities_from_chunks(
    chunk_paths: list[Path],
    valid_users: set[str] | None = None,
    valid_items: set[str] | None = None,
    show_progress: bool = False,
    desc: str | None = None,
) -> tuple[Counter, Counter, int]:
    user_counts: Counter = Counter()
    item_counts: Counter = Counter()
    kept_rows = 0
    chunk_bar = _progress(chunk_paths, total=len(chunk_paths), desc=desc or "Count chunk entities", unit="chunk", disable=not show_progress)
    for path in chunk_bar:
        df = pd.read_parquet(path, columns=["user_id", "parent_asin"])
        if valid_users is not None:
            df = df[df["user_id"].isin(valid_users)]
        if valid_items is not None:
            df = df[df["parent_asin"].isin(valid_items)]
        kept_rows += len(df)
        if df.empty:
            continue
        user_counts.update(df["user_id"].value_counts().to_dict())
        item_counts.update(df["parent_asin"].value_counts().to_dict())
        chunk_bar.set_postfix(rows=kept_rows)
    return user_counts, item_counts, kept_rows


def _compute_k_core_sets(config: PipelineConfig, positive_dir: Path, force: bool = False) -> tuple[pd.DataFrame, set[str], set[str]]:
    ensure_directories(config)
    kcore_stats_path = config.cache_dir / "kcore_stats.csv"
    valid_users_path = config.cache_dir / "valid_users.parquet"
    valid_items_path = config.cache_dir / "valid_items.parquet"
    if not force and kcore_stats_path.exists() and valid_users_path.exists() and valid_items_path.exists():
        stats = pd.read_csv(kcore_stats_path)
        valid_users = set(pd.read_parquet(valid_users_path)["user_id"].astype(str))
        valid_items = set(pd.read_parquet(valid_items_path)["parent_asin"].astype(str))
        LOGGER.info(
            "Using cached k-core sets: users=%s items=%s",
            f"{len(valid_users):,}",
            f"{len(valid_items):,}",
        )
        return stats, valid_users, valid_items
    positive_chunks = sorted(positive_dir.glob("*.parquet"))
    if not positive_chunks:
        raise FileNotFoundError(f"No positive chunks found under {positive_dir}")
    LOGGER.info("Computing k-core sets: k_core=%s positive_chunks=%s", config.k_core, len(positive_chunks))
    iteration_rows: list[Record] = []
    valid_users: set[str] | None = None
    valid_items: set[str] | None = None
    iteration_bar = _progress(range(1, 20), total=19, desc="Compute k-core", unit="iter", disable=not config.show_progress)
    for iteration in iteration_bar:
        user_counts, item_counts, kept_rows = _count_entities_from_chunks(
            positive_chunks,
            valid_users=valid_users,
            valid_items=valid_items,
            show_progress=config.show_progress,
            desc=f"k-core pass {iteration}",
        )
        next_valid_users = {user for user, count in user_counts.items() if count >= config.k_core}
        next_valid_items = {item for item, count in item_counts.items() if count >= config.k_core}
        iteration_bar.set_postfix(rows=kept_rows, users=len(next_valid_users), items=len(next_valid_items))
        LOGGER.info(
            "k-core pass %s: rows=%s users=%s items=%s",
            iteration,
            f"{kept_rows:,}",
            f"{len(next_valid_users):,}",
            f"{len(next_valid_items):,}",
        )
        iteration_rows.append(
            {
                "iteration": iteration,
                "rows_after_filter": kept_rows,
                "users_after_filter": len(next_valid_users),
                "items_after_filter": len(next_valid_items),
            }
        )
        if next_valid_users == (valid_users or set()) and next_valid_items == (valid_items or set()):
            valid_users = next_valid_users
            valid_items = next_valid_items
            break
        valid_users = next_valid_users
        valid_items = next_valid_items
    if valid_users is None or valid_items is None:
        raise RuntimeError("Failed to compute k-core sets")
    pd.DataFrame(iteration_rows).to_csv(kcore_stats_path, index=False)
    pd.DataFrame({"user_id": sorted(valid_users)}).to_parquet(valid_users_path, index=False)
    pd.DataFrame({"parent_asin": sorted(valid_items)}).to_parquet(valid_items_path, index=False)
    LOGGER.info("k-core complete: users=%s items=%s", f"{len(valid_users):,}", f"{len(valid_items):,}")
    return pd.DataFrame(iteration_rows), valid_users, valid_items


def _load_filtered_interactions(
    positive_dir: Path,
    hard_negative_dir: Path,
    valid_users: set[str],
    valid_items: set[str],
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = [
        "source_category",
        "user_id",
        "asin",
        "parent_asin",
        "rating",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
        "review_title",
        "review_text",
    ]
    positive_frames: list[pd.DataFrame] = []
    positive_paths = sorted(positive_dir.glob("*.parquet"))
    LOGGER.info("Loading filtered positive chunks: chunks=%s", len(positive_paths))
    for path in _progress(positive_paths, total=len(positive_paths), desc="Load positive chunks", unit="chunk", disable=not show_progress):
        df = pd.read_parquet(path)
        df = df[df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)]
        if not df.empty:
            positive_frames.append(df)
    hard_negative_frames: list[pd.DataFrame] = []
    hard_negative_paths = sorted(hard_negative_dir.glob("*.parquet"))
    LOGGER.info("Loading filtered hard-negative chunks: chunks=%s", len(hard_negative_paths))
    for path in _progress(hard_negative_paths, total=len(hard_negative_paths), desc="Load hard-negative chunks", unit="chunk", disable=not show_progress):
        df = pd.read_parquet(path)
        df = df[df["user_id"].isin(valid_users)]
        if not df.empty:
            hard_negative_frames.append(df)
    positive_df = pd.concat(positive_frames, ignore_index=True) if positive_frames else pd.DataFrame(columns=base_columns)
    hard_negative_df = pd.concat(hard_negative_frames, ignore_index=True) if hard_negative_frames else pd.DataFrame(columns=base_columns)
    for frame in [positive_df, hard_negative_df]:
        if not frame.empty:
            frame["timestamp_dt"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
    LOGGER.info(
        "Filtered interaction loading complete: positives=%s hard_negatives=%s",
        f"{len(positive_df):,}",
        f"{len(hard_negative_df):,}",
    )
    return positive_df, hard_negative_df


def _fit_text_features(config: PipelineConfig, item_table: pd.DataFrame) -> tuple[TfidfVectorizer, TruncatedSVD, np.ndarray]:
    texts = item_table["item_text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=config.text_max_features, ngram_range=(1, 2), min_df=2)
    LOGGER.info(
        "Fitting text features: items=%s max_features=%s svd_dim=%s",
        f"{len(item_table):,}",
        config.text_max_features,
        config.text_svd_dim,
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        text_matrix = np.zeros((len(item_table), config.text_svd_dim), dtype=np.float32)
        svd = TruncatedSVD(n_components=1, random_state=config.seed)
        return vectorizer, svd, text_matrix
    target_dim = min(config.text_svd_dim, max(2, tfidf.shape[1] - 1)) if tfidf.shape[1] > 2 else min(tfidf.shape[1], 2)
    if tfidf.shape[1] <= 1 or tfidf.shape[0] <= 2:
        text_matrix = np.zeros((len(item_table), config.text_svd_dim), dtype=np.float32)
        svd = TruncatedSVD(n_components=1, random_state=config.seed)
        return vectorizer, svd, text_matrix
    svd = TruncatedSVD(n_components=target_dim, random_state=config.seed)
    reduced = svd.fit_transform(tfidf)
    if reduced.shape[1] < config.text_svd_dim:
        reduced = np.pad(reduced, ((0, 0), (0, config.text_svd_dim - reduced.shape[1])))
    LOGGER.info("Text feature fitting complete: tfidf_terms=%s output_shape=%s", tfidf.shape[1], reduced.shape)
    return vectorizer, svd, reduced.astype(np.float32)


def _build_item_features(
    config: PipelineConfig,
    interactions: pd.DataFrame,
    hard_negatives: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, TfidfVectorizer, TruncatedSVD]:
    review_stats = (
        pd.concat([interactions.assign(label_bucket="positive"), hard_negatives.assign(label_bucket="hard_negative")], ignore_index=True)
        .groupby("parent_asin")
        .agg(
            source_category=("source_category", "last"),
            train_positive_count=("label_bucket", lambda x: int((x == "positive").sum())),
            train_hard_negative_count=("label_bucket", lambda x: int((x == "hard_negative").sum())),
            verified_purchase_rate=("verified_purchase", "mean"),
            helpful_vote_mean=("helpful_vote", "mean"),
            helpful_nonzero_rate=("helpful_vote", lambda x: float((x > 0).mean())),
            observed_review_count=("rating", "size"),
            observed_average_rating=("rating", "mean"),
            item_last_timestamp=("timestamp", "max"),
            fallback_title=("review_title", lambda x: next((str(v) for v in x if str(v).strip()), "")),
            fallback_text=("review_text", lambda x: " ".join(str(v) for v in x.head(3) if str(v).strip())),
        )
        .reset_index()
    )
    item_table = review_stats.merge(metadata, on=["parent_asin", "source_category"], how="left")
    item_table["title"] = item_table["meta_title"].fillna("").replace("", np.nan).fillna(item_table["fallback_title"]).fillna("")
    item_table["price"] = item_table["price"].astype(float)
    item_table["average_rating"] = pd.to_numeric(item_table["average_rating"], errors="coerce")
    item_table["rating_number"] = pd.to_numeric(item_table["rating_number"], errors="coerce")
    item_table["average_rating"] = item_table["average_rating"].fillna(item_table["observed_average_rating"])
    item_table["rating_number"] = item_table["rating_number"].fillna(item_table["observed_review_count"])
    item_table["price"] = item_table["price"].fillna(item_table["price"].median() if not item_table["price"].dropna().empty else 0.0)
    item_table["train_positive_count"] = item_table["train_positive_count"].fillna(0).astype(int)
    item_table["train_hard_negative_count"] = item_table["train_hard_negative_count"].fillna(0).astype(int)
    item_table["verified_purchase_rate"] = item_table["verified_purchase_rate"].fillna(0.0)
    item_table["helpful_vote_mean"] = item_table["helpful_vote_mean"].fillna(0.0)
    item_table["helpful_nonzero_rate"] = item_table["helpful_nonzero_rate"].fillna(0.0)
    item_table["days_since_last_interaction"] = (
        (item_table["item_last_timestamp"].max() - item_table["item_last_timestamp"]) / (1000 * 60 * 60 * 24)
    ).fillna(0.0)
    item_table["item_text"] = (
        item_table["title"].fillna("")
        + " "
        + item_table["store"].fillna("")
        + " "
        + item_table["categories_text"].fillna("")
        + " "
        + item_table["description_text"].fillna("")
        + " "
        + item_table["features_text"].fillna("")
        + " "
        + item_table["fallback_text"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    item_table = item_table.sort_values(["source_category", "parent_asin"]).reset_index(drop=True)
    item_table["item_idx"] = np.arange(1, len(item_table) + 1, dtype=np.int32)
    category_codes = {category: index + 1 for index, category in enumerate(sorted(item_table["source_category"].dropna().unique()))}
    item_table["source_category_idx"] = item_table["source_category"].map(category_codes).astype(np.int32)
    vectorizer, svd, text_matrix = _fit_text_features(config, item_table)
    item_table["log_rating_number"] = np.log1p(item_table["rating_number"].fillna(0.0))
    item_table["log_positive_count"] = np.log1p(item_table["train_positive_count"].fillna(0.0))
    keep_columns = [
        "item_idx",
        "parent_asin",
        "source_category",
        "source_category_idx",
        "title",
        "price",
        "average_rating",
        "rating_number",
        "train_positive_count",
        "train_hard_negative_count",
        "verified_purchase_rate",
        "helpful_vote_mean",
        "helpful_nonzero_rate",
        "days_since_last_interaction",
        "log_rating_number",
        "log_positive_count",
    ]
    item_table = item_table[keep_columns].copy()
    return item_table, text_matrix, vectorizer, svd


def prepare_corpus(
    config: PipelineConfig,
    force_rebuild: bool = False,
) -> PreparedArtifacts:
    ensure_directories(config)
    set_global_seed(config.seed)
    if not force_rebuild and _should_force_rebuild_for_config(config):
        force_rebuild = True
        LOGGER.info("Config change detected for cached corpus artifacts. Rebuilding cache for the current settings.")
    raw_stats, positive_dir, hard_negative_dir = _extract_review_signals(config, force=force_rebuild)
    kcore_stats, valid_users, valid_items = _compute_k_core_sets(config, positive_dir, force=force_rebuild)
    interactions_path = config.cache_dir / "filtered_positive_interactions.parquet"
    hard_negatives_path = config.cache_dir / "filtered_hard_negatives.parquet"
    item_features_path = config.cache_dir / "item_features.parquet"
    item_text_path = config.cache_dir / "item_text_matrix.npy"
    vectorizer_path = config.cache_dir / "tfidf_vectorizer.pkl"
    svd_path = config.cache_dir / "text_svd.pkl"
    built_in_memory = False
    if force_rebuild or not all(path.exists() for path in [interactions_path, hard_negatives_path, item_features_path, item_text_path, vectorizer_path, svd_path]):
        LOGGER.info("Building prepared corpus artifacts from chunks")
        interactions, hard_negatives = _load_filtered_interactions(
            positive_dir,
            hard_negative_dir,
            valid_users,
            valid_items,
            show_progress=config.show_progress,
        )
        metadata_parent_asins = set(valid_items).union(set(hard_negatives["parent_asin"].astype(str).unique()))
        LOGGER.info("Loading metadata for %s candidate items", f"{len(metadata_parent_asins):,}")
        metadata = load_metadata(
            config,
            categories=config.categories,
            download_if_missing=config.metadata_download_if_missing,
            valid_parent_asins=metadata_parent_asins,
        )
        item_features, item_text_matrix, vectorizer, svd = _build_item_features(config, interactions, hard_negatives, metadata)
        valid_item_ids = set(item_features["parent_asin"])
        interactions = interactions[interactions["parent_asin"].isin(valid_item_ids)].copy()
        hard_negatives = hard_negatives[hard_negatives["parent_asin"].isin(valid_item_ids)].copy()
        if interactions.empty or item_features.empty:
            raise ValueError(
                "No interactions survived preprocessing. If dev_mode is enabled, increase dev_fraction or lower k_core. "
                "If dev_mode is disabled, inspect the raw review files and metadata coverage."
            )
        interactions.to_parquet(interactions_path, index=False)
        hard_negatives.to_parquet(hard_negatives_path, index=False)
        item_features.to_parquet(item_features_path, index=False)
        np.save(item_text_path, item_text_matrix)
        with open(vectorizer_path, "wb") as handle:
            pickle.dump(vectorizer, handle)
        with open(svd_path, "wb") as handle:
            pickle.dump(svd, handle)
        built_in_memory = True
        del metadata
        gc.collect()
    if not built_in_memory:
        LOGGER.info("Loading prepared corpus artifacts from cache: %s", config.cache_dir)
        interactions = pd.read_parquet(interactions_path)
        hard_negatives = pd.read_parquet(hard_negatives_path)
        item_features = pd.read_parquet(item_features_path)
        item_text_matrix = np.load(item_text_path, mmap_mode="r" if config.memory_map_item_text else None)
        with open(vectorizer_path, "rb") as handle:
            vectorizer = pickle.load(handle)
        with open(svd_path, "rb") as handle:
            svd = pickle.load(handle)
    interactions["timestamp_dt"] = pd.to_datetime(interactions["timestamp"], unit="ms", utc=True, errors="coerce")
    hard_negatives["timestamp_dt"] = pd.to_datetime(hard_negatives["timestamp"], unit="ms", utc=True, errors="coerce")
    item_id_to_idx = dict(zip(item_features["parent_asin"], item_features["item_idx"]))
    item_idx_to_id = dict(zip(item_features["item_idx"], item_features["parent_asin"]))
    category_to_idx = dict(zip(item_features["source_category"], item_features["source_category_idx"]))
    save_config(config)
    LOGGER.info(
        "Prepared corpus ready: interactions=%s hard_negatives=%s items=%s cache_dir=%s",
        f"{len(interactions):,}",
        f"{len(hard_negatives):,}",
        f"{len(item_features):,}",
        config.cache_dir,
    )
    return PreparedArtifacts(
        config=config,
        raw_review_stats=raw_stats,
        kcore_stats=kcore_stats,
        interactions=interactions,
        hard_negatives=hard_negatives,
        item_features=item_features,
        item_text_matrix=item_text_matrix,
        vectorizer=vectorizer,
        svd=svd,
        item_id_to_idx=item_id_to_idx,
        item_idx_to_id=item_idx_to_id,
        category_to_idx=category_to_idx,
    )


def _sample_training_examples(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(df) <= cap:
        return df.reset_index(drop=True)
    sampled_parts: list[pd.DataFrame] = []
    target_month = (
        pd.to_datetime(df["target_timestamp"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m")
    )
    strata = df.assign(target_month=target_month.fillna("unknown"))
    grouped = strata.groupby(["target_source_category", "target_month"], group_keys=False)
    total_rows = len(df)
    rng = np.random.default_rng(seed)
    for _, group in grouped:
        desired = max(1, int(round((len(group) / total_rows) * cap)))
        sampled_parts.append(group.sample(n=min(desired, len(group)), random_state=int(rng.integers(0, 1_000_000))))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > cap:
        sampled = sampled.sample(n=cap, random_state=seed)
    if len(sampled) < cap:
        remaining_ids = set(sampled["example_id"])
        remainder = strata[~strata["example_id"].isin(remaining_ids)]
        if not remainder.empty:
            extra = remainder.sample(n=min(cap - len(sampled), len(remainder)), random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.drop(columns=["target_month"], errors="ignore").reset_index(drop=True)


def _append_reservoir_sample(
    rows: list[Record],
    record: Record,
    *,
    seen_count: int,
    cap: int | None,
    rng: np.random.Generator,
) -> int:
    seen_count += 1
    if cap is None or cap <= 0 or len(rows) < cap:
        rows.append(record)
        return seen_count
    replacement_index = int(rng.integers(0, seen_count))
    if replacement_index < cap:
        rows[replacement_index] = record
    return seen_count


def _compute_prefix_features(
    prefix_rows: pd.DataFrame,
    target_timestamp_ms: int,
    item_to_category: dict[int, str],
    categories: tuple[str, ...],
) -> Record:
    history_item_idxs = prefix_rows["item_idx"].tolist()
    history_ratings = prefix_rows["rating"].astype(float).tolist()
    history_verified = prefix_rows["verified_purchase"].astype(float).tolist()
    history_timestamps = prefix_rows["timestamp"].astype(int).tolist()
    category_counts = Counter(item_to_category[item_idx] for item_idx in history_item_idxs)
    total_history = max(len(history_item_idxs), 1)
    days_since_last = 0.0
    avg_days_between = 0.0
    if history_timestamps:
        days_since_last = max((target_timestamp_ms - history_timestamps[-1]) / (1000 * 60 * 60 * 24), 0.0)
    if len(history_timestamps) > 1:
        diffs = np.diff(history_timestamps) / (1000 * 60 * 60 * 24)
        avg_days_between = float(np.mean(diffs))
    return {
        "history_item_idxs": history_item_idxs,
        "history_length": len(history_item_idxs),
        "user_interaction_count": len(history_item_idxs),
        "user_mean_rating": float(np.mean(history_ratings)) if history_ratings else 0.0,
        "user_verified_rate": float(np.mean(history_verified)) if history_verified else 0.0,
        "days_since_last": days_since_last,
        "avg_days_between": avg_days_between,
        **{f"pref_{category}": category_counts.get(category, 0) / total_history for category in categories},
    }


def make_splits(prepared: PreparedArtifacts) -> SplitArtifacts:
    config = prepared.config
    set_global_seed(config.seed)
    LOGGER.info("Building chronological train/validation/test splits")
    item_to_category = dict(zip(prepared.item_features["item_idx"], prepared.item_features["source_category"]))
    interactions = prepared.interactions.copy()
    interactions["item_idx"] = interactions["parent_asin"].map(prepared.item_id_to_idx).astype(np.int32)
    interactions = interactions.sort_values(["user_id", "timestamp", "item_idx"]).reset_index(drop=True)
    split_rows_train: list[Record] = []
    split_rows_val: list[Record] = []
    split_rows_test: list[Record] = []
    train_seen_map: dict[str, set[int]] = {}
    val_seen_map: dict[str, set[int]] = {}
    test_seen_map: dict[str, set[int]] = {}
    train_sequence_rows = 0
    train_seen_count = 0
    val_seen_count = 0
    test_seen_count = 0
    rng = np.random.default_rng(config.seed)
    example_id = 1
    for user_id, group in interactions.groupby("user_id", sort=False):
        if len(group) < max(config.k_core, 3):
            continue
        group = group.reset_index(drop=True)
        train_group = group.iloc[:-2].copy()
        val_row = group.iloc[-2]
        test_row = group.iloc[-1]
        if len(train_group) < 2:
            continue
        item_list = group["item_idx"].tolist()
        user_key = str(user_id)
        train_seen_map[user_key] = set(item_list[:-2])
        val_seen_map[user_key] = set(item_list[:-2])
        test_seen_map[user_key] = set(item_list[:-1])
        train_sequence_rows += len(group)
        for target_pos in range(1, len(train_group)):
            prefix = train_group.iloc[max(0, target_pos - config.history_len):target_pos]
            target_row = train_group.iloc[target_pos]
            prefix_features = _compute_prefix_features(prefix, int(target_row["timestamp"]), item_to_category, config.categories)
            train_record = {
                "example_id": example_id,
                "split": "train",
                "user_id": user_id,
                "target_item_idx": int(target_row["item_idx"]),
                "target_parent_asin": target_row["parent_asin"],
                "target_source_category": target_row["source_category"],
                "target_timestamp": pd.to_datetime(int(target_row["timestamp"]), unit="ms", utc=True),
                **prefix_features,
            }
            train_seen_count = _append_reservoir_sample(
                split_rows_train,
                train_record,
                seen_count=train_seen_count,
                cap=config.train_positive_cap,
                rng=rng,
            )
            example_id += 1
        prefix = train_group.iloc[max(0, len(train_group) - config.history_len):].copy()
        prefix_features = _compute_prefix_features(prefix, int(val_row["timestamp"]), item_to_category, config.categories)
        val_record = {
            "example_id": example_id,
            "split": "val",
            "user_id": user_id,
            "target_item_idx": int(val_row["item_idx"]),
            "target_parent_asin": val_row["parent_asin"],
            "target_source_category": val_row["source_category"],
            "target_timestamp": pd.to_datetime(int(val_row["timestamp"]), unit="ms", utc=True),
            **prefix_features,
        }
        val_seen_count = _append_reservoir_sample(
            split_rows_val,
            val_record,
            seen_count=val_seen_count,
            cap=config.split_eval_example_cap,
            rng=rng,
        )
        example_id += 1
        test_prefix_end = len(group) - 1
        prefix = group.iloc[max(0, test_prefix_end - config.history_len):test_prefix_end].copy()
        prefix_features = _compute_prefix_features(prefix, int(test_row["timestamp"]), item_to_category, config.categories)
        test_record = {
            "example_id": example_id,
            "split": "test",
            "user_id": user_id,
            "target_item_idx": int(test_row["item_idx"]),
            "target_parent_asin": test_row["parent_asin"],
            "target_source_category": test_row["source_category"],
            "target_timestamp": pd.to_datetime(int(test_row["timestamp"]), unit="ms", utc=True),
            **prefix_features,
        }
        test_seen_count = _append_reservoir_sample(
            split_rows_test,
            test_record,
            seen_count=test_seen_count,
            cap=config.split_eval_example_cap,
            rng=rng,
        )
        example_id += 1
    train_examples = pd.DataFrame(split_rows_train)
    val_examples = pd.DataFrame(split_rows_val)
    test_examples = pd.DataFrame(split_rows_test)
    del split_rows_train
    del split_rows_val
    del split_rows_test
    del interactions
    gc.collect()
    if train_examples.empty or val_examples.empty or test_examples.empty:
        raise RuntimeError(
            "Split generation produced an empty train/val/test split. "
            "Increase dev_fraction or max_rows_per_category, or lower k_core if you are using a very small dev sample."
        )
    LOGGER.info(
        "Split examples built: train=%s/%s val=%s/%s test=%s/%s train_positive_cap=%s split_eval_example_cap=%s",
        f"{len(train_examples):,}",
        f"{train_seen_count:,}",
        f"{len(val_examples):,}",
        f"{val_seen_count:,}",
        f"{len(test_examples):,}",
        f"{test_seen_count:,}",
        config.train_positive_cap,
        config.split_eval_example_cap,
    )
    unique_users = pd.Index(pd.concat([train_examples["user_id"], val_examples["user_id"], test_examples["user_id"]]).astype(str).unique())
    user_id_to_idx = {user_id: idx + 1 for idx, user_id in enumerate(unique_users)}
    user_idx_to_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    needed_users = set(user_id_to_idx)
    train_seen_map = {user_id: seen for user_id, seen in train_seen_map.items() if user_id in needed_users}
    val_seen_map = {user_id: seen for user_id, seen in val_seen_map.items() if user_id in needed_users}
    test_seen_map = {user_id: seen for user_id, seen in test_seen_map.items() if user_id in needed_users}
    for frame in [train_examples, val_examples, test_examples]:
        frame["user_idx"] = frame["user_id"].map(user_id_to_idx).astype(np.int32)
    LOGGER.info(
        "Split state ready: users=%s train_sequence_rows=%s",
        f"{len(user_id_to_idx):,}",
        f"{train_sequence_rows:,}",
    )
    train_item_popularity = Counter(train_examples["target_item_idx"].tolist())
    category_item_popularity: dict[str, Counter] = defaultdict(Counter)
    for _, row in train_examples.iterrows():
        category_item_popularity[str(row["target_source_category"])][int(row["target_item_idx"])] += 1
    cooccurrence: dict[int, Counter] = defaultdict(Counter)
    for _, row in train_examples.iterrows():
        for history_item in row["history_item_idxs"]:
            cooccurrence[int(history_item)][int(row["target_item_idx"])] += 1
    hard_negative_history: dict[str, list[tuple[int, int]]] = defaultdict(list)
    hard_negatives = prepared.hard_negatives[["user_id", "timestamp", "parent_asin"]].copy()
    hard_negatives = hard_negatives[hard_negatives["user_id"].astype(str).isin(needed_users)]
    hard_negatives["item_idx"] = hard_negatives["parent_asin"].map(prepared.item_id_to_idx)
    hard_negatives = hard_negatives.dropna(subset=["item_idx"])
    for _, row in hard_negatives.iterrows():
        hard_negative_history[str(row["user_id"])].append((int(row["timestamp"]), int(row["item_idx"])))
    for user_id in list(hard_negative_history.keys()):
        hard_negative_history[user_id] = sorted(hard_negative_history[user_id], key=lambda value: value[0])
    del hard_negatives
    gc.collect()
    train_examples = train_examples.sort_values("target_timestamp").reset_index(drop=True)
    val_examples = val_examples.sort_values("target_timestamp").reset_index(drop=True)
    test_examples = test_examples.sort_values("target_timestamp").reset_index(drop=True)
    return SplitArtifacts(
        config=config,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        user_id_to_idx=user_id_to_idx,
        user_idx_to_id=user_idx_to_id,
        train_seen_map=train_seen_map,
        val_seen_map=val_seen_map,
        test_seen_map=test_seen_map,
        train_item_popularity=train_item_popularity,
        category_item_popularity=dict(category_item_popularity),
        cooccurrence=dict(cooccurrence),
        hard_negative_history=dict(hard_negative_history),
    )


def _get_training_interactions(prepared: PreparedArtifacts) -> pd.DataFrame:
    config = prepared.config
    interactions = prepared.interactions.copy()
    interactions["item_idx"] = interactions["parent_asin"].map(prepared.item_id_to_idx).astype(np.int32)
    interactions = interactions.sort_values(["user_id", "timestamp", "item_idx"]).reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    for _, group in interactions.groupby("user_id", sort=False):
        if len(group) < max(config.k_core, 3):
            continue
        train_group = group.iloc[:-2].copy()
        if len(train_group) < 2:
            continue
        rows.append(train_group)
    if not rows:
        return pd.DataFrame(columns=list(interactions.columns))
    return pd.concat(rows, ignore_index=True)


def split_diagnostics(prepared: PreparedArtifacts, split_artifacts: SplitArtifacts) -> dict[str, pd.DataFrame]:
    training_interactions = _get_training_interactions(prepared)
    interactions = prepared.interactions.copy()
    interactions["item_idx"] = interactions["parent_asin"].map(prepared.item_id_to_idx).astype(np.int32)
    interactions = interactions.sort_values(["user_id", "timestamp", "item_idx"]).reset_index(drop=True)
    val_repeat = 0
    test_repeat = 0
    val_category: Counter = Counter()
    test_category: Counter = Counter()
    val_history_lengths: list[int] = []
    test_history_lengths: list[int] = []
    covered_items = set(training_interactions["parent_asin"].astype(str))
    train_user_sizes = training_interactions.groupby("user_id").size() if not training_interactions.empty else pd.Series(dtype=np.int64)
    train_item_sizes = training_interactions.groupby("parent_asin").size() if not training_interactions.empty else pd.Series(dtype=np.int64)
    for _, group in interactions.groupby("user_id", sort=False):
        if len(group) < max(prepared.config.k_core, 3):
            continue
        train_group = group.iloc[:-2].copy()
        if len(train_group) < 2:
            continue
        val_row = group.iloc[-2]
        test_row = group.iloc[-1]
        val_history = train_group["item_idx"].tolist()
        test_history = group.iloc[:-1]["item_idx"].tolist()
        val_history_lengths.append(len(val_history[-prepared.config.history_len:]))
        test_history_lengths.append(len(test_history[-prepared.config.history_len:]))
        val_repeat += int(int(val_row["item_idx"]) in set(val_history))
        test_repeat += int(int(test_row["item_idx"]) in set(test_history))
        val_category[str(val_row["source_category"])] += 1
        test_category[str(test_row["source_category"])] += 1
    summary = pd.DataFrame(
        [
            {"metric": "training_interactions", "value": int(len(training_interactions))},
            {"metric": "training_users", "value": int(training_interactions["user_id"].nunique())},
            {"metric": "training_items", "value": int(training_interactions["parent_asin"].nunique())},
            {"metric": "catalog_coverage_after_kcore", "value": float(len(covered_items) / max(len(prepared.item_features), 1))},
            {"metric": "hard_negative_density", "value": float(len(prepared.hard_negatives) / max(len(prepared.interactions), 1))},
            {"metric": "mean_val_history_length", "value": float(np.mean(val_history_lengths)) if val_history_lengths else 0.0},
            {"metric": "mean_test_history_length", "value": float(np.mean(test_history_lengths)) if test_history_lengths else 0.0},
            {"metric": "p90_val_history_length", "value": float(np.percentile(val_history_lengths, 90)) if val_history_lengths else 0.0},
            {"metric": "p90_test_history_length", "value": float(np.percentile(test_history_lengths, 90)) if test_history_lengths else 0.0},
            {"metric": "val_repeat_rate", "value": float(val_repeat / max(len(split_artifacts.val_examples), 1))},
            {"metric": "test_repeat_rate", "value": float(test_repeat / max(len(split_artifacts.test_examples), 1))},
            {"metric": "median_train_user_interactions", "value": float(train_user_sizes.median()) if not train_user_sizes.empty else 0.0},
            {"metric": "p90_train_user_interactions", "value": float(train_user_sizes.quantile(0.9)) if not train_user_sizes.empty else 0.0},
            {"metric": "median_train_item_frequency", "value": float(train_item_sizes.median()) if not train_item_sizes.empty else 0.0},
            {"metric": "p90_train_item_frequency", "value": float(train_item_sizes.quantile(0.9)) if not train_item_sizes.empty else 0.0},
        ]
    )
    category_share = pd.DataFrame(
        [
            {"split": "val", "source_category": category, "share": count / max(len(split_artifacts.val_examples), 1), "count": count}
            for category, count in sorted(val_category.items())
        ]
        + [
            {"split": "test", "source_category": category, "share": count / max(len(split_artifacts.test_examples), 1), "count": count}
            for category, count in sorted(test_category.items())
        ]
    )
    history_summary = pd.DataFrame(
        [
            {"split": "train", "metric": "rows", "value": len(split_artifacts.train_examples)},
            {"split": "val", "metric": "rows", "value": len(split_artifacts.val_examples)},
            {"split": "test", "metric": "rows", "value": len(split_artifacts.test_examples)},
            {"split": "train", "metric": "users", "value": split_artifacts.train_examples["user_id"].nunique()},
            {"split": "val", "metric": "users", "value": split_artifacts.val_examples["user_id"].nunique()},
            {"split": "test", "metric": "users", "value": split_artifacts.test_examples["user_id"].nunique()},
            {"split": "train", "metric": "median_history_length", "value": float(split_artifacts.train_examples["history_length"].median())},
            {"split": "val", "metric": "median_history_length", "value": float(split_artifacts.val_examples["history_length"].median())},
            {"split": "test", "metric": "median_history_length", "value": float(split_artifacts.test_examples["history_length"].median())},
        ]
    )
    return {
        "summary": summary,
        "category_share": category_share,
        "history_summary": history_summary,
    }


def _user_dense_columns(config: PipelineConfig) -> list[str]:
    return [
        "user_interaction_count",
        "user_mean_rating",
        "user_verified_rate",
        "days_since_last",
        "avg_days_between",
        *[f"pref_{category}" for category in config.categories],
    ]


def _item_dense_columns() -> list[str]:
    return [
        "price",
        "average_rating",
        "log_rating_number",
        "log_positive_count",
        "verified_purchase_rate",
        "helpful_vote_mean",
        "helpful_nonzero_rate",
        "days_since_last_interaction",
    ]


def _item_dense_matrix(item_features: pd.DataFrame) -> np.ndarray:
    dense_cols = _item_dense_columns()
    return item_features[dense_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)


def _build_item_lookup(prepared: PreparedArtifacts) -> dict[int, Record]:
    lookup: dict[int, Record] = {}
    dense_matrix = _item_dense_matrix(prepared.item_features)
    for row_number, row in enumerate(prepared.item_features.itertuples(index=False)):
        lookup[int(row.item_idx)] = {
            "item_idx": int(row.item_idx),
            "parent_asin": row.parent_asin,
            "source_category": row.source_category,
            "source_category_idx": int(row.source_category_idx),
            "dense": dense_matrix[row_number],
            "title": row.title,
            "price": float(row.price) if pd.notna(row.price) else 0.0,
        }
    return lookup


def _mean_text_profile(history_item_idxs: list[int], item_text_matrix: np.ndarray) -> np.ndarray:
    if not history_item_idxs:
        return np.zeros((item_text_matrix.shape[1],), dtype=np.float32)
    valid_indices = [idx - 1 for idx in history_item_idxs if idx > 0]
    if not valid_indices:
        return np.zeros((item_text_matrix.shape[1],), dtype=np.float32)
    return item_text_matrix[valid_indices].mean(axis=0).astype(np.float32)


def _pad_history(history_item_idxs: list[int], history_len: int) -> np.ndarray:
    padded = np.zeros((history_len,), dtype=np.int32)
    if history_item_idxs:
        truncated = history_item_idxs[-history_len:]
        padded[-len(truncated):] = np.asarray(truncated, dtype=np.int32)
    return padded


def _feature_arrays_from_examples(
    examples: pd.DataFrame,
    prepared: PreparedArtifacts,
) -> dict[str, np.ndarray]:
    item_lookup = _build_item_lookup(prepared)
    user_dense_cols = _user_dense_columns(prepared.config)
    user_ids = examples["user_idx"].to_numpy(dtype=np.int32)
    histories = np.stack([_pad_history(history, prepared.config.history_len) for history in examples["history_item_idxs"]])
    user_dense = examples[user_dense_cols].to_numpy(dtype=np.float32)
    user_text = np.stack([_mean_text_profile(history, prepared.item_text_matrix) for history in examples["history_item_idxs"]]).astype(np.float32)
    item_ids = examples["target_item_idx"].to_numpy(dtype=np.int32)
    item_categories = np.asarray([item_lookup[item_id]["source_category_idx"] for item_id in item_ids], dtype=np.int32)
    item_dense = np.stack([item_lookup[item_id]["dense"] for item_id in item_ids]).astype(np.float32)
    item_text = np.asarray(prepared.item_text_matrix[item_ids - 1], dtype=np.float32)
    labels = np.ones((len(examples), 1), dtype=np.float32)
    return {
        "user_id": user_ids,
        "history_item_ids": histories,
        "user_dense": user_dense,
        "user_text": user_text,
        "item_id": item_ids,
        "item_category": item_categories,
        "item_dense": item_dense,
        "item_text": item_text,
        "labels": labels,
    }


def _sample_negative_item(
    target_category: str,
    seen_items: set[int],
    category_items: dict[str, np.ndarray],
    global_items: np.ndarray,
    rng: np.random.Generator,
) -> int:
    pool = category_items.get(target_category)
    if pool is None or len(pool) == 0:
        pool = global_items
    for _ in range(20):
        sampled = int(rng.choice(pool))
        if sampled not in seen_items:
            return sampled
    fallback_pool = [item for item in global_items.tolist() if item not in seen_items]
    if not fallback_pool:
        return int(rng.choice(global_items))
    return int(rng.choice(fallback_pool))


def _cooccurrence_hard_negatives(
    history_items: list[int],
    target_item_idx: int,
    seen_items: set[int],
    cooccurrence: dict[int, Counter],
    limit: int,
) -> list[int]:
    score_counter: Counter = Counter()
    for history_item in history_items:
        score_counter.update(cooccurrence.get(int(history_item), Counter()))
    hard_items: list[int] = []
    for item_idx, _ in score_counter.most_common():
        item_idx = int(item_idx)
        if item_idx == target_item_idx or item_idx in seen_items:
            continue
        hard_items.append(item_idx)
        if len(hard_items) >= limit:
            break
    return hard_items


def _build_retriever_training_examples(
    prepared: PreparedArtifacts,
    train_examples: pd.DataFrame,
    split_artifacts: SplitArtifacts | None = None,
    negatives_per_positive: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(prepared.config.seed)
    negatives_per_positive = int(negatives_per_positive or prepared.config.negatives_per_positive)
    user_dense_cols = _user_dense_columns(prepared.config)
    items_by_category: dict[str, np.ndarray] = {}
    for category, group in prepared.item_features.groupby("source_category"):
        items_by_category[str(category)] = group["item_idx"].to_numpy(dtype=np.int32)
    global_items = prepared.item_features["item_idx"].to_numpy(dtype=np.int32)
    rows: list[Record] = []
    for _, row in train_examples.iterrows():
        base_row = {
            "user_idx": int(row["user_idx"]),
            "history_item_idxs": list(row["history_item_idxs"]),
            **{column: float(row[column]) for column in user_dense_cols},
        }
        target_category = str(row["target_source_category"])
        target_item_idx = int(row["target_item_idx"])
        rows.append(
            {
                **base_row,
                "candidate_item_idx": target_item_idx,
                "label": 1.0,
            }
        )
        seen_items = set(row["history_item_idxs"])
        seen_items.add(target_item_idx)
        hard_negative_pool = []
        if split_artifacts is not None:
            hard_negative_pool = _cooccurrence_hard_negatives(
                list(row["history_item_idxs"]),
                target_item_idx,
                seen_items,
                split_artifacts.cooccurrence,
                limit=max(negatives_per_positive * 2, 10),
            )
        sampled_hard = 0
        for _ in range(negatives_per_positive):
            if sampled_hard < len(hard_negative_pool):
                sampled_item_idx = int(hard_negative_pool[sampled_hard])
                sampled_hard += 1
            else:
                sampled_item_idx = _sample_negative_item(
                    target_category,
                    seen_items,
                    items_by_category,
                    global_items,
                    rng,
                )
            rows.append(
                {
                    **base_row,
                    "candidate_item_idx": sampled_item_idx,
                    "label": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df["candidate_item_idx"] = df["candidate_item_idx"].astype(np.int32)
    df["label"] = df["label"].astype(np.float32)
    return df


def _filter_retriever_examples(examples: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    filtered = examples[examples["history_length"] >= int(config.retriever_quality_min_history)].copy()
    if filtered.empty:
        raise RuntimeError(
            "No retriever examples satisfied retriever_quality_min_history. "
            "Lower CONFIG.retriever_quality_min_history or increase the effective corpus density."
        )
    return filtered.reset_index(drop=True)


def _retriever_pair_summary(pairs: pd.DataFrame, split_name: str) -> pd.DataFrame:
    positive_mask = pairs["label"] > 0.5
    negative_mask = ~positive_mask
    return pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": int(len(pairs)),
                "positives": int(positive_mask.sum()),
                "negatives": int(negative_mask.sum()),
                "positive_rate": float(pairs["label"].mean()) if not pairs.empty else 0.0,
                "users": int(pairs["user_idx"].nunique()) if "user_idx" in pairs.columns else 0,
                "items": int(pairs["candidate_item_idx"].nunique()) if "candidate_item_idx" in pairs.columns else 0,
                "mean_history_length": float(pairs["history_item_idxs"].map(len).mean()) if "history_item_idxs" in pairs.columns and not pairs.empty else 0.0,
            }
        ]
    )


def _retriever_batch_arrays(
    examples: pd.DataFrame,
    prepared: PreparedArtifacts,
) -> dict[str, np.ndarray]:
    item_lookup = _build_item_lookup(prepared)
    user_dense_cols = _user_dense_columns(prepared.config)
    histories = np.stack([_pad_history(history, prepared.config.history_len) for history in examples["history_item_idxs"]])
    user_text = np.stack([_mean_text_profile(history, prepared.item_text_matrix) for history in examples["history_item_idxs"]]).astype(np.float32)
    item_ids = examples["candidate_item_idx"].to_numpy(dtype=np.int32)
    return {
        "user_id": examples["user_idx"].to_numpy(dtype=np.int32),
        "history_item_ids": histories,
        "user_dense": examples[user_dense_cols].to_numpy(dtype=np.float32),
        "user_text": user_text,
        "item_id": item_ids,
        "item_category": np.asarray([item_lookup[item_id]["source_category_idx"] for item_id in item_ids], dtype=np.int32),
        "item_dense": np.stack([item_lookup[item_id]["dense"] for item_id in item_ids]).astype(np.float32),
        "item_text": np.asarray(prepared.item_text_matrix[item_ids - 1], dtype=np.float32),
        "labels": examples["label"].to_numpy(dtype=np.float32).reshape(-1, 1),
    }


class UserTower(tf.keras.Model):
    def __init__(self, num_users: int, num_items: int, history_len: int, user_dense_dim: int, text_dim: int, hidden_dims: tuple[int, ...], embedding_dim: int):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users + 1, embedding_dim, mask_zero=True, name="user_embedding")
        self.item_history_embedding = tf.keras.layers.Embedding(num_items + 1, embedding_dim, mask_zero=True, name="history_item_embedding")
        self.history_pool = tf.keras.layers.GlobalAveragePooling1D(name="history_pool")
        self.concat = tf.keras.layers.Concatenate(name="user_concat")
        self.hidden = [tf.keras.layers.Dense(dim, activation="relu", name=f"user_dense_{index}") for index, dim in enumerate(hidden_dims, start=1)]
        self.projection = tf.keras.layers.Dense(embedding_dim, activation=None, name="user_projection")
        self.normalize = tf.keras.layers.UnitNormalization(axis=-1, name="user_normalize")

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        user_vec = self.user_embedding(inputs["user_id"])
        history_vec = self.item_history_embedding(inputs["history_item_ids"])
        history_vec = self.history_pool(history_vec)
        x = self.concat([user_vec, history_vec, inputs["user_dense"], inputs["user_text"]])
        for layer in self.hidden:
            x = layer(x)
        x = self.projection(x)
        return self.normalize(x)


class ItemTower(tf.keras.Model):
    def __init__(self, num_items: int, num_categories: int, item_dense_dim: int, text_dim: int, hidden_dims: tuple[int, ...], embedding_dim: int, include_augmented: bool = False):
        super().__init__()
        self.item_embedding = tf.keras.layers.Embedding(num_items + 1, embedding_dim, mask_zero=True, name="item_embedding")
        self.category_embedding = tf.keras.layers.Embedding(num_categories + 1, min(embedding_dim, 16), mask_zero=True, name="category_embedding")
        self.include_augmented = include_augmented
        self.concat = tf.keras.layers.Concatenate(name="item_concat")
        self.hidden = [tf.keras.layers.Dense(dim, activation="relu", name=f"item_dense_{index}") for index, dim in enumerate(hidden_dims, start=1)]
        self.projection = tf.keras.layers.Dense(embedding_dim, activation=None, name="item_projection")
        self.normalize = tf.keras.layers.UnitNormalization(axis=-1, name="item_normalize")

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        item_vec = self.item_embedding(inputs["item_id"])
        category_vec = self.category_embedding(inputs["item_category"])
        concat_values = [item_vec, category_vec, inputs["item_dense"], inputs["item_text"]]
        if self.include_augmented:
            concat_values.append(inputs["item_aug"])
        x = self.concat(concat_values)
        for layer in self.hidden:
            x = layer(x)
        x = self.projection(x)
        return self.normalize(x)


def tfp_covariance(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    x = x - tf.reduce_mean(x, axis=0, keepdims=True)
    sample_count = tf.maximum(tf.shape(x)[0] - 1, 1)
    return tf.matmul(x, x, transpose_a=True) / tf.cast(sample_count, tf.float32)


class TwoTowerRetrieverModel(tf.keras.Model):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        config: PipelineConfig,
        user_dense_dim: int,
        item_dense_dim: int,
        text_dim: int,
    ):
        super().__init__()
        self.config = config
        self.user_tower = UserTower(num_users, num_items, config.history_len, user_dense_dim, text_dim, config.retriever_hidden_dims, config.retriever_embedding_dim)
        self.item_tower = ItemTower(num_items, num_categories, item_dense_dim, text_dim, config.retriever_hidden_dims, config.retriever_embedding_dim)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.bce_tracker = tf.keras.metrics.Mean(name="bce_loss")
        self.in_batch_tracker = tf.keras.metrics.Mean(name="in_batch_loss")
        self.auc = tf.keras.metrics.AUC(name="auc")
        self.logit_scale_log = tf.Variable(
            np.log(float(config.retriever_logit_scale)),
            trainable=True,
            dtype=tf.float32,
            name="retriever_logit_scale_log",
        )

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.loss_tracker, self.bce_tracker, self.in_batch_tracker, self.auc]

    def encode_user(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        return self.user_tower(inputs, training=training)

    def encode_item(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        return self.item_tower(inputs, training=training)

    def _logit_scale(self) -> tf.Tensor:
        return tf.clip_by_value(tf.exp(self.logit_scale_log), 1.0, 100.0)

    def _scaled_similarity(self, user_embedding: tf.Tensor, item_embedding: tf.Tensor) -> tf.Tensor:
        return self._logit_scale() * tf.reduce_sum(user_embedding * item_embedding, axis=-1, keepdims=True)

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> dict[str, tf.Tensor]:
        user_embedding = self.encode_user(inputs, training=training)
        item_embedding = self.encode_item(inputs, training=training)
        logits = self._scaled_similarity(user_embedding, item_embedding)
        return {
            "logits": logits,
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
            "logit_scale": self._logit_scale(),
        }

    def _base_losses(self, labels: tf.Tensor, outputs: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs["logits"]))
        positive_mask = tf.squeeze(labels > 0.5, axis=-1)
        positive_users = tf.boolean_mask(outputs["user_embedding"], positive_mask)
        positive_items = tf.boolean_mask(outputs["item_embedding"], positive_mask)

        def _compute_in_batch_loss() -> tf.Tensor:
            positive_users = tf.boolean_mask(outputs["user_embedding"], positive_mask)
            positive_items = tf.boolean_mask(outputs["item_embedding"], positive_mask)
            similarity = self._logit_scale() * tf.matmul(positive_users, positive_items, transpose_b=True)
            targets = tf.range(tf.shape(positive_users)[0])
            return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, similarity, from_logits=True))

        in_batch_loss = tf.cond(
            tf.shape(positive_users)[0] > 1,
            _compute_in_batch_loss,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )
        total_loss = bce_loss + (self.config.in_batch_weight * in_batch_loss)
        return total_loss, in_batch_loss

    def train_step(self, data: tuple[dict[str, tf.Tensor], tf.Tensor]) -> dict[str, tf.Tensor]:
        features, labels = data
        with tf.GradientTape() as tape:
            outputs = self(features, training=True)
            total_loss, in_batch_loss = self._base_losses(labels, outputs)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        bce_value = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs["logits"]))
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_value)
        self.in_batch_tracker.update_state(in_batch_loss)
        self.auc.update_state(labels, tf.nn.sigmoid(outputs["logits"]))
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: tuple[dict[str, tf.Tensor], tf.Tensor]) -> dict[str, tf.Tensor]:
        features, labels = data
        outputs = self(features, training=False)
        total_loss, in_batch_loss = self._base_losses(labels, outputs)
        bce_value = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs["logits"]))
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_value)
        self.in_batch_tracker.update_state(in_batch_loss)
        self.auc.update_state(labels, tf.nn.sigmoid(outputs["logits"]))
        return {metric.name: metric.result() for metric in self.metrics}


class DATLiteRetrieverModel(TwoTowerRetrieverModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        config: PipelineConfig,
        user_dense_dim: int,
        item_dense_dim: int,
        text_dim: int,
    ):
        super().__init__(num_users, num_items, num_categories, config, user_dense_dim, item_dense_dim, text_dim)
        self.user_aug_embedding = tf.keras.layers.Embedding(num_users + 1, config.retriever_embedding_dim, mask_zero=True, name="user_aug_embedding")
        self.item_aug_embedding = tf.keras.layers.Embedding(num_items + 1, config.retriever_embedding_dim, mask_zero=True, name="item_aug_embedding")
        self.item_tower = ItemTower(num_items, num_categories, item_dense_dim, text_dim, config.retriever_hidden_dims, config.retriever_embedding_dim, include_augmented=True)
        self.user_concat = tf.keras.layers.Concatenate(name="dat_user_concat")
        self.user_hidden = [tf.keras.layers.Dense(dim, activation="relu", name=f"dat_user_dense_{index}") for index, dim in enumerate(config.retriever_hidden_dims, start=1)]
        self.user_projection = tf.keras.layers.Dense(config.retriever_embedding_dim, activation=None, name="dat_user_projection")
        self.user_normalize = tf.keras.layers.UnitNormalization(axis=-1, name="dat_user_normalize")
        self.mimic_tracker = tf.keras.metrics.Mean(name="mimic_loss")
        self.category_alignment_tracker = tf.keras.metrics.Mean(name="category_alignment_loss")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.loss_tracker, self.bce_tracker, self.in_batch_tracker, self.mimic_tracker, self.category_alignment_tracker, self.auc]

    def encode_user(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        user_vec = self.user_tower.user_embedding(inputs["user_id"])
        history_vec = self.user_tower.item_history_embedding(inputs["history_item_ids"])
        history_vec = self.user_tower.history_pool(history_vec)
        user_aug = self.user_aug_embedding(inputs["user_id"])
        x = self.user_concat([user_vec, history_vec, inputs["user_dense"], inputs["user_text"], user_aug])
        for layer in self.user_hidden:
            x = layer(x)
        x = self.user_projection(x)
        return self.user_normalize(x)

    def encode_item(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        augmented_inputs = dict(inputs)
        item_aug = self.item_aug_embedding(inputs["item_id"])
        augmented_inputs["item_aug"] = item_aug
        return self.item_tower(augmented_inputs, training=training)

    def _category_alignment_loss(self, item_embeddings: tf.Tensor, categories: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        positive_mask = tf.squeeze(labels > 0.5, axis=-1)
        pos_embeddings = tf.boolean_mask(item_embeddings, positive_mask)
        pos_categories = tf.boolean_mask(categories, positive_mask)
        unique_categories, _, counts = tf.unique_with_counts(pos_categories)
        if tf.size(unique_categories) <= 1:
            return tf.constant(0.0, dtype=tf.float32)
        major_index = tf.argmax(counts)
        major_category = tf.gather(unique_categories, major_index)
        major_embeddings = tf.boolean_mask(pos_embeddings, tf.equal(pos_categories, major_category))
        if tf.shape(major_embeddings)[0] < 2:
            return tf.constant(0.0, dtype=tf.float32)
        major_cov = tfp_covariance(major_embeddings)
        losses = []
        for category in tf.unstack(unique_categories):
            if tf.equal(category, major_category):
                continue
            category_embeddings = tf.boolean_mask(pos_embeddings, tf.equal(pos_categories, category))
            if tf.shape(category_embeddings)[0] < 2:
                continue
            category_cov = tfp_covariance(category_embeddings)
            losses.append(tf.reduce_mean(tf.square(major_cov - category_cov)))
        if not losses:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.add_n(losses) / tf.cast(len(losses), tf.float32)

    def _base_losses(
        self,
        labels: tf.Tensor,
        outputs: dict[str, tf.Tensor],
        features: dict[str, tf.Tensor] | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        total_loss, in_batch_loss = super()._base_losses(labels, outputs)
        positive_mask = tf.squeeze(labels > 0.5, axis=-1)
        mimic_loss = tf.constant(0.0, dtype=tf.float32)
        if tf.reduce_any(positive_mask):
            pos_user_ids = tf.boolean_mask(features["user_id"], positive_mask)
            pos_item_ids = tf.boolean_mask(features["item_id"], positive_mask)
            pos_user_aug = self.user_aug_embedding(pos_user_ids)
            pos_item_aug = self.item_aug_embedding(pos_item_ids)
            pos_user_emb = tf.stop_gradient(tf.boolean_mask(outputs["user_embedding"], positive_mask))
            pos_item_emb = tf.stop_gradient(tf.boolean_mask(outputs["item_embedding"], positive_mask))
            mimic_loss = tf.reduce_mean(tf.square(pos_user_aug - pos_item_emb)) + tf.reduce_mean(tf.square(pos_item_aug - pos_user_emb))
        category_alignment_loss = self._category_alignment_loss(outputs["item_embedding"], features["item_category"], labels)
        total_loss = total_loss + (self.config.dat_mimic_weight * mimic_loss) + (self.config.dat_category_alignment_weight * category_alignment_loss)
        return total_loss, in_batch_loss, mimic_loss, category_alignment_loss

    def train_step(self, data: tuple[dict[str, tf.Tensor], tf.Tensor]) -> dict[str, tf.Tensor]:
        features, labels = data
        with tf.GradientTape() as tape:
            outputs = self(features, training=True)
            total_loss, in_batch_loss, mimic_loss, category_alignment_loss = self._base_losses(labels, outputs, features)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        bce_value = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs["logits"]))
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_value)
        self.in_batch_tracker.update_state(in_batch_loss)
        self.mimic_tracker.update_state(mimic_loss)
        self.category_alignment_tracker.update_state(category_alignment_loss)
        self.auc.update_state(labels, tf.nn.sigmoid(outputs["logits"]))
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: tuple[dict[str, tf.Tensor], tf.Tensor]) -> dict[str, tf.Tensor]:
        features, labels = data
        outputs = self(features, training=False)
        total_loss, in_batch_loss, mimic_loss, category_alignment_loss = self._base_losses(labels, outputs, features)
        bce_value = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs["logits"]))
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_value)
        self.in_batch_tracker.update_state(in_batch_loss)
        self.mimic_tracker.update_state(mimic_loss)
        self.category_alignment_tracker.update_state(category_alignment_loss)
        self.auc.update_state(labels, tf.nn.sigmoid(outputs["logits"]))
        return {metric.name: metric.result() for metric in self.metrics}


def _make_retriever_tf_dataset(
    features: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
    prefetch_batches: int,
) -> tf.data.Dataset:
    labels = features["labels"]
    model_features = {key: value for key, value in features.items() if key != "labels"}
    dataset = tf.data.Dataset.from_tensor_slices((model_features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(labels), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)
    prefetch_value = max(int(prefetch_batches), 1)
    return dataset.batch(batch_size).prefetch(prefetch_value)


def _build_encoder_models(
    model: tf.keras.Model,
    history_len: int,
    user_dense_dim: int,
    item_dense_dim: int,
    text_dim: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    user_inputs = {
        "user_id": tf.keras.Input(shape=(), dtype=tf.int32, name="user_id"),
        "history_item_ids": tf.keras.Input(shape=(history_len,), dtype=tf.int32, name="history_item_ids"),
        "user_dense": tf.keras.Input(shape=(user_dense_dim,), dtype=tf.float32, name="user_dense"),
        "user_text": tf.keras.Input(shape=(text_dim,), dtype=tf.float32, name="user_text"),
    }
    item_inputs = {
        "item_id": tf.keras.Input(shape=(), dtype=tf.int32, name="item_id"),
        "item_category": tf.keras.Input(shape=(), dtype=tf.int32, name="item_category"),
        "item_dense": tf.keras.Input(shape=(item_dense_dim,), dtype=tf.float32, name="item_dense"),
        "item_text": tf.keras.Input(shape=(text_dim,), dtype=tf.float32, name="item_text"),
    }
    user_encoder = tf.keras.Model(inputs=user_inputs, outputs=model.encode_user(user_inputs, training=False), name="user_encoder")
    item_encoder = tf.keras.Model(inputs=item_inputs, outputs=model.encode_item(item_inputs, training=False), name="item_encoder")
    return user_encoder, item_encoder


def build_ann_index(
    config: PipelineConfig,
    item_embeddings: np.ndarray,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = AnnoyIndex(item_embeddings.shape[1], "angular")
    for row_index, embedding in enumerate(item_embeddings):
        index.add_item(row_index, embedding.tolist())
    index.build(config.ann_trees)
    index.save(str(output_path))
    return output_path


def _top_items_from_counter(counter: Counter, seen_items: set[int], k: int) -> list[int]:
    items = []
    for item_idx, _ in counter.most_common():
        if item_idx in seen_items:
            continue
        items.append(int(item_idx))
        if len(items) >= k:
            break
    return items


def popularity_by_category_candidates(
    split_artifacts: SplitArtifacts,
    examples: pd.DataFrame,
    top_k: int = 100,
) -> pd.DataFrame:
    rows: list[Record] = []
    global_counter = split_artifacts.train_item_popularity
    for _, example in examples.iterrows():
        seen_items = set(example["history_item_idxs"])
        if not split_artifacts.config.category_backfill_enabled:
            selected_items = _top_items_from_counter(global_counter, seen_items, top_k)
        else:
            category_preferences = {
                category: max(float(example[f"pref_{category}"]), 0.0)
                for category in split_artifacts.config.categories
            }
            positive_preferences = [(category, value) for category, value in category_preferences.items() if value > 0.0]
            if not positive_preferences:
                positive_preferences = [(category, 1.0) for category in split_artifacts.config.categories]
            preference_total = sum(value for _, value in positive_preferences) or float(len(positive_preferences))
            global_reserve = min(max(int(round(top_k * 0.20)), 1), max(top_k - len(positive_preferences), 0)) if top_k > 1 else 0
            category_budget = max(top_k - global_reserve, 0)
            raw_allocations = [
                (category, (value / preference_total) * category_budget)
                for category, value in sorted(positive_preferences, key=lambda item: (-item[1], item[0]))
            ]
            allocations = {category: int(math.floor(raw_value)) for category, raw_value in raw_allocations}
            for category, _ in raw_allocations:
                allocations[category] = max(1, allocations[category])
            remainder = category_budget - sum(allocations.values())
            if remainder > 0 and raw_allocations:
                ranked_remainders = sorted(
                    raw_allocations,
                    key=lambda item: (-(item[1] - math.floor(item[1])), item[0]),
                )
                for index in range(remainder):
                    allocations[ranked_remainders[index % len(ranked_remainders)][0]] += 1
            elif remainder < 0:
                for category, _ in sorted(raw_allocations, key=lambda item: (item[1] - math.floor(item[1]), item[0])):
                    if remainder == 0:
                        break
                    removable = min(allocations[category] - 1, abs(remainder))
                    if removable > 0:
                        allocations[category] -= removable
                        remainder += removable

            selected_items: list[int] = []
            selected = set(seen_items)
            for category, _ in raw_allocations:
                quota = allocations.get(category, 0)
                if quota <= 0:
                    continue
                candidates = _top_items_from_counter(split_artifacts.category_item_popularity.get(category, Counter()), selected, quota)
                selected_items.extend(candidates)
                selected.update(candidates)
            if len(selected_items) < category_budget:
                fallback = _top_items_from_counter(global_counter, selected, category_budget - len(selected_items))
                selected_items.extend(fallback)
                selected.update(fallback)
            if len(selected_items) < top_k:
                fallback = _top_items_from_counter(global_counter, selected, top_k - len(selected_items))
                selected_items.extend(fallback)
        for rank, item_idx in enumerate(selected_items[:top_k], start=1):
            rows.append(
                {
                    "example_id": int(example["example_id"]),
                    "split": str(example["split"]),
                    "user_id": str(example["user_id"]),
                    "item_idx": int(item_idx),
                    "retrieval_score": float(top_k - rank + 1),
                    "rank": rank,
                    "label": int(item_idx == int(example["target_item_idx"])),
                    "target_item_idx": int(example["target_item_idx"]),
                }
            )
    return pd.DataFrame(rows)


def item_item_cooccurrence_candidates(
    split_artifacts: SplitArtifacts,
    examples: pd.DataFrame,
    top_k: int = 100,
) -> pd.DataFrame:
    rows: list[Record] = []
    for _, example in examples.iterrows():
        seen_items = set(example["history_item_idxs"])
        score_counter: Counter = Counter()
        history_items = list(example["history_item_idxs"])
        history_count = max(len(history_items), 1)
        for position, history_item in enumerate(history_items):
            neighbors = split_artifacts.cooccurrence.get(int(history_item), Counter())
            if not split_artifacts.config.recency_cooccurrence_enabled:
                score_counter.update(neighbors)
                continue
            recency_weight = float((position + 1) / history_count)
            for candidate_idx, count in neighbors.items():
                score_counter[int(candidate_idx)] += float(count) * recency_weight
        candidates = [item_idx for item_idx, _ in score_counter.most_common() if item_idx not in seen_items][:top_k]
        if len(candidates) < top_k:
            fallback = _top_items_from_counter(split_artifacts.train_item_popularity, seen_items.union(candidates), top_k - len(candidates))
            candidates.extend(fallback)
        for rank, item_idx in enumerate(candidates[:top_k], start=1):
            rows.append(
                {
                    "example_id": int(example["example_id"]),
                    "split": str(example["split"]),
                    "user_id": str(example["user_id"]),
                    "item_idx": int(item_idx),
                    "retrieval_score": float(score_counter.get(item_idx, 0.0)),
                    "rank": rank,
                    "label": int(item_idx == int(example["target_item_idx"])),
                    "target_item_idx": int(example["target_item_idx"]),
                }
            )
    return pd.DataFrame(rows)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _ann_candidates_from_vectors(
    item_embeddings: np.ndarray,
    ann_index: AnnoyIndex,
    query_vectors: np.ndarray,
    examples: pd.DataFrame,
    split_artifacts: SplitArtifacts,
    top_k: int,
    source_name: str,
    inject_target_if_missing: bool = True,
) -> pd.DataFrame:
    rows: list[Record] = []
    seen_maps = {
        "train": split_artifacts.train_seen_map,
        "val": split_artifacts.val_seen_map,
        "test": split_artifacts.test_seen_map,
    }
    for row_index, (_, example) in enumerate(examples.iterrows()):
        query_vector = np.asarray(query_vectors[row_index], dtype=np.float32)
        split_name = str(example["split"])
        if split_name == "train":
            seen_items = set(example["history_item_idxs"])
        elif split_name in seen_maps:
            seen_items = seen_maps[split_name].get(str(example["user_id"]), set(example["history_item_idxs"]))
        else:
            seen_items = set(example["history_item_idxs"])
        requested = max(top_k * 3, 300)
        candidate_rows: list[tuple[int, float]] = []
        while len(candidate_rows) < top_k and requested <= len(item_embeddings) * 2:
            indices, distances = ann_index.get_nns_by_vector(query_vector.tolist(), requested, include_distances=True)
            candidate_rows.clear()
            for annoy_row, distance in zip(indices, distances):
                item_idx = int(annoy_row + 1)
                if item_idx in seen_items:
                    continue
                cosine_like_score = 1.0 - (distance ** 2) / 2.0
                candidate_rows.append((item_idx, cosine_like_score))
                if len(candidate_rows) >= top_k:
                    break
            requested *= 2
        target_item_idx = int(example["target_item_idx"])
        retrieved_items = {item_idx for item_idx, _ in candidate_rows}
        if inject_target_if_missing and split_name != "inference" and target_item_idx not in retrieved_items:
            target_score = float(np.dot(query_vector, item_embeddings[target_item_idx - 1]))
            candidate_rows.append((target_item_idx, target_score))
        candidate_rows = sorted(candidate_rows, key=lambda value: value[1], reverse=True)[:top_k]
        for rank, (item_idx, score) in enumerate(candidate_rows, start=1):
            rows.append(
                {
                    "example_id": int(example["example_id"]),
                    "split": split_name,
                    "user_id": str(example["user_id"]),
                    "user_idx": int(example["user_idx"]),
                    "history_item_idxs": example["history_item_idxs"],
                    "target_item_idx": target_item_idx,
                    "target_parent_asin": example["target_parent_asin"],
                    "target_timestamp": example["target_timestamp"],
                    "target_source_category": example.get("target_source_category", ""),
                    "history_length": int(example.get("history_length", len(example["history_item_idxs"]))),
                    "item_idx": int(item_idx),
                    "retrieval_score": float(score),
                    "rank": rank,
                    "label": int(item_idx == target_item_idx),
                    "user_interaction_count": float(example["user_interaction_count"]),
                    "user_mean_rating": float(example["user_mean_rating"]),
                    "user_verified_rate": float(example["user_verified_rate"]),
                    "days_since_last": float(example["days_since_last"]),
                    "avg_days_between": float(example["avg_days_between"]),
                    "source": source_name,
                    **{f"pref_{category}": float(example[f"pref_{category}"]) for category in split_artifacts.config.categories},
                }
            )
    return pd.DataFrame(rows)


def train_content_retriever(prepared: PreparedArtifacts, split_artifacts: SplitArtifacts) -> RetrieverArtifacts:
    item_embeddings = _normalize_rows(np.asarray(prepared.item_text_matrix, dtype=np.float32))
    ann_index_path = build_ann_index(prepared.config, item_embeddings, prepared.config.model_dir / "content_item_index.ann")
    retriever = RetrieverArtifacts(
        config=prepared.config,
        variant="content_based",
        model={"retriever": "content_based"},
        item_encoder=None,
        user_encoder=None,
        item_embeddings=item_embeddings,
        ann_index_path=ann_index_path,
        ann_index=_load_ann_index(item_embeddings.shape[1], ann_index_path),
        metrics=pd.DataFrame(),
        history={},
        retriever_kind="vector",
        metadata={},
    )
    retriever.metrics = evaluate_retriever(prepared, split_artifacts, retriever)
    retriever.metrics.to_csv(prepared.config.eval_dir / "content_based_retriever_metrics.csv", index=False)
    return retriever


def train_latent_cf_retriever(prepared: PreparedArtifacts, split_artifacts: SplitArtifacts) -> RetrieverArtifacts:
    train_interactions = _get_training_interactions(prepared)
    if train_interactions.empty:
        raise RuntimeError("No training interactions were available for the latent collaborative retriever.")
    train_interactions = train_interactions.copy()
    train_interactions["user_id"] = train_interactions["user_id"].astype(str)
    known_users = set(split_artifacts.user_id_to_idx)
    original_interactions = len(train_interactions)
    train_interactions = train_interactions[train_interactions["user_id"].isin(known_users)].copy()
    if train_interactions.empty:
        raise RuntimeError(
            "No latent collaborative-filtering interactions matched the split user index. "
            "Increase train_positive_cap/split_eval_example_cap or disable very aggressive split sampling."
        )
    dropped_interactions = original_interactions - len(train_interactions)
    if dropped_interactions:
        LOGGER.info(
            "Latent CF training restricted to split users: kept_interactions=%s dropped_interactions=%s users=%s",
            f"{len(train_interactions):,}",
            f"{dropped_interactions:,}",
            f"{len(known_users):,}",
        )
    train_interactions["user_idx"] = train_interactions["user_id"].map(split_artifacts.user_id_to_idx).astype(np.int32)
    train_interactions["item_idx"] = train_interactions["parent_asin"].map(prepared.item_id_to_idx).astype(np.int32)
    grouped = (
        train_interactions.groupby(["user_idx", "item_idx"], as_index=False)
        .size()
        .rename(columns={"size": "strength"})
    )
    matrix = sparse.coo_matrix(
        (
            grouped["strength"].to_numpy(dtype=np.float32),
            (grouped["user_idx"].to_numpy(dtype=np.int32) - 1, grouped["item_idx"].to_numpy(dtype=np.int32) - 1),
        ),
        shape=(len(split_artifacts.user_id_to_idx), len(prepared.item_features)),
    ).tocsr()
    component_cap = min(prepared.config.latent_cf_components, max(2, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=component_cap, random_state=prepared.config.seed)
    user_vectors = svd.fit_transform(matrix).astype(np.float32)
    item_vectors = svd.components_.T.astype(np.float32)
    user_vectors = _normalize_rows(user_vectors)
    item_vectors = _normalize_rows(item_vectors)
    ann_index_path = build_ann_index(prepared.config, item_vectors, prepared.config.model_dir / "latent_cf_item_index.ann")
    retriever = RetrieverArtifacts(
        config=prepared.config,
        variant="latent_cf",
        model={"svd": svd, "user_vectors": user_vectors},
        item_encoder=None,
        user_encoder=None,
        item_embeddings=item_vectors,
        ann_index_path=ann_index_path,
        ann_index=_load_ann_index(item_vectors.shape[1], ann_index_path),
        metrics=pd.DataFrame(),
        history={"explained_variance": [float(svd.explained_variance_ratio_.sum())]},
        retriever_kind="vector",
        metadata={"user_vectors": user_vectors},
    )
    retriever.metrics = evaluate_retriever(prepared, split_artifacts, retriever)
    retriever.metrics.to_csv(prepared.config.eval_dir / "latent_cf_retriever_metrics.csv", index=False)
    return retriever


def _prepare_user_encoder_inputs(
    examples: pd.DataFrame,
    prepared: PreparedArtifacts,
) -> dict[str, np.ndarray]:
    user_dense_cols = _user_dense_columns(prepared.config)
    return {
        "user_id": examples["user_idx"].to_numpy(dtype=np.int32),
        "history_item_ids": np.stack([_pad_history(history, prepared.config.history_len) for history in examples["history_item_idxs"]]),
        "user_dense": examples[user_dense_cols].to_numpy(dtype=np.float32),
        "user_text": np.stack([_mean_text_profile(history, prepared.item_text_matrix) for history in examples["history_item_idxs"]]).astype(np.float32),
    }


def _load_ann_index(embedding_dim: int, ann_index_path: Path) -> AnnoyIndex:
    index = AnnoyIndex(embedding_dim, "angular")
    index.load(str(ann_index_path))
    return index


def _vector_retriever_queries(
    prepared: PreparedArtifacts,
    retriever: RetrieverArtifacts,
    examples: pd.DataFrame,
) -> np.ndarray:
    if retriever.variant == "content_based":
        return _normalize_rows(
            np.stack([_mean_text_profile(history, prepared.item_text_matrix) for history in examples["history_item_idxs"]]).astype(np.float32)
        )
    if retriever.variant == "latent_cf":
        user_vectors = np.asarray(retriever.metadata.get("user_vectors"), dtype=np.float32)
        query_vectors: list[np.ndarray] = []
        for row in examples.itertuples(index=False):
            user_idx = int(getattr(row, "user_idx"))
            if 0 < user_idx <= len(user_vectors):
                vector = user_vectors[user_idx - 1]
            else:
                history = list(getattr(row, "history_item_idxs"))
                if history:
                    vector = np.mean(retriever.item_embeddings[np.asarray(history, dtype=np.int32) - 1], axis=0)
                else:
                    vector = np.zeros((retriever.item_embeddings.shape[1],), dtype=np.float32)
            query_vectors.append(vector.astype(np.float32, copy=False))
        return _normalize_rows(np.stack(query_vectors).astype(np.float32))
    if retriever.variant in {"two_tower", "dat_lite"}:
        query_vectors = []
        for row in examples.itertuples(index=False):
            history = list(getattr(row, "history_item_idxs"))
            if history:
                vector = np.mean(retriever.item_embeddings[np.asarray(history, dtype=np.int32) - 1], axis=0)
            else:
                vector = np.zeros((retriever.item_embeddings.shape[1],), dtype=np.float32)
            query_vectors.append(vector.astype(np.float32, copy=False))
        return _normalize_rows(np.stack(query_vectors).astype(np.float32))
    raise ValueError(f"Unsupported vector retriever variant: {retriever.variant}")


def generate_candidates(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retriever: RetrieverArtifacts,
    examples: pd.DataFrame,
    top_k: int | None = None,
    inject_target_if_missing: bool = True,
) -> pd.DataFrame:
    top_k = top_k or prepared.config.retrieval_top_k
    if retriever.retriever_kind == "vector":
        if retriever.ann_index is None:
            if retriever.ann_index_path is None:
                raise ValueError(f"Retriever {retriever.variant} is missing its ANN index.")
            retriever.ann_index = _load_ann_index(retriever.item_embeddings.shape[1], retriever.ann_index_path)
        query_vectors = _vector_retriever_queries(prepared, retriever, examples)
        return _ann_candidates_from_vectors(
            retriever.item_embeddings,
            retriever.ann_index,
            query_vectors,
            examples,
            split_artifacts,
            top_k,
            retriever.variant,
            inject_target_if_missing=inject_target_if_missing,
        )
    user_inputs = _prepare_user_encoder_inputs(examples, prepared)
    user_embeddings = retriever.user_encoder.predict(user_inputs, batch_size=prepared.config.retriever_batch_size, verbose=0)
    index = retriever.ann_index
    if index is None:
        index = _load_ann_index(retriever.item_embeddings.shape[1], retriever.ann_index_path)
        retriever.ann_index = index
    rows: list[Record] = []
    seen_maps = {
        "train": split_artifacts.train_seen_map,
        "val": split_artifacts.val_seen_map,
        "test": split_artifacts.test_seen_map,
    }
    for row_index, (_, example) in enumerate(examples.iterrows()):
        user_embedding = user_embeddings[row_index]
        split_name = str(example["split"])
        if split_name == "train":
            seen_items = set(example["history_item_idxs"])
        elif split_name in seen_maps:
            seen_items = seen_maps[split_name].get(str(example["user_id"]), set(example["history_item_idxs"]))
        else:
            seen_items = set(example["history_item_idxs"])
        requested = max(top_k * 3, 300)
        candidate_rows: list[tuple[int, float]] = []
        while len(candidate_rows) < top_k and requested <= len(prepared.item_features) * 2:
            indices, distances = index.get_nns_by_vector(user_embedding.tolist(), requested, include_distances=True)
            candidate_rows.clear()
            for annoy_row, distance in zip(indices, distances):
                item_idx = int(annoy_row + 1)
                if item_idx in seen_items:
                    continue
                cosine_like_score = 1.0 - (distance ** 2) / 2.0
                candidate_rows.append((item_idx, cosine_like_score))
                if len(candidate_rows) >= top_k:
                    break
            requested *= 2
        retrieved_items = {item_idx for item_idx, _ in candidate_rows}
        target_item_idx = int(example["target_item_idx"])
        if inject_target_if_missing and target_item_idx not in retrieved_items:
            target_score = float(np.dot(user_embedding, retriever.item_embeddings[target_item_idx - 1]))
            candidate_rows.append((target_item_idx, target_score))
        candidate_rows = sorted(candidate_rows, key=lambda value: value[1], reverse=True)[:top_k]
        for rank, (item_idx, score) in enumerate(candidate_rows, start=1):
            rows.append(
                {
                    "example_id": int(example["example_id"]),
                    "split": split_name,
                    "user_id": str(example["user_id"]),
                    "user_idx": int(example["user_idx"]),
                    "history_item_idxs": example["history_item_idxs"],
                    "target_item_idx": target_item_idx,
                    "target_parent_asin": example["target_parent_asin"],
                    "target_timestamp": example["target_timestamp"],
                    "target_source_category": example.get("target_source_category", ""),
                    "history_length": int(example.get("history_length", len(example["history_item_idxs"]))),
                    "item_idx": int(item_idx),
                    "retrieval_score": float(score),
                    "rank": rank,
                    "label": int(item_idx == target_item_idx),
                    "user_interaction_count": float(example["user_interaction_count"]),
                    "user_mean_rating": float(example["user_mean_rating"]),
                    "user_verified_rate": float(example["user_verified_rate"]),
                    "days_since_last": float(example["days_since_last"]),
                    "avg_days_between": float(example["avg_days_between"]),
                    "source": retriever.variant,
                    **{f"pref_{category}": float(example[f"pref_{category}"]) for category in prepared.config.categories},
                }
            )
    return pd.DataFrame(rows)


def _metrics_from_ranked_candidates(candidates: pd.DataFrame, ks: tuple[int, ...] = (10, 50, 100)) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["K", "recall", "mrr", "ndcg"])
    metrics_rows: list[Record] = []
    grouped = candidates.sort_values(["example_id", "rank"]).groupby("example_id", sort=False)
    for k in ks:
        hits = 0.0
        reciprocal_ranks = 0.0
        ndcg = 0.0
        total = 0
        for _, group in grouped:
            topk = group.head(k)
            positive = topk[topk["label"] == 1]
            total += 1
            if not positive.empty:
                hits += 1.0
                rank_value = int(positive.iloc[0]["rank"])
                reciprocal_ranks += 1.0 / rank_value
                ndcg += 1.0 / math.log2(rank_value + 1)
        metrics_rows.append(
            {
                "K": k,
                "recall": hits / max(total, 1),
                "mrr": reciprocal_ranks / max(total, 1),
                "ndcg": ndcg / max(total, 1),
            }
        )
    return pd.DataFrame(metrics_rows)


def _normalize_metrics_frame(
    metrics: pd.DataFrame | None,
    *,
    extra_columns: tuple[str, ...] = ("split", "variant"),
) -> pd.DataFrame:
    expected_columns = ["K", "recall", "mrr", "ndcg", *extra_columns]
    if metrics is None:
        return pd.DataFrame(columns=expected_columns)
    normalized = metrics.copy()
    for column in expected_columns:
        if column not in normalized.columns:
            normalized[column] = np.nan
    ordered = [column for column in expected_columns if column in normalized.columns]
    remainder = [column for column in normalized.columns if column not in ordered]
    return normalized[ordered + remainder]


def evaluate_candidate_table(candidates: pd.DataFrame, ks: tuple[int, ...] = (10, 50, 100)) -> pd.DataFrame:
    return _metrics_from_ranked_candidates(candidates, ks=ks)


def evaluate_retriever(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retriever: RetrieverArtifacts,
) -> pd.DataFrame:
    eval_frames: list[pd.DataFrame] = []
    for split_name, examples in [("val", split_artifacts.val_examples), ("test", split_artifacts.test_examples)]:
        eval_examples = examples
        if prepared.config.eval_user_cap is not None and len(eval_examples) > prepared.config.eval_user_cap:
            eval_examples = eval_examples.sample(n=prepared.config.eval_user_cap, random_state=prepared.config.seed).sort_values("example_id")
        candidates = generate_candidates(
            prepared,
            split_artifacts,
            retriever,
            eval_examples,
            top_k=prepared.config.retrieval_top_k,
            inject_target_if_missing=False,
        )
        metrics = _metrics_from_ranked_candidates(candidates, ks=(10, 50, 100))
        metrics["split"] = split_name
        metrics["variant"] = retriever.variant
        eval_frames.append(metrics)
        candidates_with_context = _add_candidate_item_context(prepared, candidates)
        diagnostics = candidate_recall_diagnostics(
            candidates_with_context,
            split=split_name,
            variant=retriever.variant,
            stage="retriever",
        )
        diagnostics.to_csv(prepared.config.eval_dir / f"{retriever.variant}_{split_name}_candidate_diagnostics_metrics.csv", index=False)
        candidates_with_context.to_parquet(prepared.config.eval_dir / f"{retriever.variant}_{split_name}_retrieval_candidates.parquet", index=False)
    if not eval_frames:
        return _normalize_metrics_frame(None)
    return _normalize_metrics_frame(pd.concat(eval_frames, ignore_index=True))


def train_retriever(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    variant: str = "two_tower",
) -> RetrieverArtifacts:
    config = prepared.config
    if variant not in {"two_tower", "dat_lite"}:
        raise ValueError("variant must be 'two_tower' or 'dat_lite'")
    train_examples = _filter_retriever_examples(split_artifacts.train_examples, config)
    if config.retriever_train_example_cap is not None and len(train_examples) > config.retriever_train_example_cap:
        train_examples = _sample_training_examples(train_examples, config.retriever_train_example_cap, config.seed)
        print(f"Retriever training capped to {len(train_examples):,} base examples for notebook-safe memory usage.")
    val_examples = _filter_retriever_examples(split_artifacts.val_examples, config)
    if config.eval_user_cap is not None and len(val_examples) > config.eval_user_cap:
        val_examples = val_examples.sample(n=config.eval_user_cap, random_state=config.seed).sort_values("example_id")
    train_pairs = _build_retriever_training_examples(prepared, train_examples, split_artifacts=split_artifacts)
    val_pairs = _build_retriever_training_examples(
        prepared,
        val_examples,
        split_artifacts=split_artifacts,
        negatives_per_positive=config.retriever_validation_negatives_per_positive,
    )
    pair_summaries = pd.concat(
        [
            _retriever_pair_summary(train_pairs, "train"),
            _retriever_pair_summary(val_pairs, "val"),
        ],
        ignore_index=True,
    )
    train_arrays = _retriever_batch_arrays(train_pairs, prepared)
    val_arrays = _retriever_batch_arrays(val_pairs, prepared)
    del train_pairs
    del val_pairs
    gc.collect()
    train_dataset = _make_retriever_tf_dataset(
        train_arrays,
        batch_size=config.retriever_batch_size,
        shuffle=True,
        seed=config.seed,
        shuffle_buffer=config.retriever_shuffle_buffer,
        prefetch_batches=config.tf_prefetch_batches,
    )
    val_dataset = _make_retriever_tf_dataset(
        val_arrays,
        batch_size=config.retriever_batch_size,
        shuffle=False,
        seed=config.seed,
        shuffle_buffer=config.retriever_shuffle_buffer,
        prefetch_batches=config.tf_prefetch_batches,
    )
    num_users = len(split_artifacts.user_id_to_idx)
    num_items = len(prepared.item_features)
    num_categories = len(prepared.category_to_idx)
    user_dense_dim = train_arrays["user_dense"].shape[1]
    item_dense_dim = train_arrays["item_dense"].shape[1]
    text_dim = train_arrays["item_text"].shape[1]
    gc.collect()
    if variant == "two_tower":
        model = TwoTowerRetrieverModel(num_users, num_items, num_categories, config, user_dense_dim, item_dense_dim, text_dim)
        run_eagerly = False
    else:
        model = DATLiteRetrieverModel(num_users, num_items, num_categories, config, user_dense_dim, item_dense_dim, text_dim)
        run_eagerly = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), run_eagerly=run_eagerly)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.retriever_epochs,
        verbose=config.training_verbose,
    ).history
    sanity_sample_size = min(len(val_arrays["labels"]), 4096)
    sanity_features = {
        key: value[:sanity_sample_size]
        for key, value in val_arrays.items()
        if key != "labels"
    }
    sanity_labels = val_arrays["labels"][:sanity_sample_size].reshape(-1)
    sanity_logits = model(sanity_features, training=False)["logits"].numpy().reshape(-1)
    sanity_scores = 1.0 / (1.0 + np.exp(-sanity_logits))
    positive_scores = sanity_scores[sanity_labels > 0.5]
    negative_scores = sanity_scores[sanity_labels <= 0.5]
    sanity_checks = pd.DataFrame(
        [
            {
                "variant": variant,
                "sample_rows": int(sanity_sample_size),
                "positive_rows": int((sanity_labels > 0.5).sum()),
                "negative_rows": int((sanity_labels <= 0.5).sum()),
                "positive_mean_score": float(np.mean(positive_scores)) if len(positive_scores) else np.nan,
                "negative_mean_score": float(np.mean(negative_scores)) if len(negative_scores) else np.nan,
                "score_gap": float(np.mean(positive_scores) - np.mean(negative_scores)) if len(positive_scores) and len(negative_scores) else np.nan,
                "logit_scale": float(model._logit_scale().numpy()) if hasattr(model, "_logit_scale") else np.nan,
            }
        ]
    )
    history["final_logit_scale"] = [float(model._logit_scale().numpy())] if hasattr(model, "_logit_scale") else []
    user_encoder, item_encoder = _build_encoder_models(
        model,
        history_len=config.history_len,
        user_dense_dim=user_dense_dim,
        item_dense_dim=item_dense_dim,
        text_dim=text_dim,
    )
    del train_dataset
    del val_dataset
    del train_arrays
    del val_arrays
    gc.collect()
    item_dense_matrix = _item_dense_matrix(prepared.item_features)
    item_inputs = {
        "item_id": prepared.item_features["item_idx"].to_numpy(dtype=np.int32),
        "item_category": prepared.item_features["source_category_idx"].to_numpy(dtype=np.int32),
        "item_dense": item_dense_matrix,
        "item_text": np.asarray(prepared.item_text_matrix, dtype=np.float32),
    }
    item_embeddings = item_encoder.predict(item_inputs, batch_size=config.retriever_batch_size, verbose=0).astype(np.float32)
    del item_inputs
    del item_dense_matrix
    gc.collect()
    ann_index_path = build_ann_index(config, item_embeddings, config.model_dir / f"{variant}_item_index.ann")
    model.save_weights(config.model_dir / f"{variant}_retriever.weights.h5")
    np.save(config.model_dir / f"{variant}_item_embeddings.npy", item_embeddings)
    retriever_metadata = {
        "variant": variant,
        "retriever_kind": "neural",
        "embedding_dim": int(item_embeddings.shape[1]),
        "history_len": int(config.history_len),
        "final_logit_scale": float(model._logit_scale().numpy()) if hasattr(model, "_logit_scale") else None,
        "train_examples_used": int(len(train_examples)),
        "val_examples_used": int(len(val_examples)),
    }
    with open(config.model_dir / f"{variant}_retriever_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(retriever_metadata, handle, indent=2)
    if config.persist_encoder_models:
        user_encoder.save(config.model_dir / f"{variant}_user_encoder.keras", overwrite=True)
        item_encoder.save(config.model_dir / f"{variant}_item_encoder.keras", overwrite=True)
    provisional = RetrieverArtifacts(
        config=config,
        variant=variant,
        model=model,
        item_encoder=item_encoder,
        user_encoder=user_encoder,
        item_embeddings=item_embeddings,
        ann_index_path=ann_index_path,
        ann_index=_load_ann_index(item_embeddings.shape[1], ann_index_path),
        metrics=pd.DataFrame(),
        history=history,
        metadata={
            "pair_summaries": pair_summaries,
            "sanity_checks": sanity_checks,
            **retriever_metadata,
        },
    )
    metrics = evaluate_retriever(prepared, split_artifacts, provisional)
    metrics.to_csv(config.eval_dir / f"{variant}_retriever_metrics.csv", index=False)
    pair_summaries.to_csv(config.eval_dir / f"{variant}_pair_summaries.csv", index=False)
    sanity_checks.to_csv(config.eval_dir / f"{variant}_sanity_checks.csv", index=False)
    provisional.metrics = metrics
    return provisional


def train_retrievers(prepared: PreparedArtifacts, split_artifacts: SplitArtifacts) -> dict[str, RetrieverArtifacts]:
    retrievers: dict[str, RetrieverArtifacts] = {}
    LOGGER.info("Training content-based retriever")
    retrievers["content_based"] = train_content_retriever(prepared, split_artifacts)
    LOGGER.info("Content-based retriever complete")
    LOGGER.info("Training latent collaborative-filtering retriever")
    retrievers["latent_cf"] = train_latent_cf_retriever(prepared, split_artifacts)
    LOGGER.info("Latent collaborative-filtering retriever complete")
    if prepared.config.enable_neural_retriever:
        LOGGER.info("Training neural two-tower retriever")
        neural_retriever = train_retriever(prepared, split_artifacts, variant="two_tower")
        if _retriever_recovers_positives(neural_retriever.metrics):
            retrievers["two_tower"] = neural_retriever
            LOGGER.info("Neural two-tower retriever complete and enabled for candidate union")
        else:
            LOGGER.warning(
                "Neural two-tower retriever completed but recovered no positives in evaluation. "
                "It will be logged for diagnostics but excluded from candidate union."
            )
    else:
        LOGGER.info("Skipping two_tower retriever because enable_neural_retriever is false")
    return retrievers


def _retriever_recovers_positives(metrics: pd.DataFrame) -> bool:
    if metrics.empty or "recall" not in metrics.columns:
        return False
    return bool(pd.to_numeric(metrics["recall"], errors="coerce").fillna(0.0).max() > 0.0)


def _candidate_source_budgets(config: PipelineConfig) -> dict[str, int]:
    return {
        "cooccurrence": int(config.cooccurrence_candidate_k),
        "latent_cf": int(config.latent_cf_candidate_k),
        "content_based": int(config.content_candidate_k),
        "two_tower": int(config.neural_candidate_k),
        "popularity": int(config.popularity_backfill_k),
    }


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "example_id",
            "split",
            "user_id",
            "user_idx",
            "history_item_idxs",
            "target_item_idx",
            "target_parent_asin",
            "target_timestamp",
            "target_source_category",
            "history_length",
            "item_idx",
            "retrieval_score",
            "rank",
            "label",
            "user_interaction_count",
            "user_mean_rating",
            "user_verified_rate",
            "days_since_last",
            "avg_days_between",
            "source",
        ],
    )


def _normalize_candidate_frame(frame: pd.DataFrame | None, source_name: str | None = None) -> pd.DataFrame:
    template = _empty_candidate_frame()
    expected_columns = list(template.columns)
    if frame is None or frame.empty:
        normalized = template.copy()
    else:
        normalized = frame.copy()
        for column in expected_columns:
            if column not in normalized.columns:
                normalized[column] = np.nan
        ordered = [column for column in expected_columns if column in normalized.columns]
        remainder = [column for column in normalized.columns if column not in ordered]
        normalized = normalized[ordered + remainder]
    if source_name is not None:
        normalized["source"] = source_name
    return normalized


def _source_candidate_frames(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: dict[str, RetrieverArtifacts],
    examples: pd.DataFrame,
    inject_target_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    if examples.empty:
        return {
            "cooccurrence": _normalize_candidate_frame(None, "cooccurrence"),
            "popularity": _normalize_candidate_frame(None, "popularity"),
            "latent_cf": _normalize_candidate_frame(None, "latent_cf"),
            "content_based": _normalize_candidate_frame(None, "content_based"),
            "two_tower": _normalize_candidate_frame(None, "two_tower"),
        }
    budgets = _candidate_source_budgets(prepared.config)
    frames: dict[str, pd.DataFrame] = {
        "cooccurrence": _normalize_candidate_frame(
            item_item_cooccurrence_candidates(split_artifacts, examples, top_k=budgets["cooccurrence"]),
            "cooccurrence",
        ),
        "popularity": _normalize_candidate_frame(
            popularity_by_category_candidates(split_artifacts, examples, top_k=budgets["popularity"]),
            "popularity",
        ),
    }
    if "latent_cf" in retrievers:
        frames["latent_cf"] = _normalize_candidate_frame(
            generate_candidates(
                prepared,
                split_artifacts,
                retrievers["latent_cf"],
                examples,
                top_k=budgets["latent_cf"],
                inject_target_if_missing=inject_target_if_missing,
            ),
            "latent_cf",
        )
    else:
        frames["latent_cf"] = _normalize_candidate_frame(None, "latent_cf")
    if "content_based" in retrievers:
        frames["content_based"] = _normalize_candidate_frame(
            generate_candidates(
                prepared,
                split_artifacts,
                retrievers["content_based"],
                examples,
                top_k=budgets["content_based"],
                inject_target_if_missing=inject_target_if_missing,
            ),
            "content_based",
        )
    else:
        frames["content_based"] = _normalize_candidate_frame(None, "content_based")
    if "two_tower" in retrievers:
        frames["two_tower"] = _normalize_candidate_frame(
            generate_candidates(
                prepared,
                split_artifacts,
                retrievers["two_tower"],
                examples,
                top_k=budgets["two_tower"],
                inject_target_if_missing=inject_target_if_missing,
            ),
            "two_tower",
        )
    else:
        frames["two_tower"] = _normalize_candidate_frame(None, "two_tower")
    return frames


def _candidate_metadata_from_example(example: pd.Series) -> Record:
    preference_features = {
        str(column): float(example[column])
        for column in example.index
        if str(column).startswith("pref_")
    }
    return {
        "example_id": int(example["example_id"]),
        "split": str(example["split"]),
        "user_id": str(example["user_id"]),
        "user_idx": int(example["user_idx"]),
        "history_item_idxs": list(example["history_item_idxs"]),
        "target_item_idx": int(example["target_item_idx"]),
        "target_parent_asin": example["target_parent_asin"],
        "target_timestamp": example["target_timestamp"],
        "target_source_category": example["target_source_category"],
        "user_interaction_count": float(example["user_interaction_count"]),
        "history_length": int(example["history_length"]) if "history_length" in example.index else len(example["history_item_idxs"]),
        "user_mean_rating": float(example["user_mean_rating"]),
        "user_verified_rate": float(example["user_verified_rate"]),
        "days_since_last": float(example["days_since_last"]),
        "avg_days_between": float(example["avg_days_between"]),
        **preference_features,
    }


def generate_candidate_union(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: dict[str, RetrieverArtifacts],
    examples: pd.DataFrame,
    top_k: int | None = None,
    inject_target_if_missing: bool = True,
    include_candidate_sources: bool = False,
) -> pd.DataFrame:
    top_k = int(top_k or prepared.config.candidate_union_top_k)
    if examples.empty:
        return _empty_candidate_frame()
    source_frames = _source_candidate_frames(
        prepared,
        split_artifacts,
        retrievers,
        examples,
        inject_target_if_missing=inject_target_if_missing,
    )
    budgets = _candidate_source_budgets(prepared.config)
    grouped_frames = {name: frame.groupby("example_id", sort=False) for name, frame in source_frames.items()}
    source_weights = {
        "cooccurrence": 1.00,
        "latent_cf": 1.00,
        "content_based": 0.85,
        "two_tower": 0.90,
        "popularity": 0.25,
    }
    rows: list[Record] = []
    for _, example in examples.iterrows():
        example_id = int(example["example_id"])
        item_map: dict[int, Record] = {}
        example_meta = _candidate_metadata_from_example(example)
        source_order = ["cooccurrence", "latent_cf", "content_based", "two_tower"]
        for source_name in source_order:
            frame = grouped_frames[source_name].get_group(example_id) if example_id in grouped_frames[source_name].groups else None
            if frame is None:
                continue
            for candidate in frame.itertuples(index=False):
                item_idx = int(candidate.item_idx)
                row = item_map.setdefault(
                    item_idx,
                    {
                        **example_meta,
                        "item_idx": item_idx,
                        "label": int(item_idx == example_meta["target_item_idx"]),
                        "source_count": 0,
                        "union_score": 0.0,
                    },
                )
                row[f"from_{source_name}"] = 1
                row[f"score_{source_name}"] = float(candidate.retrieval_score)
                row[f"rank_{source_name}"] = int(candidate.rank)
                row["source_count"] += 1
                row["union_score"] += source_weights[source_name] / max(int(candidate.rank), 1)
        if len(item_map) < top_k:
            popularity_frame = grouped_frames["popularity"].get_group(example_id) if example_id in grouped_frames["popularity"].groups else None
            if popularity_frame is not None:
                for candidate in popularity_frame.itertuples(index=False):
                    item_idx = int(candidate.item_idx)
                    row = item_map.setdefault(
                        item_idx,
                        {
                            **example_meta,
                            "item_idx": item_idx,
                            "label": int(item_idx == example_meta["target_item_idx"]),
                            "source_count": 0,
                            "union_score": 0.0,
                        },
                    )
                    row.setdefault("from_popularity", 1)
                    row.setdefault("score_popularity", float(candidate.retrieval_score))
                    row.setdefault("rank_popularity", int(candidate.rank))
                    if row["source_count"] == 0:
                        row["source_count"] = 1
                    row["union_score"] += source_weights["popularity"] / max(int(candidate.rank), 1)
                    if len(item_map) >= top_k:
                        break
        target_item_idx = int(example_meta["target_item_idx"])
        if inject_target_if_missing and str(example_meta["split"]) != "inference" and target_item_idx not in item_map:
            item_map[target_item_idx] = {
                **example_meta,
                "item_idx": target_item_idx,
                "label": 1,
                "source_count": 0,
                "union_score": 0.0,
                "from_injected_positive": 1,
            }
        example_rows = pd.DataFrame(item_map.values())
        if example_rows.empty and inject_target_if_missing and str(example_meta["split"]) != "inference":
            example_rows = pd.DataFrame(
                [
                    {
                        **example_meta,
                        "item_idx": target_item_idx,
                        "label": 1,
                        "source_count": 0,
                        "union_score": 0.0,
                        "from_injected_positive": 1,
                    }
                ]
            )
        if "union_score" not in example_rows.columns:
            example_rows["union_score"] = pd.Series(dtype=float)
        if "source_count" not in example_rows.columns:
            example_rows["source_count"] = pd.Series(dtype=np.int32)
        for source_name, budget in budgets.items():
            if source_name == "popularity":
                example_rows[f"from_{source_name}"] = example_rows.get(f"from_{source_name}", 0)
                example_rows[f"score_{source_name}"] = example_rows.get(f"score_{source_name}", 0.0)
                example_rows[f"rank_{source_name}"] = example_rows.get(f"rank_{source_name}", budget + 1)
                continue
            example_rows[f"from_{source_name}"] = example_rows.get(f"from_{source_name}", 0)
            example_rows[f"score_{source_name}"] = example_rows.get(f"score_{source_name}", 0.0)
            example_rows[f"rank_{source_name}"] = example_rows.get(f"rank_{source_name}", budget + 1)
        example_rows["union_score"] = example_rows["union_score"].astype(float)
        example_rows = example_rows.sort_values(["union_score", "source_count"], ascending=[False, False]).head(top_k).copy()
        example_rows["retrieval_score"] = example_rows["union_score"]
        example_rows["rank"] = np.arange(1, len(example_rows) + 1, dtype=np.int32)
        rows.append(example_rows)
    if not rows:
        return _empty_candidate_frame()
    union = pd.concat(rows, ignore_index=True)
    if union.empty:
        union = _empty_candidate_frame()
        if include_candidate_sources:
            union["candidate_sources"] = pd.Series(dtype=str)
        return union
    source_names = ["cooccurrence", "latent_cf", "content_based", "two_tower", "popularity"]
    for source_name in source_names:
        flag_col = f"from_{source_name}"
        if flag_col not in union.columns:
            union[flag_col] = 0
        union[flag_col] = union[flag_col].fillna(0).astype(np.int8)
    if include_candidate_sources:
        if union.empty:
            union["candidate_sources"] = pd.Series(dtype=str)
        else:
            union["candidate_sources"] = union.apply(
                lambda row: " + ".join(
                    [
                        source_name
                        for source_name in source_names
                        if int(row[f"from_{source_name}"]) == 1
                    ]
                ),
                axis=1,
            )
    return union


def candidate_source_diagnostics(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if candidates.empty:
        return {
            "metrics": pd.DataFrame(),
            "per_category": pd.DataFrame(),
            "source_summary": pd.DataFrame(),
            "worst_slice": pd.DataFrame(),
        }
    metrics = _metrics_from_ranked_candidates(candidates, ks=(10, 50, 100))
    if "item_idx" in candidates.columns:
        metrics["coverage"] = [
            float(candidates[candidates["rank"] <= k]["item_idx"].nunique() / max(candidates["item_idx"].nunique(), 1))
            for k in metrics["K"].tolist()
        ]
    per_category_hits = (
        candidates.groupby(["target_source_category", "example_id"], as_index=False)["label"]
        .max()
        .rename(columns={"label": "hit"})
    )
    per_category = (
        per_category_hits.groupby("target_source_category", as_index=False)
        .agg(
            hit_rate=("hit", "mean"),
            hits=("hit", "sum"),
            count=("hit", "size"),
        )
        .sort_values(["hit_rate", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    source_rows: list[Record] = []
    for source_name in ["cooccurrence", "latent_cf", "content_based", "two_tower", "popularity"]:
        flag_col = f"from_{source_name}"
        if flag_col not in candidates.columns:
            continue
        source_positive = candidates[(candidates["label"] == 1) & (candidates[flag_col] == 1)]
        source_rows.append(
            {
                "source": source_name,
                "positive_recoveries": int(len(source_positive)),
                "examples_with_positive": int(source_positive["example_id"].nunique()),
                "median_positive_rank": float(source_positive["rank"].median()) if not source_positive.empty else np.nan,
            }
        )
    worst_slice = per_category.sort_values(["hit_rate", "count"], ascending=[True, False]).head(5)
    return {
        "metrics": metrics,
        "per_category": per_category,
        "source_summary": pd.DataFrame(source_rows),
        "worst_slice": worst_slice,
    }


def candidate_recall_diagnostics(
    candidates: pd.DataFrame,
    *,
    split: str,
    variant: str,
    stage: str,
) -> pd.DataFrame:
    columns = [
        "split",
        "variant",
        "stage",
        "scope",
        "name",
        "examples",
        "hit_rate",
        "positive_recoveries",
        "examples_with_positive",
        "median_positive_rank",
    ]
    if candidates.empty or "example_id" not in candidates.columns or "label" not in candidates.columns:
        return pd.DataFrame(columns=columns)

    rows: list[Record] = []
    example_hits = candidates.groupby("example_id")["label"].max()
    positive_rows = candidates[candidates["label"] == 1]
    rows.append(
        {
            "split": split,
            "variant": variant,
            "stage": stage,
            "scope": "overall",
            "name": "all",
            "examples": int(len(example_hits)),
            "hit_rate": float(example_hits.mean()) if not example_hits.empty else 0.0,
            "positive_recoveries": int(len(positive_rows)),
            "examples_with_positive": int(positive_rows["example_id"].nunique()) if not positive_rows.empty else 0,
            "median_positive_rank": float(positive_rows["rank"].median()) if not positive_rows.empty and "rank" in positive_rows.columns else np.nan,
        }
    )
    if "target_source_category" in candidates.columns:
        category_hits = (
            candidates.groupby(["target_source_category", "example_id"], as_index=False)["label"]
            .max()
        )
        for category, group in category_hits.groupby("target_source_category", sort=True):
            positives = positive_rows[positive_rows["target_source_category"] == category] if "target_source_category" in positive_rows.columns else pd.DataFrame()
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "stage": stage,
                    "scope": "target_category",
                    "name": str(category),
                    "examples": int(len(group)),
                    "hit_rate": float(group["label"].mean()) if not group.empty else 0.0,
                    "positive_recoveries": int(len(positives)),
                    "examples_with_positive": int(positives["example_id"].nunique()) if not positives.empty else 0,
                    "median_positive_rank": float(positives["rank"].median()) if not positives.empty and "rank" in positives.columns else np.nan,
                }
            )
    if "history_length" in candidates.columns:
        enriched = candidates.copy()
        enriched["history_length_bucket"] = enriched["history_length"].map(_history_length_bucket)
        history_hits = enriched.groupby(["history_length_bucket", "example_id"], as_index=False)["label"].max()
        positive_with_bucket = enriched[enriched["label"] == 1]
        for bucket, group in history_hits.groupby("history_length_bucket", sort=True):
            positives = positive_with_bucket[positive_with_bucket["history_length_bucket"] == bucket]
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "stage": stage,
                    "scope": "history_length_bucket",
                    "name": str(bucket),
                    "examples": int(len(group)),
                    "hit_rate": float(group["label"].mean()) if not group.empty else 0.0,
                    "positive_recoveries": int(len(positives)),
                    "examples_with_positive": int(positives["example_id"].nunique()) if not positives.empty else 0,
                    "median_positive_rank": float(positives["rank"].median()) if not positives.empty and "rank" in positives.columns else np.nan,
                }
            )
    if "target_price_bucket" in candidates.columns:
        price_hits = candidates.groupby(["target_price_bucket", "example_id"], as_index=False)["label"].max()
        positive_with_price = positive_rows[positive_rows["target_price_bucket"].notna()] if "target_price_bucket" in positive_rows.columns else pd.DataFrame()
        for bucket, group in price_hits.groupby("target_price_bucket", sort=True):
            positives = positive_with_price[positive_with_price["target_price_bucket"] == bucket]
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "stage": stage,
                    "scope": "target_price_bucket",
                    "name": str(bucket),
                    "examples": int(len(group)),
                    "hit_rate": float(group["label"].mean()) if not group.empty else 0.0,
                    "positive_recoveries": int(len(positives)),
                    "examples_with_positive": int(positives["example_id"].nunique()) if not positives.empty else 0,
                    "median_positive_rank": float(positives["rank"].median()) if not positives.empty and "rank" in positives.columns else np.nan,
                }
            )
    source_names = ["cooccurrence", "latent_cf", "content_based", "two_tower", "popularity"]
    if any(f"from_{source_name}" in candidates.columns for source_name in source_names):
        for source_name in source_names:
            flag_col = f"from_{source_name}"
            if flag_col not in candidates.columns:
                continue
            source_candidates = candidates[candidates[flag_col].fillna(0).astype(int) == 1]
            source_positive = source_candidates[source_candidates["label"] == 1]
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "stage": stage,
                    "scope": "candidate_source",
                    "name": source_name,
                    "examples": int(source_candidates["example_id"].nunique()) if not source_candidates.empty else 0,
                    "hit_rate": float(source_positive["example_id"].nunique() / max(candidates["example_id"].nunique(), 1)),
                    "positive_recoveries": int(len(source_positive)),
                    "examples_with_positive": int(source_positive["example_id"].nunique()) if not source_positive.empty else 0,
                    "median_positive_rank": float(source_positive["rank"].median()) if not source_positive.empty and "rank" in source_positive.columns else np.nan,
                }
            )
    elif "source" in candidates.columns:
        for source_name, source_candidates in candidates.groupby("source", sort=True):
            source_positive = source_candidates[source_candidates["label"] == 1]
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "stage": stage,
                    "scope": "candidate_source",
                    "name": str(source_name),
                    "examples": int(source_candidates["example_id"].nunique()) if not source_candidates.empty else 0,
                    "hit_rate": float(source_positive["example_id"].nunique() / max(candidates["example_id"].nunique(), 1)),
                    "positive_recoveries": int(len(source_positive)),
                    "examples_with_positive": int(source_positive["example_id"].nunique()) if not source_positive.empty else 0,
                    "median_positive_rank": float(source_positive["rank"].median()) if not source_positive.empty and "rank" in source_positive.columns else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _history_length_bucket(value: object) -> str:
    length = int(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
    if length <= 2:
        return "01-02"
    if length <= 5:
        return "03-05"
    if length <= 10:
        return "06-10"
    if length <= 25:
        return "11-25"
    if length <= 50:
        return "26-50"
    return "51+"


def _price_bucket_labels(edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for index in range(max(len(edges) - 1, 0)):
        if index == 0:
            labels.append("price_low")
        elif index == len(edges) - 2:
            labels.append("price_very_high")
        elif index == 1:
            labels.append("price_medium")
        else:
            labels.append("price_high")
    return labels


def _item_price_buckets(item_prices: pd.Series) -> tuple[np.ndarray, list[str]]:
    prices = pd.to_numeric(item_prices, errors="coerce").dropna().astype(float)
    if prices.empty:
        return np.asarray([-np.inf, np.inf], dtype=float), ["price_unknown"]
    if prices.nunique() == 1:
        value = float(prices.iloc[0])
        return np.asarray([-np.inf, value, np.inf], dtype=float), ["price_at_or_below_reference", "price_above_reference"]
    edges = np.quantile(prices.to_numpy(dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.asarray([float(prices.min()) - 0.5, float(prices.max()) + 0.5], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges, _price_bucket_labels(edges)


def _assign_price_bucket(values: pd.Series, edges: np.ndarray, labels: list[str]) -> pd.Series:
    if len(labels) != len(edges) - 1:
        labels = [f"price_bucket_{index + 1}" for index in range(len(edges) - 1)]
    buckets = pd.cut(pd.to_numeric(values, errors="coerce"), bins=edges, labels=labels, include_lowest=True)
    return buckets.astype("string").fillna("price_unknown")


def _add_candidate_item_context(prepared: PreparedArtifacts, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    enriched = candidates.copy()
    item_context = prepared.item_features[["item_idx", "source_category", "price"]].copy()
    item_context["item_idx"] = item_context["item_idx"].astype(int)
    edges, labels = _item_price_buckets(item_context["price"])
    candidate_context = item_context.rename(
        columns={
            "source_category": "item_source_category",
            "price": "item_price",
        }
    )
    candidate_context["item_price_bucket"] = _assign_price_bucket(candidate_context["item_price"], edges, labels)
    target_context = item_context.rename(
        columns={
            "item_idx": "target_item_idx",
            "source_category": "resolved_target_source_category",
            "price": "target_price",
        }
    )
    target_context["target_price_bucket"] = _assign_price_bucket(target_context["target_price"], edges, labels)
    enriched = enriched.merge(candidate_context, on="item_idx", how="left")
    enriched = enriched.merge(target_context, on="target_item_idx", how="left")
    if "target_source_category" not in enriched.columns:
        enriched["target_source_category"] = enriched["resolved_target_source_category"]
    else:
        enriched["target_source_category"] = enriched["target_source_category"].fillna(enriched["resolved_target_source_category"])
    return enriched


def candidate_distribution_by_category_price(
    candidates: pd.DataFrame,
    *,
    split: str,
    variant: str,
    stage: str,
    top_k: int = 10,
) -> pd.DataFrame:
    columns = [
        "split",
        "variant",
        "stage",
        "top_k",
        "source_category",
        "price_bucket",
        "rows",
        "proportion",
        "mean_rank",
        "positive_rows",
    ]
    if candidates.empty or "rank" not in candidates.columns:
        return pd.DataFrame(columns=columns)
    frame = candidates[candidates["rank"].astype(int) <= int(top_k)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    category_col = "item_source_category" if "item_source_category" in frame.columns else "target_source_category"
    price_col = "item_price_bucket" if "item_price_bucket" in frame.columns else "target_price_bucket"
    if category_col not in frame.columns or price_col not in frame.columns:
        return pd.DataFrame(columns=columns)
    frame[category_col] = frame[category_col].astype("string").fillna("__missing__")
    frame[price_col] = frame[price_col].astype("string").fillna("price_unknown")
    total = max(len(frame), 1)
    rows: list[Record] = []
    for (category, price_bucket), group in frame.groupby([category_col, price_col], sort=True):
        rows.append(
            {
                "split": split,
                "variant": variant,
                "stage": stage,
                "top_k": int(top_k),
                "source_category": str(category),
                "price_bucket": str(price_bucket),
                "rows": int(len(group)),
                "proportion": float(len(group) / total),
                "mean_rank": float(pd.to_numeric(group["rank"], errors="coerce").mean()),
                "positive_rows": int(pd.to_numeric(group.get("label", 0), errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _hard_negative_flag_for_candidate(
    user_id: str,
    item_idx: int,
    target_timestamp: pd.Timestamp,
    hard_negative_history: dict[str, list[tuple[int, int]]],
) -> int:
    history = hard_negative_history.get(user_id, [])
    cutoff = int(target_timestamp.timestamp() * 1000)
    for ts, neg_item in history:
        if ts >= cutoff:
            break
        if neg_item == item_idx:
            return 1
    return 0


def _build_ranker_payload(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retriever: RetrieverArtifacts,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    if candidates.empty:
        empty_metadata = pd.DataFrame(
            columns=[
                "example_id",
                "split",
                "user_id",
                "user_idx",
                "item_idx",
                "item_category_idx",
                "retrieval_score",
                "rank",
                "label",
                "target_item_idx",
            ]
        )
        embed_dim = prepared.config.retriever_embedding_dim
        empty_features = {
            "user_idx": np.zeros((0,), dtype=np.int32),
            "item_idx": np.zeros((0,), dtype=np.int32),
            "item_category_idx": np.zeros((0,), dtype=np.int32),
            "dense_features": np.zeros((0, len(_ranker_dense_feature_columns(prepared.config))), dtype=np.float32),
            "user_embedding": np.zeros((0, embed_dim), dtype=np.float32),
            "item_embedding": np.zeros((0, embed_dim), dtype=np.float32),
        }
        return empty_metadata, empty_features, np.zeros((0, 1), dtype=np.float32)

    unique_examples = candidates.drop_duplicates("example_id").sort_values("example_id")
    user_embedding_inputs = _prepare_user_encoder_inputs(unique_examples, prepared)
    if retriever.user_encoder is not None:
        user_embeddings = retriever.user_encoder.predict(
            user_embedding_inputs,
            batch_size=prepared.config.retriever_batch_size,
            verbose=0,
        ).astype(np.float32)
    elif retriever.retriever_kind == "vector":
        user_embeddings = _vector_retriever_queries(prepared, retriever, unique_examples).astype(np.float32)
    else:
        user_embeddings = np.stack(
            [
                _mean_text_profile(history, prepared.item_text_matrix)
                for history in unique_examples["history_item_idxs"]
            ]
        ).astype(np.float32)
    user_text_profiles = np.stack([_mean_text_profile(history, prepared.item_text_matrix) for history in unique_examples["history_item_idxs"]]).astype(np.float32)
    item_prices_full = prepared.item_features["price"].fillna(0.0).to_numpy(dtype=np.float32)
    item_ratings_full = prepared.item_features["average_rating"].fillna(0.0).to_numpy(dtype=np.float32)
    history_price_means = np.asarray(
        [
            float(np.mean(item_prices_full[np.asarray(history, dtype=np.int32) - 1])) if history else 0.0
            for history in unique_examples["history_item_idxs"]
        ],
        dtype=np.float32,
    )
    history_rating_means = np.asarray(
        [
            float(np.mean(item_ratings_full[np.asarray(history, dtype=np.int32) - 1])) if history else 0.0
            for history in unique_examples["history_item_idxs"]
        ],
        dtype=np.float32,
    )
    example_to_row = {int(example_id): index for index, example_id in enumerate(unique_examples["example_id"].tolist())}
    candidate_example_rows = np.fromiter(
        (example_to_row[int(example_id)] for example_id in candidates["example_id"]),
        dtype=np.int32,
        count=len(candidates),
    )
    item_indices = candidates["item_idx"].to_numpy(dtype=np.int32)
    item_rows = item_indices - 1
    item_dense_matrix = _item_dense_matrix(prepared.item_features)
    item_dense_rows = item_dense_matrix[item_rows]
    item_category_idx = prepared.item_features["source_category_idx"].to_numpy(dtype=np.int32)[item_rows]
    if retriever.item_embeddings is not None and len(retriever.item_embeddings) >= len(prepared.item_features):
        item_embeddings = retriever.item_embeddings[item_rows].astype(np.float32, copy=False)
    else:
        item_embeddings = np.zeros((len(candidates), prepared.config.retriever_embedding_dim), dtype=np.float32)
    user_embedding_rows = user_embeddings[candidate_example_rows].astype(np.float32, copy=False)
    user_text_rows = user_text_profiles[candidate_example_rows].astype(np.float32, copy=False)
    item_text_rows = np.asarray(prepared.item_text_matrix[item_rows], dtype=np.float32)
    item_category_names = prepared.item_features["source_category"].to_numpy(dtype=object)[item_rows]
    category_alignment = np.asarray(
        [
            float(candidates.iloc[row_index].get(f"pref_{item_category_names[row_index]}", 0.0))
            for row_index in range(len(candidates))
        ],
        dtype=np.float32,
    ).reshape(-1, 1)
    history_price_rows = history_price_means[candidate_example_rows].reshape(-1, 1)
    history_rating_rows = history_rating_means[candidate_example_rows].reshape(-1, 1)
    item_price_rows = prepared.item_features["price"].fillna(0.0).to_numpy(dtype=np.float32)[item_rows].reshape(-1, 1)
    item_rating_rows = prepared.item_features["average_rating"].fillna(0.0).to_numpy(dtype=np.float32)[item_rows].reshape(-1, 1)

    if user_embedding_rows.ndim != 2:
        user_embedding_rows = np.asarray(user_embedding_rows, dtype=np.float32).reshape(len(candidates), -1)
    if item_embeddings.ndim != 2:
        item_embeddings = np.asarray(item_embeddings, dtype=np.float32).reshape(len(candidates), -1)
    shared_embedding_dim = min(user_embedding_rows.shape[1], item_embeddings.shape[1]) if len(candidates) else 0
    if shared_embedding_dim > 0:
        cosine_similarity = np.sum(
            user_embedding_rows[:, :shared_embedding_dim] * item_embeddings[:, :shared_embedding_dim],
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
    else:
        cosine_similarity = np.zeros((len(candidates), 1), dtype=np.float32)

    def _align_embedding_dim(matrix: np.ndarray, target_dim: int) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            matrix = matrix.reshape(len(candidates), -1)
        current_dim = matrix.shape[1]
        if current_dim == target_dim:
            return matrix.astype(np.float32, copy=False)
        if current_dim > target_dim:
            return matrix[:, :target_dim].astype(np.float32, copy=False)
        return np.pad(matrix, ((0, 0), (0, target_dim - current_dim)), mode="constant").astype(np.float32, copy=False)

    aligned_user_embeddings = _align_embedding_dim(user_embedding_rows, prepared.config.retriever_embedding_dim)
    aligned_item_embeddings = _align_embedding_dim(item_embeddings, prepared.config.retriever_embedding_dim)

    def _candidate_column(name: str, default_value: float) -> np.ndarray:
        if name in candidates.columns:
            return candidates[name].to_numpy(dtype=np.float32).reshape(-1, 1)
        return np.full((len(candidates), 1), default_value, dtype=np.float32)

    dense_blocks: list[np.ndarray] = [
        candidates["retrieval_score"].to_numpy(dtype=np.float32).reshape(-1, 1),
        cosine_similarity,
        np.asarray([len(history) for history in candidates["history_item_idxs"]], dtype=np.float32).reshape(-1, 1),
        candidates["user_interaction_count"].to_numpy(dtype=np.float32).reshape(-1, 1),
        candidates["user_mean_rating"].to_numpy(dtype=np.float32).reshape(-1, 1),
        candidates["user_verified_rate"].to_numpy(dtype=np.float32).reshape(-1, 1),
        candidates["days_since_last"].to_numpy(dtype=np.float32).reshape(-1, 1),
        candidates["avg_days_between"].to_numpy(dtype=np.float32).reshape(-1, 1),
    ]
    preference_columns = [f"pref_{category}" for category in prepared.config.categories]
    if preference_columns:
        dense_blocks.append(candidates[preference_columns].to_numpy(dtype=np.float32))
    for source_name in ["cooccurrence", "latent_cf", "content_based", "two_tower", "popularity"]:
        dense_blocks.extend(
            [
                _candidate_column(f"from_{source_name}", 0.0),
                _candidate_column(f"score_{source_name}", 0.0),
                _candidate_column(f"rank_{source_name}", float(_candidate_source_budgets(prepared.config).get(source_name, prepared.config.candidate_union_top_k) + 1)),
            ]
        )
    dense_blocks.extend(
        [
            item_dense_rows,
            np.sum(user_text_rows * item_text_rows, axis=1, keepdims=True, dtype=np.float32),
            category_alignment,
            np.abs(item_price_rows - history_price_rows),
            np.abs(item_rating_rows - history_rating_rows),
            _candidate_column("source_count", 0.0),
            _candidate_column("union_score", 0.0),
            np.fromiter(
                (
                    _hard_negative_flag_for_candidate(str(row.user_id), int(row.item_idx), row.target_timestamp, split_artifacts.hard_negative_history)
                    for row in candidates.itertuples(index=False)
                ),
                dtype=np.float32,
                count=len(candidates),
            ).reshape(-1, 1),
        ]
    )
    metadata = candidates[
        [
            "example_id",
            "split",
            "user_id",
            "user_idx",
            "item_idx",
            "retrieval_score",
            "rank",
            "label",
            "target_item_idx",
        ]
    ].copy()
    metadata["item_category_idx"] = item_category_idx
    features = {
        "user_idx": metadata["user_idx"].to_numpy(dtype=np.int32),
        "item_idx": item_indices,
        "item_category_idx": item_category_idx,
        "dense_features": np.concatenate(dense_blocks, axis=1).astype(np.float32, copy=False),
        "user_embedding": aligned_user_embeddings,
        "item_embedding": aligned_item_embeddings,
    }
    labels = metadata["label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    del user_embedding_inputs
    del unique_examples
    del user_embeddings
    del user_text_profiles
    del history_price_means
    del history_rating_means
    del candidate_example_rows
    del item_dense_matrix
    del item_dense_rows
    del user_text_rows
    del item_text_rows
    del dense_blocks
    gc.collect()
    return metadata, features, labels


def _ranker_dense_feature_columns(config: PipelineConfig) -> list[str]:
    return [
        "retrieval_score",
        "cosine_similarity",
        "history_item_count",
        "user_interaction_count",
        "user_mean_rating",
        "user_verified_rate",
        "days_since_last",
        "avg_days_between",
        *[f"pref_{category}" for category in config.categories],
        "from_cooccurrence",
        "score_cooccurrence",
        "rank_cooccurrence",
        "from_latent_cf",
        "score_latent_cf",
        "rank_latent_cf",
        "from_content_based",
        "score_content_based",
        "rank_content_based",
        "from_two_tower",
        "score_two_tower",
        "rank_two_tower",
        "from_popularity",
        "score_popularity",
        "rank_popularity",
        "price",
        "average_rating",
        "log_rating_number",
        "log_positive_count",
        "verified_purchase_rate_item",
        "helpful_vote_mean",
        "helpful_nonzero_rate",
        "days_since_last_interaction",
        "history_item_text_similarity",
        "category_preference_alignment",
        "price_delta_vs_history",
        "rating_delta_vs_history",
        "source_count",
        "union_score",
        "explicit_hard_negative",
    ]


def _build_ranker_model(config: PipelineConfig, num_users: int, num_items: int, num_categories: int, dense_dim: int, retriever_embedding_dim: int) -> tf.keras.Model:
    user_idx = tf.keras.Input(shape=(), dtype=tf.int32, name="user_idx")
    item_idx = tf.keras.Input(shape=(), dtype=tf.int32, name="item_idx")
    item_category_idx = tf.keras.Input(shape=(), dtype=tf.int32, name="item_category_idx")
    dense_features = tf.keras.Input(shape=(dense_dim,), dtype=tf.float32, name="dense_features")
    user_embedding = tf.keras.Input(shape=(retriever_embedding_dim,), dtype=tf.float32, name="user_embedding")
    item_embedding = tf.keras.Input(shape=(retriever_embedding_dim,), dtype=tf.float32, name="item_embedding")
    user_id_emb = tf.keras.layers.Embedding(num_users + 1, config.ranker_embedding_dim, mask_zero=True, name="ranker_user_embedding")(user_idx)
    user_id_emb = tf.keras.layers.Flatten()(user_id_emb)
    item_id_emb = tf.keras.layers.Embedding(num_items + 1, config.ranker_embedding_dim, mask_zero=True, name="ranker_item_embedding")(item_idx)
    item_id_emb = tf.keras.layers.Flatten()(item_id_emb)
    category_emb = tf.keras.layers.Embedding(num_categories + 1, config.ranker_embedding_dim, mask_zero=True, name="ranker_category_embedding")(item_category_idx)
    category_emb = tf.keras.layers.Flatten()(category_emb)
    dense_projection = tf.keras.layers.Dense(config.ranker_embedding_dim, activation="relu", name="dense_projection")(dense_features)
    interaction_vectors = [user_id_emb, item_id_emb, category_emb, dense_projection]
    interaction_terms = []
    for left_index in range(len(interaction_vectors)):
        for right_index in range(left_index + 1, len(interaction_vectors)):
            interaction_terms.append(
                tf.keras.layers.Dot(axes=1, name=f"ranker_interaction_{left_index}_{right_index}")(
                    [interaction_vectors[left_index], interaction_vectors[right_index]]
                )
            )
    ranker_input = tf.keras.layers.Concatenate(name="ranker_concat")(
        [dense_features, user_embedding, item_embedding, user_id_emb, item_id_emb, category_emb, dense_projection, *interaction_terms]
    )
    x = ranker_input
    for index, dim in enumerate(config.ranker_hidden_dims, start=1):
        x = tf.keras.layers.Dense(dim, activation="relu", name=f"ranker_dense_{index}")(x)
        x = tf.keras.layers.Dropout(0.1, name=f"ranker_dropout_{index}")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="ranker_output")(x)
    model = tf.keras.Model(
        inputs={
            "user_idx": user_idx,
            "item_idx": item_idx,
            "item_category_idx": item_category_idx,
            "dense_features": dense_features,
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
        },
        outputs=output,
        name="dlrm_lite_ranker",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def _rebalance_ranker_candidates(candidates: pd.DataFrame, negatives_per_positive: int, seed: int) -> pd.DataFrame:
    if candidates.empty or negatives_per_positive <= 0:
        return candidates
    sampled_groups: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for _, group in candidates.groupby("example_id", sort=False):
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        if positives.empty:
            sampled_groups.append(group.sort_values("rank").head(max(1, negatives_per_positive)))
            continue
        max_negatives = max(len(positives) * negatives_per_positive, negatives_per_positive)
        if len(negatives) > max_negatives:
            negatives = negatives.sample(n=max_negatives, random_state=int(rng.integers(0, 1_000_000)))
        sampled_groups.append(pd.concat([positives, negatives], ignore_index=True).sort_values("rank"))
    return pd.concat(sampled_groups, ignore_index=True)


def _select_embedding_retriever(retrievers: RetrieverArtifacts | dict[str, RetrieverArtifacts]) -> RetrieverArtifacts:
    if isinstance(retrievers, RetrieverArtifacts):
        return retrievers
    for key in ["two_tower", "latent_cf", "content_based"]:
        if key in retrievers:
            return retrievers[key]
    return next(iter(retrievers.values()))


def _iter_frame_batches(frame: pd.DataFrame, batch_size: int | None) -> Iterable[pd.DataFrame]:
    if batch_size is None or batch_size <= 0 or len(frame) <= batch_size:
        yield frame
        return
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start:start + batch_size].copy()


def _candidate_union_for_examples(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: dict[str, RetrieverArtifacts],
    examples: pd.DataFrame,
    inject_target_if_missing: bool = True,
) -> pd.DataFrame:
    batch_size = int(prepared.config.candidate_union_batch_size) if prepared.config.candidate_union_batch_size else None
    candidate_batches: list[pd.DataFrame] = []
    for example_batch in _iter_frame_batches(examples, batch_size):
        candidates = generate_candidate_union(
            prepared,
            split_artifacts,
            retrievers,
            example_batch,
            top_k=prepared.config.candidate_union_top_k,
            inject_target_if_missing=inject_target_if_missing,
            include_candidate_sources=False,
        )
        candidate_batches.append(candidates)
        gc.collect()
    if not candidate_batches:
        return pd.DataFrame()
    return pd.concat(candidate_batches, ignore_index=True)


def _ranker_candidates_for_examples(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: RetrieverArtifacts | dict[str, RetrieverArtifacts],
    examples: pd.DataFrame,
    inject_target_if_missing: bool = True,
) -> pd.DataFrame:
    batch_size = int(prepared.config.candidate_union_batch_size) if prepared.config.candidate_union_batch_size else None
    candidate_batches: list[pd.DataFrame] = []
    if isinstance(retrievers, dict):
        for example_batch in _iter_frame_batches(examples, batch_size):
            candidates = generate_candidate_union(
                prepared,
                split_artifacts,
                retrievers,
                example_batch,
                top_k=prepared.config.candidate_union_top_k,
                inject_target_if_missing=inject_target_if_missing,
                include_candidate_sources=False,
            )
            pruned = (
                candidates.sort_values(["example_id", "retrieval_score"], ascending=[True, False])
                .groupby("example_id", group_keys=False)
                .head(prepared.config.ranker_candidate_top_k)
                .reset_index(drop=True)
            )
            candidate_batches.append(pruned)
            del candidates
            gc.collect()
    else:
        for example_batch in _iter_frame_batches(examples, batch_size):
            candidates = generate_candidates(
                prepared,
                split_artifacts,
                retrievers,
                example_batch,
                top_k=prepared.config.ranker_candidate_top_k,
                inject_target_if_missing=inject_target_if_missing,
            )
            candidate_batches.append(candidates)
    if not candidate_batches:
        return pd.DataFrame()
    return pd.concat(candidate_batches, ignore_index=True)


def _xgboost_ranker_frame(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retriever: RetrieverArtifacts,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[int]]:
    metadata, features, labels = _build_ranker_payload(prepared, split_artifacts, retriever, candidates)
    feature_frame = pd.DataFrame(features["dense_features"], columns=_ranker_dense_feature_columns(prepared.config))
    feature_frame["item_category_idx"] = metadata["item_category_idx"].to_numpy(dtype=np.int32)
    feature_frame["rank"] = metadata["rank"].to_numpy(dtype=np.float32)
    metadata = metadata.reset_index(drop=True)
    feature_frame = feature_frame.reset_index(drop=True)
    sort_order = metadata.sort_values(["example_id", "rank"]).index.to_numpy(dtype=np.int32)
    metadata = metadata.iloc[sort_order].reset_index(drop=True)
    feature_frame = feature_frame.iloc[sort_order].reset_index(drop=True)
    labels = labels[sort_order]
    grouped = metadata.groupby("example_id", sort=False).size()
    return metadata, feature_frame, labels.reshape(-1).astype(np.int32), grouped.astype(int).tolist()


def evaluate_ranker(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: RetrieverArtifacts | dict[str, RetrieverArtifacts],
    ranker_model: RankerPredictor,
    backend: str = "xgboost",
) -> pd.DataFrame:
    embedding_retriever = _select_embedding_retriever(retrievers)
    rows: list[pd.DataFrame] = []
    union_diagnostic_frames: list[pd.DataFrame] = []
    served_distribution_frames: list[pd.DataFrame] = []
    for split_name, examples in [("val", split_artifacts.val_examples), ("test", split_artifacts.test_examples)]:
        eval_examples = examples
        eval_cap = prepared.config.ranker_val_example_cap if prepared.config.ranker_val_example_cap is not None else prepared.config.eval_user_cap
        if eval_cap is not None and len(eval_examples) > eval_cap:
            eval_examples = eval_examples.sample(n=eval_cap, random_state=prepared.config.seed).sort_values("example_id")
        LOGGER.info("Evaluating ranker on %s split: examples=%s", split_name, f"{len(eval_examples):,}")
        variant = "hybrid_union" if isinstance(retrievers, dict) else embedding_retriever.variant
        if isinstance(retrievers, dict):
            union_frame = _candidate_union_for_examples(
                prepared,
                split_artifacts,
                retrievers,
                eval_examples,
                inject_target_if_missing=False,
            )
            union_with_context = _add_candidate_item_context(prepared, union_frame)
            union_diagnostics = candidate_recall_diagnostics(
                union_with_context,
                split=split_name,
                variant=variant,
                stage="candidate_union",
            )
            union_diagnostics.to_csv(prepared.config.eval_dir / f"{variant}_{split_name}_candidate_union_diagnostics_metrics.csv", index=False)
            union_diagnostic_frames.append(union_diagnostics)
            candidate_frame = (
                union_frame.sort_values(["example_id", "retrieval_score"], ascending=[True, False])
                .groupby("example_id", group_keys=False)
                .head(prepared.config.ranker_candidate_top_k)
                .reset_index(drop=True)
            )
            del union_frame
            del union_with_context
        else:
            candidate_frame = _ranker_candidates_for_examples(
                prepared,
                split_artifacts,
                retrievers,
                eval_examples,
                inject_target_if_missing=False,
            )
        candidate_frame_with_context = _add_candidate_item_context(prepared, candidate_frame)
        diagnostics = candidate_recall_diagnostics(
            candidate_frame_with_context,
            split=split_name,
            variant=variant,
            stage="ranker_candidates",
        )
        diagnostics.to_csv(prepared.config.eval_dir / f"{variant}_{split_name}_candidate_diagnostics_metrics.csv", index=False)
        if backend == "xgboost":
            ranker_metadata, feature_frame, _, _ = _xgboost_ranker_frame(prepared, split_artifacts, embedding_retriever, candidate_frame)
            ranker_metadata["ranker_score"] = ranker_model.predict(feature_frame).reshape(-1)
            del feature_frame
        else:
            ranker_metadata, features, _ = _build_ranker_payload(prepared, split_artifacts, embedding_retriever, candidate_frame)
            ranker_metadata["ranker_score"] = ranker_model.predict(features, batch_size=prepared.config.ranker_batch_size, verbose=0).reshape(-1)
            del features
        ranked = ranker_metadata.sort_values(["example_id", "ranker_score"], ascending=[True, False]).copy()
        ranked["rank"] = ranked.groupby("example_id").cumcount() + 1
        ranked_with_context = _add_candidate_item_context(prepared, ranked)
        served_distribution_frames.append(
            candidate_distribution_by_category_price(
                ranked_with_context,
                split=split_name,
                variant=variant,
                stage="ranker_top_10",
                top_k=10,
            )
        )
        metrics = _metrics_from_ranked_candidates(ranked.rename(columns={"ranker_score": "retrieval_score"}), ks=(10, 50, 100))
        metrics["split"] = split_name
        metrics["variant"] = variant
        metrics["stage"] = "ranker"
        rows.append(metrics)
        ranked_with_context.to_parquet(prepared.config.eval_dir / f"{metrics.iloc[0]['variant']}_{split_name}_ranked_candidates.parquet", index=False)
        del candidate_frame
        del candidate_frame_with_context
        del ranker_metadata
        del ranked
        del ranked_with_context
        gc.collect()
    if union_diagnostic_frames:
        union_diagnostics = pd.concat(union_diagnostic_frames, ignore_index=True)
        scope_outputs = {
            "target_category": "candidate_union_recall_by_category.csv",
            "candidate_source": "candidate_union_recall_by_source.csv",
            "history_length_bucket": "candidate_union_recall_by_history_bucket.csv",
            "target_price_bucket": "candidate_union_recall_by_price_bucket.csv",
        }
        for scope, filename in scope_outputs.items():
            scope_frame = union_diagnostics[union_diagnostics["scope"] == scope].copy()
            if not scope_frame.empty:
                scope_frame.to_csv(prepared.config.eval_dir / filename, index=False)
    if served_distribution_frames:
        served_distribution = pd.concat(served_distribution_frames, ignore_index=True)
        if not served_distribution.empty:
            served_distribution.to_csv(prepared.config.eval_dir / "served_distribution_by_category_price.csv", index=False)
    if not rows:
        return _normalize_metrics_frame(None, extra_columns=("split", "variant", "stage"))
    return pd.concat(rows, ignore_index=True)


def train_ranker(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: RetrieverArtifacts | dict[str, RetrieverArtifacts],
    backend: str | None = None,
) -> RankerArtifacts:
    config = prepared.config
    backend = backend or config.ranker_backend
    embedding_retriever = _select_embedding_retriever(retrievers)
    train_examples = split_artifacts.train_examples
    if config.ranker_train_example_cap is not None and len(train_examples) > config.ranker_train_example_cap:
        train_examples = train_examples.sample(n=config.ranker_train_example_cap, random_state=config.seed).sort_values("example_id")
    val_examples = split_artifacts.val_examples
    val_cap = config.ranker_val_example_cap if config.ranker_val_example_cap is not None else config.eval_user_cap
    if val_cap is not None and len(val_examples) > val_cap:
        val_examples = val_examples.sample(n=val_cap, random_state=config.seed).sort_values("example_id")
    LOGGER.info(
        "Preparing ranker candidates: backend=%s train_examples=%s val_examples=%s candidate_top_k=%s negatives_per_positive=%s",
        backend,
        f"{len(train_examples):,}",
        f"{len(val_examples):,}",
        config.ranker_candidate_top_k,
        config.ranker_negatives_per_positive,
    )
    train_candidates = _ranker_candidates_for_examples(
        prepared,
        split_artifacts,
        retrievers,
        train_examples,
        inject_target_if_missing=True,
    )
    val_candidates = _ranker_candidates_for_examples(
        prepared,
        split_artifacts,
        retrievers,
        val_examples,
        inject_target_if_missing=True,
    )
    train_candidates = _rebalance_ranker_candidates(train_candidates, config.ranker_negatives_per_positive, config.seed)
    val_candidates = _rebalance_ranker_candidates(val_candidates, config.ranker_negatives_per_positive, config.seed)
    LOGGER.info(
        "Ranker candidate tables ready: train_candidates=%s val_candidates=%s",
        f"{len(train_candidates):,}",
        f"{len(val_candidates):,}",
    )
    if backend == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is required for backend='xgboost' but is not installed.")
        _, train_frame, train_labels, train_group = _xgboost_ranker_frame(prepared, split_artifacts, embedding_retriever, train_candidates)
        _, val_frame, val_labels, val_group = _xgboost_ranker_frame(prepared, split_artifacts, embedding_retriever, val_candidates)
        del train_candidates
        del val_candidates
        gc.collect()
        model = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric=["ndcg@10", "ndcg@20"],
            learning_rate=config.xgb_learning_rate,
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            tree_method="hist",
            random_state=config.seed,
        )
        LOGGER.info(
            "Fitting XGBoost ranker: rows=%s groups=%s n_estimators=%s max_depth=%s",
            f"{len(train_frame):,}",
            f"{len(train_group):,}",
            config.xgb_n_estimators,
            config.xgb_max_depth,
        )
        model.fit(
            train_frame,
            train_labels,
            group=train_group,
            eval_set=[(val_frame, val_labels)],
            eval_group=[val_group],
            verbose=bool(config.training_verbose),
        )
        history = model.evals_result()
        del train_frame
        del train_labels
        del train_group
        del val_frame
        del val_labels
        del val_group
        gc.collect()
        model_path = config.model_dir / "hybrid_union_xgboost_ranker.json"
        model.save_model(model_path)
    else:
        _, train_features, train_labels = _build_ranker_payload(prepared, split_artifacts, embedding_retriever, train_candidates)
        _, val_features, val_labels = _build_ranker_payload(prepared, split_artifacts, embedding_retriever, val_candidates)
        del train_candidates
        del val_candidates
        gc.collect()
        train_positive_count = max(float(train_labels.sum()), 1.0)
        train_negative_count = max(float(len(train_labels) - train_positive_count), 1.0)
        positive_weight = min(train_negative_count / train_positive_count, 10.0)
        train_sample_weight = np.where(train_labels.reshape(-1) > 0.5, positive_weight, 1.0).astype(np.float32)
        val_sample_weight = np.where(val_labels.reshape(-1) > 0.5, positive_weight, 1.0).astype(np.float32)
        model = _build_ranker_model(
            config,
            num_users=len(split_artifacts.user_id_to_idx),
            num_items=len(prepared.item_features),
            num_categories=len(prepared.category_to_idx),
            dense_dim=train_features["dense_features"].shape[1],
            retriever_embedding_dim=train_features["user_embedding"].shape[1],
        )
        LOGGER.info(
            "Fitting DLRM ranker: rows=%s epochs=%s batch_size=%s",
            f"{len(train_labels):,}",
            config.ranker_epochs,
            config.ranker_batch_size,
        )
        history = model.fit(
            train_features,
            train_labels,
            sample_weight=train_sample_weight,
            validation_data=(val_features, val_labels, val_sample_weight),
            epochs=config.ranker_epochs,
            batch_size=config.ranker_batch_size,
            verbose=config.training_verbose,
        ).history
        del train_features
        del train_labels
        del train_sample_weight
        del val_features
        del val_labels
        del val_sample_weight
        gc.collect()
        model.save_weights(config.model_dir / f"{embedding_retriever.variant}_ranker.weights.h5")
        model.save(config.model_dir / f"{embedding_retriever.variant}_ranker.keras", overwrite=True)
    LOGGER.info("Evaluating trained ranker")
    metrics = evaluate_ranker(prepared, split_artifacts, retrievers, model, backend=backend)
    output_name = "hybrid_union" if isinstance(retrievers, dict) else embedding_retriever.variant
    metrics.to_csv(config.eval_dir / f"{output_name}_{backend}_ranker_metrics.csv", index=False)
    return RankerArtifacts(
        config=config,
        model=model,
        metrics=metrics,
        history=history,
        selected_retriever_variant=output_name,
        backend=backend,
    )


def get_user_order_history(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    user_id: str,
    split: str = "test",
    limit: int | None = 15,
) -> pd.DataFrame:
    user_id = str(user_id)
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be one of 'train', 'val', 'test', or 'all'.")
    user_history = prepared.interactions[prepared.interactions["user_id"].astype(str) == user_id].copy()
    if user_history.empty:
        raise KeyError(f"Unknown user_id: {user_id}")
    if split in {"val", "test"}:
        example_frame = split_artifacts.val_examples if split == "val" else split_artifacts.test_examples
        example_rows = example_frame[example_frame["user_id"].astype(str) == user_id].sort_values("target_timestamp")
        if example_rows.empty:
            raise KeyError(f"No {split} example found for user_id: {user_id}")
        cutoff_ts = pd.Timestamp(example_rows.iloc[0]["target_timestamp"])
        user_history = user_history[user_history["timestamp_dt"] < cutoff_ts]
    user_history = user_history.sort_values("timestamp_dt").copy()
    if limit is not None and len(user_history) > limit:
        user_history = user_history.tail(limit).copy()
    item_meta = prepared.item_features[["parent_asin", "title", "source_category", "price", "average_rating"]].copy()
    user_history = user_history.drop(columns=["source_category"], errors="ignore")
    history = user_history.merge(item_meta, on="parent_asin", how="left")
    history["ordered_at"] = history["timestamp_dt"].dt.strftime("%Y-%m-%d")
    history["review_rating"] = history["rating"].astype(float)
    history["verified_purchase"] = history["verified_purchase"].astype(int)
    return history[
        [
            "ordered_at",
            "parent_asin",
            "title",
            "source_category",
            "review_rating",
            "verified_purchase",
            "price",
            "average_rating",
        ]
    ].reset_index(drop=True)


def build_serving_index(prepared: PreparedArtifacts, split_artifacts: SplitArtifacts) -> ServingIndex:
    examples = split_artifacts.test_examples.copy()
    summary_columns = ["user_id", "interaction_count", "history_length", "last_ordered_at", "user_idx"]
    history_columns = [
        "user_id",
        "item_idx",
        "parent_asin",
        "ordered_at",
        "timestamp_dt",
        "review_rating",
        "verified_purchase",
        "title",
        "source_category",
        "price",
        "average_rating",
    ]
    if examples.empty or prepared.interactions.empty:
        return ServingIndex(
            user_summary=pd.DataFrame(columns=summary_columns),
            user_history=pd.DataFrame(columns=history_columns),
        )

    examples["user_id"] = examples["user_id"].astype(str)
    examples["history_length"] = pd.to_numeric(examples["history_length"], errors="coerce").fillna(0).astype(int)
    examples["target_timestamp"] = pd.to_datetime(examples["target_timestamp"], utc=False)
    example_cutoffs = (
        examples.sort_values("target_timestamp")
        .drop_duplicates("user_id")
        [["user_id", "target_timestamp"]]
        .copy()
    )
    example_users = set(example_cutoffs["user_id"])

    interactions = prepared.interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions = interactions[interactions["user_id"].isin(example_users)].copy()
    if interactions.empty:
        return ServingIndex(
            user_summary=pd.DataFrame(columns=summary_columns),
            user_history=pd.DataFrame(columns=history_columns),
        )
    if "timestamp_dt" not in interactions.columns:
        interactions["timestamp_dt"] = pd.to_datetime(interactions["timestamp"], unit="ms", errors="coerce")
    if "item_idx" not in interactions.columns:
        interactions["item_idx"] = interactions["parent_asin"].map(prepared.item_id_to_idx)
    interactions = interactions.dropna(subset=["item_idx", "timestamp_dt"]).copy()
    interactions["item_idx"] = interactions["item_idx"].astype(int)

    interaction_counts = (
        interactions.groupby("user_id", as_index=False)
        .size()
        .rename(columns={"size": "interaction_count"})
    )
    last_orders = (
        interactions.groupby("user_id", as_index=False)["timestamp_dt"]
        .max()
        .rename(columns={"timestamp_dt": "last_ordered_at"})
    )
    last_orders["last_ordered_at"] = last_orders["last_ordered_at"].dt.strftime("%Y-%m-%d")
    user_idx_frame = pd.DataFrame(
        {
            "user_id": list(split_artifacts.user_id_to_idx.keys()),
            "user_idx": list(split_artifacts.user_id_to_idx.values()),
        }
    )
    user_summary = (
        examples.groupby("user_id", as_index=False)["history_length"]
        .max()
        .merge(interaction_counts, on="user_id", how="left")
        .merge(last_orders, on="user_id", how="left")
        .merge(user_idx_frame, on="user_id", how="left")
    )
    user_summary["interaction_count"] = user_summary["interaction_count"].fillna(0).astype(int)
    user_summary["user_idx"] = user_summary["user_idx"].fillna(0).astype(int)
    user_summary = user_summary[summary_columns].sort_values(
        ["interaction_count", "history_length", "user_id"],
        ascending=[False, False, True],
    )

    history = interactions.merge(example_cutoffs, on="user_id", how="inner")
    history = history[history["timestamp_dt"] < history["target_timestamp"]].copy()
    if history.empty:
        return ServingIndex(
            user_summary=user_summary.reset_index(drop=True),
            user_history=pd.DataFrame(columns=history_columns),
        )
    item_meta = prepared.item_features[["parent_asin", "title", "source_category", "price", "average_rating"]].copy()
    history = history.drop(columns=["source_category"], errors="ignore")
    history = history.merge(item_meta, on="parent_asin", how="left")
    history["ordered_at"] = history["timestamp_dt"].dt.strftime("%Y-%m-%d")
    history["review_rating"] = pd.to_numeric(history["rating"], errors="coerce")
    history["verified_purchase"] = pd.to_numeric(history["verified_purchase"], errors="coerce").fillna(0).astype(int)
    history = history.sort_values(["user_id", "timestamp_dt", "parent_asin"]).reset_index(drop=True)
    return ServingIndex(
        user_summary=user_summary.reset_index(drop=True),
        user_history=history[history_columns].reset_index(drop=True),
    )


def _resolve_history_item_idxs(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    *,
    user_id: str | None,
    history_items: list[str] | None,
) -> list[int]:
    if user_id is None and not history_items:
        raise ValueError("Provide either user_id or history_items.")
    resolved_history_items = history_items
    if resolved_history_items is None:
        history_frame = get_user_order_history(prepared, split_artifacts, str(user_id), split="test", limit=None)
        resolved_history_items = history_frame["parent_asin"].tolist()
    history_item_idxs = [prepared.item_id_to_idx[item] for item in resolved_history_items if item in prepared.item_id_to_idx]
    if not history_item_idxs:
        raise ValueError("No usable history items were found in the current item catalog.")
    return history_item_idxs


def _build_inference_request_context(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    *,
    user_id: str | None,
    history_items: list[str] | None,
) -> InferenceRequestContext:
    history_item_idxs = _resolve_history_item_idxs(
        prepared,
        split_artifacts,
        user_id=user_id,
        history_items=history_items,
    )
    user_identifier = str(user_id or "__cold_start__")
    user_idx = split_artifacts.user_id_to_idx.get(user_identifier, 0)
    prefix_df = pd.DataFrame(
        {
            "item_idx": history_item_idxs,
            "rating": [5.0] * len(history_item_idxs),
            "verified_purchase": [1] * len(history_item_idxs),
            "timestamp": np.arange(len(history_item_idxs), dtype=np.int64),
        }
    )
    item_categories = dict(zip(prepared.item_features["item_idx"], prepared.item_features["source_category"]))
    prefix_features = _compute_prefix_features(
        prefix_df.tail(prepared.config.history_len),
        int(prefix_df["timestamp"].max()) + 1,
        item_categories,
        prepared.config.categories,
    )
    target_item_idx = history_item_idxs[-1]
    inference_example = pd.DataFrame(
        [
            {
                "example_id": 1,
                "split": "inference",
                "user_id": user_identifier,
                "user_idx": user_idx,
                "target_item_idx": target_item_idx,
                "target_parent_asin": prepared.item_idx_to_id[target_item_idx],
                "target_source_category": str(item_categories.get(target_item_idx, "")),
                "target_timestamp": pd.Timestamp.utcnow(),
                **prefix_features,
            }
        ]
    )
    return InferenceRequestContext(
        user_identifier=user_identifier,
        history_item_idxs=history_item_idxs,
        history_item_set=set(history_item_idxs),
        inference_example=inference_example,
    )


def _exclude_seen_candidates(candidates: pd.DataFrame, seen_items: set[int]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    return candidates[~candidates["item_idx"].isin(seen_items)].copy()


def _recommendation_item_metadata(prepared: PreparedArtifacts) -> pd.DataFrame:
    return prepared.item_features[["item_idx", "parent_asin", "title", "source_category", "price", "average_rating"]].copy()


def _empty_recommendation_output(*, include_candidate_sources: bool, include_score: bool) -> pd.DataFrame:
    columns = ["parent_asin", "title", "source_category", "price", "average_rating"]
    if include_candidate_sources:
        columns.append("candidate_sources")
    columns.append("retrieval_score")
    if include_score:
        columns.append("score")
    return pd.DataFrame(columns=columns)


def _score_inference_candidates(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retriever: RetrieverArtifacts | dict[str, RetrieverArtifacts],
    ranker: RankerArtifacts | None,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if ranker is None or candidates.empty:
        return candidates
    embedding_retriever = _select_embedding_retriever(retriever)
    if ranker.backend == "xgboost":
        ranker_metadata, feature_frame, _, _ = _xgboost_ranker_frame(prepared, split_artifacts, embedding_retriever, candidates)
        ranker_metadata["score"] = ranker.model.predict(feature_frame).reshape(-1)
        del feature_frame
        gc.collect()
        return ranker_metadata
    ranker_metadata, features, _ = _build_ranker_payload(prepared, split_artifacts, embedding_retriever, candidates)
    ranker_metadata["score"] = ranker.model.predict(features, batch_size=prepared.config.ranker_batch_size, verbose=0).reshape(-1)
    del features
    gc.collect()
    return ranker_metadata


def _finalize_recommendation_output(
    prepared: PreparedArtifacts,
    candidates: pd.DataFrame,
    ranked_candidates: pd.DataFrame,
    *,
    top_k: int,
    include_candidate_sources: bool,
    include_score: bool,
) -> pd.DataFrame:
    if ranked_candidates.empty:
        return _empty_recommendation_output(
            include_candidate_sources=include_candidate_sources,
            include_score=include_score,
        )
    output = (
        ranked_candidates.sort_values("score" if include_score else "retrieval_score", ascending=False)
        .drop_duplicates("item_idx")
        .head(top_k)
        .merge(_recommendation_item_metadata(prepared), on="item_idx", how="left")
    )
    if include_candidate_sources and "candidate_sources" in candidates.columns:
        output = output.merge(
            candidates[["item_idx", "candidate_sources"]].drop_duplicates("item_idx"),
            on="item_idx",
            how="left",
        )
    columns = ["parent_asin", "title", "source_category", "price", "average_rating"]
    if include_candidate_sources:
        columns.append("candidate_sources")
    columns.append("retrieval_score")
    if include_score:
        columns.append("score")
    return output[columns].reset_index(drop=True)


def recommend_hybrid(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retrievers: dict[str, RetrieverArtifacts],
    ranker: RankerArtifacts | None = None,
    user_id: str | None = None,
    history_items: list[str] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    request = _build_inference_request_context(
        prepared,
        split_artifacts,
        user_id=user_id,
        history_items=history_items,
    )
    candidates = generate_candidate_union(
        prepared,
        split_artifacts,
        retrievers,
        request.inference_example,
        top_k=max(top_k * 10, prepared.config.candidate_union_top_k),
        include_candidate_sources=True,
    )
    candidates = _exclude_seen_candidates(candidates, request.history_item_set)
    ranked_candidates = _score_inference_candidates(
        prepared,
        split_artifacts,
        retrievers,
        ranker,
        candidates,
    )
    return _finalize_recommendation_output(
        prepared,
        candidates,
        ranked_candidates,
        top_k=top_k,
        include_candidate_sources=True,
        include_score=ranker is not None,
    )


def recommend(
    prepared: PreparedArtifacts,
    split_artifacts: SplitArtifacts,
    retriever: RetrieverArtifacts | dict[str, RetrieverArtifacts],
    ranker: RankerArtifacts | None = None,
    user_id: str | None = None,
    history_items: list[str] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    if isinstance(retriever, dict):
        return recommend_hybrid(prepared, split_artifacts, retriever, ranker=ranker, user_id=user_id, history_items=history_items, top_k=top_k)
    request = _build_inference_request_context(
        prepared,
        split_artifacts,
        user_id=user_id,
        history_items=history_items,
    )
    candidate_top_k = max(top_k * 5, prepared.config.retrieval_top_k)
    candidates = generate_candidates(prepared, split_artifacts, retriever, request.inference_example, top_k=candidate_top_k)
    candidates = _exclude_seen_candidates(candidates, request.history_item_set)
    ranked_candidates = _score_inference_candidates(
        prepared,
        split_artifacts,
        retriever,
        ranker,
        candidates,
    )
    return _finalize_recommendation_output(
        prepared,
        candidates,
        ranked_candidates,
        top_k=top_k,
        include_candidate_sources=False,
        include_score=ranker is not None,
    )


def pipeline_summary(config: PipelineConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "base_dir", "value": str(config.base_dir)},
            {"name": "artifact_root", "value": str(config.artifact_root)},
            {"name": "cache_dir", "value": str(config.cache_dir)},
            {"name": "model_dir", "value": str(config.model_dir)},
            {"name": "eval_dir", "value": str(config.eval_dir)},
            {"name": "categories", "value": ", ".join(config.categories)},
            {"name": "run_name", "value": config.run_name},
            {"name": "run_profile", "value": config.run_profile},
            {"name": "dev_mode", "value": config.dev_mode},
            {"name": "dev_fraction", "value": config.dev_fraction},
            {"name": "dev_sampling_strategy", "value": config.dev_sampling_strategy},
            {"name": "dev_hard_negative_multiplier", "value": config.dev_hard_negative_multiplier},
            {"name": "dev_neutral_multiplier", "value": config.dev_neutral_multiplier},
            {"name": "show_progress", "value": config.show_progress},
            {"name": "k_core", "value": config.k_core},
            {"name": "history_len", "value": config.history_len},
            {"name": "train_positive_cap", "value": config.train_positive_cap},
            {"name": "split_eval_example_cap", "value": config.split_eval_example_cap},
            {"name": "negatives_per_positive", "value": config.negatives_per_positive},
            {"name": "retriever_train_example_cap", "value": config.retriever_train_example_cap},
            {"name": "retriever_batch_size", "value": config.retriever_batch_size},
            {"name": "retriever_validation_negatives_per_positive", "value": config.retriever_validation_negatives_per_positive},
            {"name": "retriever_quality_min_history", "value": config.retriever_quality_min_history},
            {"name": "retriever_logit_scale", "value": config.retriever_logit_scale},
            {"name": "persist_encoder_models", "value": config.persist_encoder_models},
            {"name": "enable_neural_retriever", "value": config.enable_neural_retriever},
            {"name": "retrieval_top_k", "value": config.retrieval_top_k},
            {"name": "cooccurrence_candidate_k", "value": config.cooccurrence_candidate_k},
            {"name": "latent_cf_candidate_k", "value": config.latent_cf_candidate_k},
            {"name": "content_candidate_k", "value": config.content_candidate_k},
            {"name": "neural_candidate_k", "value": config.neural_candidate_k},
            {"name": "popularity_backfill_k", "value": config.popularity_backfill_k},
            {"name": "category_backfill_enabled", "value": config.category_backfill_enabled},
            {"name": "recency_cooccurrence_enabled", "value": config.recency_cooccurrence_enabled},
            {"name": "candidate_union_top_k", "value": config.candidate_union_top_k},
            {"name": "candidate_union_batch_size", "value": config.candidate_union_batch_size},
            {"name": "ranker_candidate_top_k", "value": config.ranker_candidate_top_k},
            {"name": "ranker_train_example_cap", "value": config.ranker_train_example_cap},
            {"name": "ranker_val_example_cap", "value": config.ranker_val_example_cap},
            {"name": "ranker_negatives_per_positive", "value": config.ranker_negatives_per_positive},
            {"name": "ranker_batch_size", "value": config.ranker_batch_size},
            {"name": "ranker_backend", "value": config.ranker_backend},
            {"name": "latent_cf_components", "value": config.latent_cf_components},
            {"name": "xgb_n_estimators", "value": config.xgb_n_estimators},
            {"name": "xgb_learning_rate", "value": config.xgb_learning_rate},
            {"name": "xgb_max_depth", "value": config.xgb_max_depth},
        ]
    )


def save_config(config: PipelineConfig) -> Path:
    ensure_directories(config)
    config_path = config.artifact_root / "config.json"
    serializable = asdict(config)
    serializable["base_dir"] = str(config.base_dir)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    return config_path

