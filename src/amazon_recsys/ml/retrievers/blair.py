"""Phase-1 BLAIR-style frozen-text-encoder retriever.

Encodes each item's text once with a frozen pretrained sentence encoder (default BLAIR, with
sentence-transformers/all-MiniLM-L6-v2 as a fallback when BLAIR weights cannot be loaded),
builds an Annoy index over the resulting vectors, and serves with the existing
``mean(item_embeddings[history])`` query path that ``two_tower`` and ``dat_lite`` already use.

Production-safe scope:

* Item text is generated in chunks so production catalogs do not create a multi-million element
  Python list.
* Encoded item embeddings are written chunk-by-chunk to an ``.npy`` memmap, then loaded with
  ``mmap_mode="r"`` for ANN construction and bundle export.
* The trained retriever is wrapped exactly like ``train_content_retriever``: vector-only,
  no encoder objects, ANN index on disk, metrics CSV under ``eval_dir``.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from amazon_recsys.ml import core
from amazon_recsys.ml.io_utils import atomic_replace, open_atomic_memmap

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer  # noqa: F401


LOGGER = logging.getLogger(__name__)


def _load_sentence_transformer(model_name: str, fallback_model: str, max_seq_length: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BLAIR retriever requires sentence-transformers. Install ad-hoc with:\n"
            "    pip install sentence-transformers\n"
            "(this pulls in torch + transformers as transitive dependencies)."
        ) from exc

    try:
        encoder = SentenceTransformer(model_name)
        resolved_name = model_name
    except Exception as primary_error:  # noqa: BLE001 - any download/load failure triggers fallback
        LOGGER.warning(
            "Could not load BLAIR encoder %r (%s). Falling back to %r.",
            model_name,
            primary_error,
            fallback_model,
        )
        encoder = SentenceTransformer(fallback_model)
        resolved_name = fallback_model
    encoder.max_seq_length = int(max_seq_length)
    return encoder, resolved_name


_RICH_METADATA_COLUMNS = (
    "meta_title",
    "store",
    "categories_text",
    "description_text",
    "features_text",
)


def _metadata_lookup(metadata: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list[str]]:
    if metadata is None or metadata.empty or "parent_asin" not in metadata.columns:
        return None, []
    rich_cols_present = [col for col in _RICH_METADATA_COLUMNS if col in metadata.columns]
    if not rich_cols_present:
        return None, []
    lookup = metadata[["parent_asin", *rich_cols_present]].copy()
    lookup["parent_asin"] = lookup["parent_asin"].astype(str)
    lookup = lookup.drop_duplicates("parent_asin").set_index("parent_asin")
    return lookup, rich_cols_present


def _build_item_texts_from_lookup(
    item_features: pd.DataFrame,
    metadata_lookup: pd.DataFrame | None = None,
) -> tuple[list[str], dict[str, int]]:
    chunk = item_features.copy()
    if metadata_lookup is not None and not metadata_lookup.empty:
        parent_asins = chunk["parent_asin"].fillna("").astype(str)
        aligned = metadata_lookup.reindex(parent_asins)
        for col in _RICH_METADATA_COLUMNS:
            chunk[col] = aligned[col].to_numpy() if col in aligned.columns else ""
    else:
        for col in _RICH_METADATA_COLUMNS:
            chunk[col] = ""

    rich = chunk[list(_RICH_METADATA_COLUMNS)].fillna("").astype(str).agg(" ".join, axis=1)
    rich = rich.str.replace(r"\s+", " ", regex=True).str.strip()

    fallback = (
        chunk["title"].fillna("").astype(str)
        + " "
        + chunk["source_category"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    has_rich = rich.str.len() > 0
    final = rich.where(has_rich, fallback)

    counts = {
        "rich": int(has_rich.sum()),
        "fallback": int((~has_rich).sum()),
    }
    return final.tolist(), counts


def _build_item_texts(item_features: pd.DataFrame, metadata: pd.DataFrame | None = None) -> tuple[list[str], dict[str, int]]:
    """Build per-item text strings for BLAIR encoding.

    When ``metadata`` is provided, concat ``meta_title + store + categories_text +
    description_text + features_text`` per item — that is the rich text BLAIR was designed for.
    For items missing from metadata (or when ``metadata`` is None), fall back per row to the
    Phase-1 text (``title + source_category``) so we never feed the encoder an empty string.

    Returns the list of texts plus a counts dict ({"rich": N, "fallback": M}) for logging.
    """
    lookup, _ = _metadata_lookup(metadata)
    return _build_item_texts_from_lookup(item_features, lookup)


def _projection_matrix(raw_dim: int, target_dim: int, seed: int) -> np.ndarray | None:
    if int(target_dim) <= 0 or int(target_dim) >= int(raw_dim):
        return None
    rng = np.random.default_rng(int(seed) + 7919)
    return rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(target_dim)),
        size=(int(raw_dim), int(target_dim)),
    ).astype(np.float32)


def _project_and_normalize(embeddings: np.ndarray, projection: np.ndarray | None) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"BLAIR encoder returned embeddings with shape {vectors.shape}; expected 2D.")
    if projection is not None:
        vectors = vectors @ projection
    return core._normalize_rows(np.asarray(vectors, dtype=np.float32))


def _load_rich_metadata(config: core.PipelineConfig, item_features: pd.DataFrame) -> pd.DataFrame | None:
    """Reload raw item metadata in-memory so BLAIR can encode the rich text fields.

    Returns ``None`` when metadata cannot be obtained (no files on disk, download disabled,
    or any other read error). In that case callers must fall back to the Phase-1 text.
    """
    valid_parent_asins = set(item_features["parent_asin"].astype(str))
    try:
        metadata = core.load_metadata(
            config,
            categories=config.categories,
            download_if_missing=config.metadata_download_if_missing,
            valid_parent_asins=valid_parent_asins,
        )
    except Exception as exc:  # noqa: BLE001 - any read/parse failure should not abort training
        LOGGER.warning(
            "BLAIR could not reload raw metadata (%s). Falling back to title+source_category text.",
            exc,
        )
        return None
    if metadata is None or metadata.empty:
        LOGGER.warning(
            "BLAIR metadata reload returned an empty frame. Falling back to title+source_category text."
        )
        return None
    return metadata


def train_blair_retriever(
    prepared: core.PreparedArtifacts,
    split_artifacts: core.SplitArtifacts,
) -> core.RetrieverArtifacts:
    config = prepared.config
    item_features = prepared.item_features.sort_values("item_idx", kind="mergesort").reset_index(drop=True)
    metadata = _load_rich_metadata(config, item_features)
    metadata_index, rich_cols_present = _metadata_lookup(metadata)
    item_text_columns = (
        list(_RICH_METADATA_COLUMNS) if metadata_index is not None and not metadata_index.empty
        else ["title", "source_category"]
    )

    encoder, resolved_model = _load_sentence_transformer(
        config.blair_model_name,
        config.blair_fallback_model,
        config.blair_max_seq_length,
    )
    item_count = int(len(item_features))
    if item_count <= 0:
        raise ValueError("Cannot train BLAIR retriever with an empty item catalog.")
    LOGGER.info(
        "Encoding %d BLAIR item texts with %s (batch=%d, max_seq=%d chunk_rows=%d projection_dim=%d rich_metadata=%s)",
        item_count,
        resolved_model,
        int(config.blair_batch_size),
        int(config.blair_max_seq_length),
        int(config.blair_chunk_rows),
        int(config.blair_projection_dim),
        bool(rich_cols_present),
    )
    text_source_counts = {"rich": 0, "fallback": 0}
    embedding_path = config.model_dir / "blair_text_item_embeddings.npy"
    item_embeddings_writer: np.memmap | None = None
    temp_embedding_path = None
    projection: np.ndarray | None = None
    raw_embedding_dim: int | None = None
    embedding_dim: int | None = None
    chunk_rows = int(config.blair_chunk_rows)
    try:
        for start in range(0, item_count, chunk_rows):
            end = min(start + chunk_rows, item_count)
            texts, counts = _build_item_texts_from_lookup(item_features.iloc[start:end], metadata_index)
            text_source_counts["rich"] += counts["rich"]
            text_source_counts["fallback"] += counts["fallback"]
            raw_embeddings = encoder.encode(
                texts,
                batch_size=int(config.blair_batch_size),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
            if raw_embeddings.shape[0] != len(texts):
                raise ValueError(
                    f"BLAIR encoder returned {raw_embeddings.shape[0]} rows for {len(texts)} input texts."
                )
            if item_embeddings_writer is None:
                raw_embedding_dim = int(raw_embeddings.shape[1])
                projection = _projection_matrix(raw_embedding_dim, int(config.blair_projection_dim), config.seed)
                embedding_dim = int(projection.shape[1]) if projection is not None else raw_embedding_dim
                item_embeddings_writer, temp_embedding_path = open_atomic_memmap(
                    embedding_path,
                    dtype=np.float32,
                    shape=(item_count, embedding_dim),
                )
            projected = _project_and_normalize(raw_embeddings, projection)
            item_embeddings_writer[start:end] = projected
            item_embeddings_writer.flush()
            LOGGER.info(
                "BLAIR encoded chunk: rows=%s/%s raw_dim=%s embedding_dim=%s rich=%s fallback=%s",
                f"{end:,}",
                f"{item_count:,}",
                raw_embedding_dim,
                embedding_dim,
                f"{text_source_counts['rich']:,}",
                f"{text_source_counts['fallback']:,}",
            )
            del texts, raw_embeddings, projected
            gc.collect()
        if item_embeddings_writer is None or temp_embedding_path is None:
            raise RuntimeError("BLAIR encoding produced no embeddings.")
        item_embeddings_writer.flush()
        mmap_handle = getattr(item_embeddings_writer, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        del item_embeddings_writer
        gc.collect()
        atomic_replace(temp_embedding_path, embedding_path)
    except Exception:
        if temp_embedding_path is not None:
            if temp_embedding_path.resolve() != embedding_path.resolve():
                try:
                    temp_embedding_path.unlink(missing_ok=True)
                except PermissionError:
                    pass
        raise

    item_embeddings = np.load(embedding_path, mmap_mode="r")
    LOGGER.info(
        "BLAIR item embeddings ready: path=%s shape=%s rich=%d fallback=%d columns=%s",
        embedding_path,
        item_embeddings.shape,
        text_source_counts["rich"],
        text_source_counts["fallback"],
        item_text_columns,
    )

    ann_index_path = core.build_ann_index(
        config,
        item_embeddings,
        config.model_dir / "blair_text_item_index.ann",
        ann_trees=int(config.blair_ann_trees),
    )

    retriever = core.RetrieverArtifacts(
        config=config,
        variant="blair_text",
        model={"retriever": "blair_text", "encoder_name": resolved_model},
        item_encoder=None,
        user_encoder=None,
        item_embeddings=item_embeddings,
        ann_index_path=ann_index_path,
        ann_index=core._load_ann_index(item_embeddings.shape[1], ann_index_path),
        metrics=pd.DataFrame(),
        history={},
        retriever_kind="vector",
        metadata={
            "encoder_name": resolved_model,
            "configured_model_name": config.blair_model_name,
            "fallback_model_name": config.blair_fallback_model,
            "embedding_dim": int(item_embeddings.shape[1]),
            "raw_embedding_dim": int(raw_embedding_dim or item_embeddings.shape[1]),
            "projection_dim": int(item_embeddings.shape[1]),
            "chunk_rows": int(config.blair_chunk_rows),
            "ann_trees": int(config.blair_ann_trees),
            "batch_size": int(config.blair_batch_size),
            "max_seq_length": int(config.blair_max_seq_length),
            "serving_query": "history_item_embedding_mean",
            "item_text_columns": item_text_columns,
            "item_text_source_counts": text_source_counts,
        },
    )
    retriever.metrics = core.evaluate_retriever(prepared, split_artifacts, retriever)
    config.eval_dir.mkdir(parents=True, exist_ok=True)
    retriever.metrics.to_csv(config.eval_dir / "blair_text_retriever_metrics.csv", index=False)
    return retriever


__all__ = ["train_blair_retriever"]
