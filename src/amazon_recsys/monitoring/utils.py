from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd


MONITORED_K = 10
STATUS_RANK = {"ok": 0, "insufficient_data": 0, "warn": 1, "alert": 2}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_utc_iso(value: Any) -> str:
    if value is None:
        raise ValueError("A timestamp value is required.")
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    if isinstance(timestamp, pd.Series):
        raise TypeError("Scalar timestamp expected.")
    return timestamp.to_pydatetime().replace(microsecond=0).isoformat()


def hash_user_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return hashlib.sha256(f"amazon-recsys-monitoring::{normalized}".encode("utf-8")).hexdigest()


def split_candidate_sources(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    normalized = str(raw_value).strip()
    if not normalized:
        return []
    return [part.strip() for part in normalized.split("+") if part.strip()]


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_filename(value: str) -> str:
    return (
        str(value)
        .replace(":", "-")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
    )


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def max_status(*statuses: str) -> str:
    if not statuses:
        return "ok"
    return max(statuses, key=lambda status: STATUS_RANK.get(status, 0))
