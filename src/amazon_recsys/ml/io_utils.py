from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def _temporary_path(path: Path, *, suffix: str | None = None) -> Path:
    final = Path(path)
    extension = suffix if suffix is not None else final.suffix
    return final.parent / f".{final.stem}.{uuid4().hex}.tmp{extension}"


def _is_onedrive_path(path: Path) -> bool:
    resolved = str(Path(path).resolve()).lower()
    if "onedrive" in resolved:
        return True
    for env_name in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        root = os.environ.get(env_name)
        if root and resolved.startswith(str(Path(root).resolve()).lower()):
            return True
    return False


def _artifact_write_mode(path: Path) -> str:
    """Choose the least surprising artifact write behavior for the current storage root.

    ``atomic`` writes are safest against partial files, but they create temp files and rapid
    replace/copy operations that can look like ransomware behavior on OneDrive-managed endpoints.
    ``auto`` therefore uses direct writes under OneDrive and atomic writes elsewhere. Override with:

    * ``AMAZON_RECSYS_ARTIFACT_WRITE_MODE=direct`` for endpoint-security-friendly local runs.
    * ``AMAZON_RECSYS_ARTIFACT_WRITE_MODE=atomic`` for non-synced production artifact roots.
    """
    raw = os.environ.get("AMAZON_RECSYS_ARTIFACT_WRITE_MODE", "auto").strip().lower()
    if raw in {"direct", "atomic"}:
        return raw
    if raw not in {"", "auto"}:
        raise ValueError("AMAZON_RECSYS_ARTIFACT_WRITE_MODE must be one of: auto, direct, atomic.")
    return "direct" if _is_onedrive_path(path) else "atomic"


def set_artifact_write_mode(mode: str) -> None:
    normalized = str(mode).strip().lower()
    if normalized not in {"auto", "direct", "atomic"}:
        raise ValueError("artifact write mode must be one of: auto, direct, atomic.")
    os.environ["AMAZON_RECSYS_ARTIFACT_WRITE_MODE"] = normalized


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    if Path(temp_path).resolve() == Path(final_path).resolve():
        return
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(temp_path, final_path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.1 * (attempt + 1), 0.75))
    try:
        # Some OneDrive-managed Windows folders deny rename/replace even when regular writes and
        # copies work. Use a best-effort copy fallback so training can still finish; callers still
        # get atomic replace semantics on normal filesystems.
        shutil.copy2(temp_path, final_path)
        _safe_unlink(temp_path)
        return
    except OSError as copy_error:
        cause = copy_error if copy_error is not None else last_error
        raise OSError(
            f"Failed to replace artifact {final_path} with temporary file {temp_path}. "
            "If this path is synced by OneDrive, pause sync or move AMAZON_RECSYS_ARTIFACT_ROOT "
            "outside OneDrive for production runs."
        ) from cause


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        return


def atomic_write_json(path: Path, payload: object, *, default=None, indent: int = 2) -> None:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if _artifact_write_mode(final) == "direct":
        with open(final, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, default=default)
        return
    temp_path = _temporary_path(final, suffix=".json")
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, default=default)
        atomic_replace(temp_path, final)
    except Exception:
        _safe_unlink(temp_path)
        raise


def atomic_write_parquet(path: Path, frame: pd.DataFrame, **kwargs) -> None:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if _artifact_write_mode(final) == "direct":
        frame.to_parquet(final, **kwargs)
        return
    temp_path = _temporary_path(final, suffix=".parquet")
    try:
        frame.to_parquet(temp_path, **kwargs)
        atomic_replace(temp_path, final)
    except Exception:
        _safe_unlink(temp_path)
        raise


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if _artifact_write_mode(final) == "direct":
        np.save(final, array)
        return
    temp_path = _temporary_path(final, suffix=".npy")
    try:
        np.save(temp_path, array)
        atomic_replace(temp_path, final)
    except Exception:
        _safe_unlink(temp_path)
        raise


def open_atomic_memmap(path: Path, *, dtype: str | np.dtype, shape: tuple[int, ...]) -> tuple[np.memmap, Path]:
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if _artifact_write_mode(final) == "direct":
        memmap = np.lib.format.open_memmap(final, mode="w+", dtype=dtype, shape=shape)
        return memmap, final
    temp_path = _temporary_path(final, suffix=".npy")
    temp_path.unlink(missing_ok=True)
    memmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype=dtype, shape=shape)
    return memmap, temp_path
