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


def atomic_replace(temp_path: Path, final_path: Path) -> None:
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
    temp_path = _temporary_path(final, suffix=".npy")
    temp_path.unlink(missing_ok=True)
    memmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype=dtype, shape=shape)
    return memmap, temp_path
