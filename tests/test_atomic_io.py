from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amazon_recsys.ml.io_utils import atomic_save_npy, atomic_write_json, atomic_write_parquet


def test_atomic_json_replace_preserves_existing_file_on_write_failure(monkeypatch: pytest.MonkeyPatch, workspace_dir) -> None:
    target = workspace_dir / "artifact.json"
    target.write_text('{"ok": true}', encoding="utf-8")

    def fail_dump(*_args, **_kwargs):
        raise RuntimeError("simulated write failure")

    import amazon_recsys.ml.io_utils as io_utils

    monkeypatch.setattr(io_utils.json, "dump", fail_dump)

    with pytest.raises(RuntimeError):
        atomic_write_json(target, {"ok": False})

    assert target.read_text(encoding="utf-8") == '{"ok": true}'


def test_atomic_npy_and_parquet_write_expected_payloads(workspace_dir) -> None:
    npy_path = workspace_dir / "matrix.npy"
    parquet_path = workspace_dir / "frame.parquet"

    atomic_save_npy(npy_path, np.asarray([[1.0, 2.0]], dtype=np.float32))
    atomic_write_parquet(parquet_path, pd.DataFrame([{"a": 1, "b": "x"}]), index=False)

    np.testing.assert_allclose(np.load(npy_path), np.asarray([[1.0, 2.0]], dtype=np.float32))
    assert pd.read_parquet(parquet_path).to_dict(orient="records") == [{"a": 1, "b": "x"}]
