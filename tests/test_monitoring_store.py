from __future__ import annotations

from pathlib import Path

import pytest

from amazon_recsys.monitoring.store import LocalMonitoringStore


@pytest.mark.foundation
@pytest.mark.serving
def test_ingest_outcomes_missing_file_raises_clear_error(test_settings) -> None:
    store = LocalMonitoringStore(test_settings)

    with pytest.raises(FileNotFoundError) as exc_info:
        store.ingest_outcomes(Path("does-not-exist.csv"))

    message = str(exc_info.value)
    assert "Outcome source file was not found" in message
    assert "docs/examples/outcomes.example.csv" in message
