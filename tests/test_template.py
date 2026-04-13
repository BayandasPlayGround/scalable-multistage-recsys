from __future__ import annotations

from pathlib import Path

import pytest

from template import create_project_template


@pytest.mark.foundation
def test_template_creation_is_idempotent(workspace_dir: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    target_root = workspace_dir / "generated"

    created = create_project_template(target_root=target_root, source_root=source_root)

    assert created
    assert (target_root / "pyproject.toml").exists()
    assert (target_root / "src" / "amazon_recsys" / "api" / "app.py").exists()

    readme_path = target_root / "README.md"
    readme_path.write_text("keep-me", encoding="utf-8")
    create_project_template(target_root=target_root, source_root=source_root)

    assert readme_path.read_text(encoding="utf-8") == "keep-me"
