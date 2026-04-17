from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


def _register_module_aliases() -> None:
    current_module = sys.modules[__name__]
    for alias in ("amazon_recsys_pipeline", "amazon_recsys_legacy_pipeline"):
        sys.modules[alias] = current_module


def _reexport_core_namespace() -> None:
    from amazon_recsys.ml import core as core_module

    exported_names = [
        name
        for name in vars(core_module)
        if not (name.startswith("__") and name.endswith("__"))
    ]
    globals().update({name: getattr(core_module, name) for name in exported_names})
    globals()["__all__"] = list(
        getattr(core_module, "__all__", [name for name in exported_names if not name.startswith("_")])
    )


_bootstrap_src_path()
_register_module_aliases()
_reexport_core_namespace()
