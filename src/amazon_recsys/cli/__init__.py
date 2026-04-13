from __future__ import annotations

from typing import Any


def main(*args: Any, **kwargs: Any) -> int:
    from amazon_recsys.cli.main import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
