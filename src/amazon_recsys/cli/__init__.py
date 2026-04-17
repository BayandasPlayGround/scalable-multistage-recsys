from __future__ import annotations

def main(*args: object, **kwargs: object) -> int:
    from amazon_recsys.cli.main import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
