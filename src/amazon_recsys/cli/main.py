from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from amazon_recsys.api.app import create_app
from amazon_recsys.config.container import build_container
from amazon_recsys.config.settings import AppSettings, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amazon recommender production CLI.")
    parser.add_argument("--workspace-root", default=None, help="Override the workspace root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "evaluate", "export-bundle"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-name", default=None)
        command.add_argument("--run-profile", default=None)
        command.add_argument("--force-rebuild", action="store_true")

    export_command = subparsers.choices["export-bundle"]
    export_command.add_argument("--version", default=None)
    export_command.add_argument("--activate", action="store_true")

    activate = subparsers.add_parser("activate-bundle")
    activate.add_argument("version")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    return parser


def settings_from_args(args: argparse.Namespace) -> AppSettings:
    base = get_settings()
    updates = {}
    for field_name in ("workspace_root", "run_name", "run_profile", "host", "port"):
        value = getattr(args, field_name, None)
        if value is not None:
            updates[field_name] = Path(value) if field_name == "workspace_root" else value
    return base.model_copy(update=updates)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    container = build_container(settings)

    if args.command == "train":
        session = container.training_pipeline.run(force_rebuild=args.force_rebuild)
        print(json.dumps(session.evaluation_summary, indent=2, default=str))
        return 0

    if args.command == "evaluate":
        summary = container.training_pipeline.evaluate(force_rebuild=args.force_rebuild)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "export-bundle":
        session = container.training_pipeline.run(force_rebuild=args.force_rebuild)
        manifest = container.artifact_store.save_bundle(session, version=args.version)
        if args.activate:
            container.artifact_store.activate_bundle(manifest.version)
            container.recommendation_service.refresh()
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0

    if args.command == "activate-bundle":
        pointer = container.artifact_store.activate_bundle(args.version)
        container.recommendation_service.refresh()
        print(json.dumps(pointer.to_dict(), indent=2))
        return 0

    if args.command == "serve":
        app = create_app(settings)
        uvicorn.run(app, host=settings.host, port=settings.port, reload=False)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
