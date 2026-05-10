from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import uvicorn

from amazon_recsys.api.app import create_app
from amazon_recsys.bootstrap import build_container
from amazon_recsys.domain.entities import EvaluationSummary
from amazon_recsys.config.settings import AppSettings, get_settings


LOGGER = logging.getLogger(__name__)


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

    ingest = subparsers.add_parser("ingest-outcomes")
    ingest.add_argument("--source", required=True)

    simulate = subparsers.add_parser("simulate-outcomes")
    simulate.add_argument("--bundle-version", default="active")
    simulate.add_argument("--window-start", default=None)
    simulate.add_argument("--window-end", default=None)
    simulate.add_argument("--days", type=int, default=1)
    simulate.add_argument("--delay-minutes", type=int, default=60)

    monitor = subparsers.add_parser("monitor-drift")
    monitor.add_argument("--window-start", required=True)
    monitor.add_argument("--window-end", required=True)
    monitor.add_argument("--bundle-version", default="active")
    monitor.add_argument("--simulate-outcomes", action="store_true")

    backfill = subparsers.add_parser("monitor-backfill")
    backfill.add_argument("--days", type=int, required=True)
    backfill.add_argument("--bundle-version", default="active")
    backfill.add_argument("--simulate-outcomes", action="store_true")

    diagnose = subparsers.add_parser("diagnose-candidates")
    diagnose.add_argument("--bundle-version", default="active")
    diagnose.add_argument("--split", choices=["val", "test"], default="test")
    diagnose.add_argument("--sample-size", type=int, default=500)
    diagnose.add_argument("--persist", action="store_true")

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


def _json_payload(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    container = build_container(settings)

    if args.command == "train":
        session = container.training_pipeline.run(force_rebuild=args.force_rebuild)
        summary = EvaluationSummary.from_dict(session.evaluation_summary.to_dict())
        if session.mlflow is not None:
            summary.mlflow = session.mlflow
        print(json.dumps(summary.to_dict(), indent=2, default=str))
        return 0

    if args.command == "evaluate":
        summary = container.training_pipeline.evaluate(force_rebuild=args.force_rebuild)
        print(json.dumps(summary.to_dict(), indent=2, default=str))
        return 0

    if args.command == "export-bundle":
        LOGGER.info(
            "CLI export-bundle requested: run_name=%s run_profile=%s version=%s activate=%s",
            settings.run_name,
            settings.run_profile,
            args.version or "<auto>",
            args.activate,
        )
        session = container.training_pipeline.run(force_rebuild=args.force_rebuild)
        manifest = container.bundle_export_service.export_bundle(session, version=args.version)
        if args.activate:
            LOGGER.info("Activating exported bundle: version=%s", manifest.version)
            container.artifact_store.activate_bundle(manifest.version)
            container.recommendation_service.refresh()
            LOGGER.info("Bundle activation complete: version=%s", manifest.version)
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0

    if args.command == "activate-bundle":
        LOGGER.info("Activating bundle: version=%s", args.version)
        pointer = container.artifact_store.activate_bundle(args.version)
        container.recommendation_service.refresh()
        LOGGER.info("Bundle activation complete: version=%s", args.version)
        print(json.dumps(pointer.to_dict(), indent=2))
        return 0

    if args.command == "ingest-outcomes":
        payload = container.monitoring_service.ingest_outcomes(Path(args.source))
        print(json.dumps(_json_payload(payload), indent=2, default=str))
        return 0

    if args.command == "simulate-outcomes":
        payload = container.monitoring_service.simulate_outcomes(
            bundle_version=args.bundle_version,
            window_start=args.window_start,
            window_end=args.window_end,
            days=args.days,
            delay_minutes=args.delay_minutes,
        )
        print(json.dumps(_json_payload(payload), indent=2, default=str))
        return 0

    if args.command == "monitor-drift":
        summary = container.monitoring_service.run_monitoring(
            window_start=args.window_start,
            window_end=args.window_end,
            bundle_version=args.bundle_version,
            simulate_outcomes=args.simulate_outcomes,
        )
        print(json.dumps(summary.to_dict(), indent=2, default=str))
        return 0

    if args.command == "monitor-backfill":
        summaries = container.monitoring_service.monitor_backfill(
            days=args.days,
            bundle_version=args.bundle_version,
            simulate_outcomes=args.simulate_outcomes,
        )
        print(json.dumps([summary.to_dict() for summary in summaries], indent=2, default=str))
        return 0

    if args.command == "diagnose-candidates":
        payload = container.monitoring_service.run_candidate_diagnostics(
            bundle_version=args.bundle_version,
            split=args.split,
            sample_size=args.sample_size,
            persist=args.persist,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "serve":
        app = create_app(settings)
        uvicorn.run(app, host=settings.host, port=settings.port, reload=False)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
