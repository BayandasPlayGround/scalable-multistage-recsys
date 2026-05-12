from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import uvicorn

from amazon_recsys.api.app import create_app
from amazon_recsys.bootstrap import build_container
from amazon_recsys.domain.entities import BundleManifest, EvaluationSummary
from amazon_recsys.config.settings import DEFAULT_CATEGORIES, AppSettings, get_settings


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
    export_command.add_argument("--allow-non-prod-activation", action="store_true")

    activate = subparsers.add_parser("activate-bundle")
    activate.add_argument("version")
    activate.add_argument("--allow-non-prod-activation", action="store_true")

    report = subparsers.add_parser("artifact-report")
    report.add_argument("--json", action="store_true")
    report.add_argument("--limit", type=int, default=50)

    prune = subparsers.add_parser("prune-artifacts")
    prune.add_argument("--dry-run", action="store_true")
    prune.add_argument("--confirm", action="store_true")
    prune.add_argument("--keep-prod", type=int, default=2)
    prune.add_argument("--keep-debug", type=int, default=2)

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


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _dir_mtime(path: Path) -> float:
    try:
        return max((item.stat().st_mtime for item in path.rglob("*") if item.exists()), default=path.stat().st_mtime)
    except OSError:
        return 0.0


def _artifact_entries(settings: AppSettings) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    active_version: str | None = None
    active_path = settings.resolved_active_bundle_path
    if active_path.exists():
        try:
            payload = json.loads(active_path.read_text(encoding="utf-8"))
            active_version = str(payload.get("version") or "")
        except (OSError, json.JSONDecodeError):
            active_version = None

    run_root = settings.legacy_artifact_root.parent
    if run_root.exists():
        for path in sorted(item for item in run_root.iterdir() if item.is_dir()):
            entries.append(
                {
                    "type": "run",
                    "name": path.name,
                    "path": str(path),
                    "size_gb": round(_directory_size_bytes(path) / (1024 ** 3), 3),
                    "last_modified": _dir_mtime(path),
                    "active": False,
                }
            )
    bundle_root = settings.resolved_bundle_root
    if bundle_root.exists():
        for path in sorted(item for item in bundle_root.iterdir() if item.is_dir()):
            entries.append(
                {
                    "type": "bundle",
                    "name": path.name,
                    "path": str(path),
                    "size_gb": round(_directory_size_bytes(path) / (1024 ** 3), 3),
                    "last_modified": _dir_mtime(path),
                    "active": bool(active_version and path.name == active_version),
                }
            )
    return sorted(entries, key=lambda item: float(item["last_modified"]), reverse=True)


def _format_artifact_report(entries: list[dict[str, object]], *, limit: int) -> str:
    selected = entries[: max(1, int(limit))]
    total_gb = sum(float(item["size_gb"]) for item in entries)
    lines = [f"Artifact report: {len(entries)} directories, {total_gb:.2f} GB total"]
    for item in selected:
        active_marker = " active" if item.get("active") else ""
        lines.append(
            f"{item['type']:>6} {float(item['size_gb']):8.3f} GB {item['name']}{active_marker} :: {item['path']}"
        )
    return "\n".join(lines)


def _is_prod_artifact(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("prod-") or lowered.startswith("prod_")


def _prune_candidates(
    settings: AppSettings,
    *,
    keep_prod: int,
    keep_debug: int,
) -> list[dict[str, object]]:
    entries = _artifact_entries(settings)
    current_run = settings.run_name
    candidates: list[dict[str, object]] = []
    for artifact_type in {"run", "bundle"}:
        for prod_flag, keep in ((True, keep_prod), (False, keep_debug)):
            group = [
                item for item in entries
                if item["type"] == artifact_type and _is_prod_artifact(str(item["name"])) is prod_flag
            ]
            group = sorted(group, key=lambda item: float(item["last_modified"]), reverse=True)
            for item in group[max(0, int(keep)):]:
                if item.get("active"):
                    continue
                if item["type"] == "run" and item["name"] == current_run:
                    continue
                candidates.append(item)
    return sorted(candidates, key=lambda item: float(item["last_modified"]))


def _assert_child(path: Path, root: Path) -> None:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to prune path outside expected root: {resolved_path}") from exc


def _delete_prune_candidates(settings: AppSettings, candidates: list[dict[str, object]]) -> None:
    run_root = settings.legacy_artifact_root.parent
    bundle_root = settings.resolved_bundle_root
    for item in candidates:
        path = Path(str(item["path"]))
        root = bundle_root if item["type"] == "bundle" else run_root
        _assert_child(path, root)
        shutil.rmtree(path)


def _bundle_pipeline_categories(manifest: BundleManifest) -> list[str]:
    config_path = Path(manifest.bundle_dir) / "data" / "pipeline_config.json"
    if not config_path.exists():
        return []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return []
    return [str(item) for item in categories]


def _assert_activation_allowed(
    settings: AppSettings,
    manifest: BundleManifest,
    *,
    allow_non_prod_activation: bool,
) -> None:
    if settings.environment.lower() != "production" or allow_non_prod_activation:
        return
    reasons: list[str] = []
    if manifest.run_profile == "debug":
        reasons.append("run_profile=debug")
    categories = _bundle_pipeline_categories(manifest)
    if categories and len(categories) < len(DEFAULT_CATEGORIES):
        reasons.append(f"single/partial category bundle: {categories}")
    name = f"{manifest.version} {manifest.run_name}".lower()
    if "debug" in name:
        reasons.append("version/run_name contains 'debug'")
    if reasons:
        raise RuntimeError(
            "Production activation refused for non-prod-looking bundle "
            f"{manifest.version!r}: {', '.join(reasons)}. Re-run with "
            "--allow-non-prod-activation only if this is intentional."
        )


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
            _assert_activation_allowed(
                settings,
                manifest,
                allow_non_prod_activation=args.allow_non_prod_activation,
            )
            LOGGER.info("Activating exported bundle: version=%s", manifest.version)
            container.artifact_store.activate_bundle(manifest.version)
            container.recommendation_service.refresh()
            LOGGER.info("Bundle activation complete: version=%s", manifest.version)
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0

    if args.command == "activate-bundle":
        manifest = container.artifact_store.load_manifest(args.version)
        _assert_activation_allowed(
            settings,
            manifest,
            allow_non_prod_activation=args.allow_non_prod_activation,
        )
        LOGGER.info("Activating bundle: version=%s", args.version)
        pointer = container.artifact_store.activate_bundle(args.version)
        container.recommendation_service.refresh()
        LOGGER.info("Bundle activation complete: version=%s", args.version)
        print(json.dumps(pointer.to_dict(), indent=2))
        return 0

    if args.command == "artifact-report":
        entries = _artifact_entries(settings)
        if args.json:
            print(json.dumps(entries, indent=2, default=str))
        else:
            print(_format_artifact_report(entries, limit=args.limit))
        return 0

    if args.command == "prune-artifacts":
        if not args.dry_run and not args.confirm:
            parser.error("prune-artifacts requires --dry-run or --confirm.")
        candidates = _prune_candidates(settings, keep_prod=args.keep_prod, keep_debug=args.keep_debug)
        payload = {
            "dry_run": bool(args.dry_run),
            "candidates": candidates,
            "total_gb": round(sum(float(item["size_gb"]) for item in candidates), 3),
        }
        if not args.dry_run:
            _delete_prune_candidates(settings, candidates)
            payload["deleted"] = True
        print(json.dumps(payload, indent=2, default=str))
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
