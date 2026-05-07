# Documentation

[Back to main README](../README.md)

This folder holds the detailed project documentation that used to live in the top-level README.

## Guides

- [Architecture Guide](architecture.md)
- [Multistage Recsys Design](multistage-recsys-design.md)
- [Running The App](running-the-app.md)
- [Release Process](release-process.md)
- [MLflow Guide](mlflow.md)
- [Drift Monitoring Guide](monitoring.md)

## Suggested Reading Order

1. [Running The App](running-the-app.md)
2. [Architecture Guide](architecture.md)
3. [Multistage Recsys Design](multistage-recsys-design.md)
4. [MLflow Guide](mlflow.md)
5. [Release Process](release-process.md)
6. [Drift Monitoring Guide](monitoring.md)

## What Each File Covers

- `architecture.md`
  - package structure
  - ML architecture
  - notebook vs production workflow
  - Azure-first layout
- `multistage-recsys-design.md`
  - expanded Azure implementation design
  - two-stage candidate generation and ranking flow
  - Azure service mapping for the reference system diagram
  - rollout, security, latency, and cost notes
- `running-the-app.md`
  - exact local commands
  - beginner command walkthrough
  - local dev vs production-like mode
  - verification commands
- `release-process.md`
  - pull-request workflow for production changes
  - required CI checks
  - GitHub branch protection settings
  - rollback notes
- `mlflow.md`
  - local and production MLflow setup
  - exact commands
  - where to look in MLflow UI
- `monitoring.md`
  - drift monitoring design
  - outcome ingestion
  - monitoring commands
  - dashboard and API summary surface
- `examples/outcomes.example.csv`
  - starter delayed-outcomes file for the monitoring workflow
