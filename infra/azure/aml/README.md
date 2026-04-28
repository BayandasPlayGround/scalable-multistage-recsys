# Azure ML Scaffold

[Back to main README](../../../README.md)

This folder is a local-first placeholder for the batch training path.

Current contract:

- training happens through `python -m amazon_recsys.cli.main export-bundle`
- the bundle is the deployable serving unit
- online serving should load only the active bundle manifest

Planned Azure ML usage:

- build an environment from `environment.yml`
- run the batch job from `train-job.yml`
- publish the produced bundle artifact to the chosen registry or storage account
