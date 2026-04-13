from amazon_recsys.config.settings import AzureConfig


def aml_train_command(config: AzureConfig) -> str:
    return "python -m amazon_recsys.cli.main export-bundle --version aml-${RUN_ID}"


__all__ = ["AzureConfig", "aml_train_command"]
