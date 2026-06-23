import os
from pathlib import Path

import yaml
from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_config(config_path: str = "config.yaml") -> dict:
    path = PROJECT_ROOT / config_path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env_value(name: str, required: bool = False, default=None):
    values = dotenv_values(ENV_PATH)

    if name in values and values[name]:
        return values[name]

    load_dotenv(ENV_PATH, override=True)
    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(f"Missing required env value: {name}")

    return value
