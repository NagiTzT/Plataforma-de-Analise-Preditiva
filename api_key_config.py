"""Carrega chaves locais sem incluí-las no repositório Git."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def configured_keys(env_name: str, config_name: str) -> list[str]:
    raw = os.getenv(env_name, "").strip()
    if raw:
        keys = re.split(r"[,;\s]+", raw)
    else:
        config_path = Path(__file__).with_name("api_keys.local.json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            keys = config.get(config_name, [])
        except (OSError, ValueError, TypeError):
            keys = []
    if isinstance(keys, str):
        keys = re.split(r"[,;\s]+", keys)
    return list(dict.fromkeys(str(key).strip() for key in keys if str(key).strip()))
