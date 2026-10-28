"""
Configuration management for Local AI File Organizer.

Default settings are version-controlled, while the user's actual settings
live in a stable writable data directory, not in the checkout or EXE folder.
"""
from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

from utils.storage import app_data_path, migrate_legacy_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default_config.json"
# Backwards-compatible public constant. ConfigManager resolves its own path
# at construction time so a test or portable install can override data dir.
USER_CONFIG_PATH = app_data_path("user_config.json")


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class ConfigManager:
    """Manages persistent configuration with versioned defaults and atomic writes."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = Path(config_path) if config_path is not None else app_data_path("user_config.json")
        if config_path is None:
            migrate_legacy_config(self.config_path)
        self._config: dict = {}
        self.load()

    def load(self) -> dict:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            defaults = json.load(f)
        user_config = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            if not isinstance(user_config, dict):
                raise ValueError(f"Settings must be a JSON object: {self.config_path}")

        # Maintain one shared dictionary reference. Views hold self.config
        # and should see reloads/reset immediately.
        merged = _deep_merge(defaults, user_config)
        self._config.clear()
        self._config.update(merged)
        return self._config

    def save(self, config: dict | None = None) -> None:
        """Write configuration atomically. A failed save retains the previous file."""
        candidate = config if config is not None else self._config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(candidate, indent=2, ensure_ascii=False)
        fd, temporary = tempfile.mkstemp(
            prefix=".user_config-", suffix=".tmp", dir=self.config_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.config_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if candidate is not self._config:
            self._config.clear()
            self._config.update(deepcopy(candidate))

    def reset(self) -> dict:
        """Reset settings using the same persistent path (without losing DB data)."""
        self.config_path.unlink(missing_ok=True)
        return self.load()

    def get(self, *keys, default=None):
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys_and_value):
        if len(keys_and_value) < 2:
            raise ValueError("set requires at least one key and a value")
        *keys, value = keys_and_value
        target = self._config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    @property
    def config(self) -> dict:
        return self._config

    @property
    def ollama_settings(self) -> dict:
        return self._config.get("ollama", {})

    @property
    def scan_settings(self) -> dict:
        return self._config.get("scan", {})

    @property
    def organize_settings(self) -> dict:
        return self._config.get("organize", {})

    @property
    def ui_settings(self) -> dict:
        return self._config.get("ui", {})

    @property
    def db_settings(self) -> dict:
        return self._config.get("database", {})

    @property
    def logging_settings(self) -> dict:
        return self._config.get("logging", {})

    @property
    def ocr_settings(self) -> dict:
        return self._config.get("ocr", {})

    @property
    def advanced_settings(self) -> dict:
        return self._config.get("advanced", {})
