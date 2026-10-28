"""Stable per-user storage and one-time migration from source-tree installations."""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

APP_FOLDER = "LocalAIFileOrganizer"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """Resolve the same user-data folder regardless of current working directory."""
    override = os.environ.get("LOCAL_AI_FILE_ORGANIZER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if not base:
            base = str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_FOLDER
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".local" / "share") / APP_FOLDER


def app_data_path(*parts: str) -> Path:
    path = get_data_dir()
    for part in parts:
        path = path / part
    return path


def resolve_data_path(value: str | os.PathLike, default: str) -> Path:
    """Respect explicitly configured absolute paths; resolve relative ones in user data."""
    raw = Path(value or default).expanduser()
    return raw if raw.is_absolute() else get_data_dir() / raw


def _legacy_candidates(relative_path: str | Path) -> list[Path]:
    source = Path(relative_path)
    candidates = []
    locations = [PROJECT_ROOT, Path.cwd()]
    if getattr(sys, "frozen", False):
        locations.append(Path(sys.executable).resolve().parent)
    for base in locations:
        candidate = base / source
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _staging_file(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".migration-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    return Path(path)


def migrate_legacy_file(target: Path, sources: list[Path], sqlite_db: bool = False) -> bool:
    """Copy the first legacy file into data dir if the target does not exist.

    Never moves/deletes source files, replaces an existing target or imports
    a different source over an existing user database. SQLite is copied via
    backup() so WAL transactions aren't lost.
    """
    if target.exists():
        return False
    for source in sources:
        if not source.is_file() or source.resolve() == target.resolve():
            continue
        staged = _staging_file(target)
        try:
            if sqlite_db:
                # Connection context managers do not close SQLite connections.
                # Explicit closure is essential before replacing a file on Windows.
                with closing(sqlite3.connect(str(source))) as previous:
                    with closing(sqlite3.connect(str(staged))) as new:
                        previous.backup(new)
            else:
                shutil.copy2(source, staged)
            # Recheck prior to installing: don't overwrite existing data.
            if target.exists():
                return False
            os.replace(staged, target)
            logger.info("Imported legacy application data from %s to %s", source, target)
            return True
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cannot migrate %s to %s: %s", source, target, exc)
        finally:
            staged.unlink(missing_ok=True)
    return False


def migrate_legacy_config(config_file: Path | None = None) -> bool:
    dest = config_file or app_data_path("user_config.json")
    sources = [
        PROJECT_ROOT / "config" / "user_config.json",
        Path.cwd() / "config" / "user_config.json",
    ]
    if getattr(sys, "frozen", False):
        sources.append(Path(sys.executable).resolve().parent / "config" / "user_config.json")
    return migrate_legacy_file(dest, sources)


def resolve_database_path(value: str | os.PathLike | None = None) -> Path:
    """Get a durable DB location and import the existing SQLite file once."""
    raw = Path(value or "file_organizer.db").expanduser()
    if raw.is_absolute():
        # Explicit absolute path is a user choice; don't silently relocate it.
        return raw
    if ".." in raw.parts:
        raise ValueError("Relative database path cannot leave the application data directory")
    destination = app_data_path(str(raw))
    destination.parent.mkdir(parents=True, exist_ok=True)
    migrate_legacy_file(destination, _legacy_candidates(raw), sqlite_db=True)
    return destination
