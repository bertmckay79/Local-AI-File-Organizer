"""Persist config and SQLite in user data; import old installs safely."""
import json
import os
import sqlite3
from pathlib import Path

import pytest

from utils import storage
from utils.config import ConfigManager


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    destination = tmp_path / "user_data"
    monkeypatch.setenv("LOCAL_AI_FILE_ORGANIZER_DATA_DIR", str(destination))
    return destination


def test_settings_survive_new_manager_and_changed_cwd(data_dir, monkeypatch, tmp_path):
    manager = ConfigManager()
    manager.set("ollama", "model", "qwen3.5:4b")
    manager.set("scan", "ignore_extensions", [".zip", ".rar"])
    manager.save()
    other = tmp_path / "different-launch-folder"
    other.mkdir()
    monkeypatch.chdir(other)
    reopened = ConfigManager()
    assert reopened.get("ollama", "model") == "qwen3.5:4b"
    assert reopened.get("scan", "ignore_extensions") == [".zip", ".rar"]
    assert reopened.config_path == data_dir / "user_config.json"


def test_import_old_user_settings_without_overwriting_existing(data_dir, monkeypatch, tmp_path):
    old = tmp_path / "legacy_config.json"
    old.write_text('{"ollama":{"model":"old-model"}}', encoding="utf-8")
    monkeypatch.setattr(storage, "_legacy_candidates", lambda raw: [old])
    assert storage.migrate_legacy_file(data_dir / "user_config.json", [old])
    assert ConfigManager().get("ollama", "model") == "old-model"
    old.write_text('{"ollama":{"model":"newer-old-value"}}', encoding="utf-8")
    assert ConfigManager().get("ollama", "model") == "old-model"


def test_sqlite_database_survives_restart_and_import(data_dir, tmp_path, monkeypatch):
    old = tmp_path / "file_organizer.db"
    with sqlite3.connect(old) as conn:
        conn.execute("CREATE TABLE custom_categories (name TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO custom_categories VALUES ('RPG / Pathfinder')")
    monkeypatch.setattr(storage, "_legacy_candidates", lambda relative: [old])
    destination = storage.resolve_database_path("file_organizer.db")
    assert destination == data_dir / "file_organizer.db"
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT name FROM custom_categories").fetchone()[0] == "RPG / Pathfinder"
    # Reopen doesn't wipe database even if legacy source has changed.
    with sqlite3.connect(old) as conn:
        conn.execute("DELETE FROM custom_categories")
    with sqlite3.connect(storage.resolve_database_path("file_organizer.db")) as conn:
        assert conn.execute("SELECT count(*) FROM custom_categories").fetchone()[0] == 1


def test_absolute_database_path_remains_user_selected(data_dir, tmp_path):
    custom = tmp_path / "mydb.sqlite"
    assert storage.resolve_database_path(str(custom)) == custom
    assert not (data_dir / "mydb.sqlite").exists()


def test_reset_only_deletes_settings_and_keeps_shared_dictionary(data_dir):
    config = ConfigManager()
    original = config.config
    config.set("ui", "theme", "light")
    config.save()
    config.reset()
    assert original is config.config
    assert config.get("ui", "theme") == "dark"
    assert (data_dir / "user_config.json").exists()


def test_user_config_saved_as_valid_json_with_no_temporary_files(data_dir):
    manager = ConfigManager()
    manager.save()
    assert json.loads((data_dir / "user_config.json").read_text(encoding="utf-8"))
    assert not list(data_dir.glob(".user_config-*"))


def test_database_relative_path_rejects_traversal(data_dir):
    with pytest.raises(ValueError):
        storage.resolve_database_path("../outside.db")


def test_storage_uses_localappdata_on_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCAL_AI_FILE_ORGANIZER_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(storage.sys, "platform", "win32")
    assert storage.get_data_dir() == tmp_path / "LocalAIFileOrganizer"



def test_new_install_creates_writable_database_directory(data_dir):
    from database.db_manager import DatabaseManager
    path = storage.resolve_database_path()
    db = DatabaseManager(path)
    assert path == data_dir / "file_organizer.db"
    assert path.exists()
    assert db.get_all_files() == []
    db.disconnect()


def test_migrate_real_sqlite_database_and_reopen_with_existing_categories(
    data_dir, tmp_path, monkeypatch
):
    from database.db_manager import DatabaseManager
    legacy = tmp_path / "old_install" / "file_organizer.db"
    legacy.parent.mkdir()
    db = DatabaseManager(legacy)
    db.add_custom_category(
        "Crafts / Model Painting", [".pdf"], ["minis", "painting"]
    )
    db.upsert_file({
        "file_path": "D:/Books/mini-painting.pdf",
        "file_name": "mini-painting.pdf", "extension": ".pdf",
        "category": "RPG / Game Master Resources", "size_bytes": 400,
    })
    db.disconnect()

    monkeypatch.setattr(storage, "_legacy_candidates", lambda relative: [legacy])
    moved_path = storage.resolve_database_path()
    assert moved_path == data_dir / "file_organizer.db"
    reopened = DatabaseManager(moved_path)
    assert reopened.get_custom_categories()[0]["name"] == "Crafts / Model Painting"
    assert reopened.get_file_by_path("D:/Books/mini-painting.pdf")["size_bytes"] == 400
    reopened.disconnect()

    # A second startup must not replace migrated data with old source values.
    old = DatabaseManager(legacy)
    old.delete_custom_category("Crafts / Model Painting")
    old.disconnect()
    again = DatabaseManager(storage.resolve_database_path())
    assert again.get_custom_categories()[0]["name"] == "Crafts / Model Painting"
    again.disconnect()



def test_scan_path_and_active_tab_settings_survive_restart(data_dir):
    before = ConfigManager()
    before.set("scan", "recent_paths", ["D:/Books", "E:/Models"])
    before.set("organize", "quick_scan_paths", ["D:/Books"])
    before.set("ui", "active_tab", 3)
    before.save()
    after = ConfigManager()
    assert after.get("scan", "recent_paths") == ["D:/Books", "E:/Models"]
    assert after.get("organize", "quick_scan_paths") == ["D:/Books"]
    assert after.get("ui", "active_tab") == 3
