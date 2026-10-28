"""Database migration for legacy flat labels and custom category names."""
from database.db_manager import DatabaseManager


def test_category_data_migration_preserves_existing_file_info(tmp_path):
    path = tmp_path / "files.sqlite"
    db = DatabaseManager(path)
    db.upsert_file({
        "file_path": "D:/Books/dd.pdf", "file_name": "dd.pdf",
        "extension": ".pdf", "size_bytes": 145,
        "category": "RPG - Dungeons & Dragons",
        "content_summary": "adventure rules",
    })
    db.disconnect()

    reopened = DatabaseManager(path)
    record = reopened.get_file_by_path("D:/Books/dd.pdf")
    assert record["category"] == "RPG"
    assert record["subcategory"] == "Dungeons & Dragons"
    assert record["content_summary"] == "adventure rules"
    assert record["size_bytes"] == 145

    reopened.init_schema()  # Idempotent startup migration
    assert reopened.get_file_by_path("D:/Books/dd.pdf")["subcategory"] == "Dungeons & Dragons"
    reopened.disconnect()


def test_custom_category_old_name_upgrades_without_duplicate(tmp_path):
    db = DatabaseManager(tmp_path / "custom.sqlite")
    db.conn.execute(
        """INSERT INTO custom_categories(name, extensions_json, keywords_json, created_at)
           VALUES (?, ?, ?, ?)""",
        ("Game Development - Godot", '[".pdf"]', '["GDScript"]', "2026-01-01"),
    )
    db.conn.commit()
    db.add_custom_category(
        "Game Development / Godot", [".pdf", ".epub"], ["Godot"]
    )
    categories = db.get_custom_categories()
    assert len(categories) == 1
    assert categories[0]["name"] == "Game Development / Godot"
    assert categories[0]["extensions"] == [".pdf", ".epub"]
    db.delete_custom_category("Game Development / Godot")
    assert db.get_custom_categories() == []
    db.disconnect()



def test_query_parent_or_specific_subcategory(tmp_path):
    db = DatabaseManager(tmp_path / "indexed.sqlite")
    for name, child in [("a.pdf", "Dungeons & Dragons"), ("b.pdf", "Pathfinder")]:
        db.upsert_file({
            "file_path": name, "file_name": name, "extension": ".pdf",
            "category": "RPG", "subcategory": child
        })
    assert len(db.get_files_by_category("RPG")) == 2
    assert [r["file_name"] for r in db.get_files_by_category(
        "RPG / Pathfinder"
    )] == ["b.pdf"]
    assert [r["file_name"] for r in db.get_files_by_category(
        "RPG - Dungeons & Dragons"
    )] == ["a.pdf"]
    db.disconnect()
