"""Two-level categories retain existing file classifications safely."""
import pytest
from core.category_hierarchy import (
    split_category, normalise_category, category_fields,
    category_key, normalise_definition, full_category_from_record,
)


@pytest.mark.parametrize("original,expected", [
    ("RPG - Dungeons & Dragons", "RPG / Dungeons & Dragons"),
    ("Software - Web Development", "Software / Web Development"),
    ("Game Development - Unity", "Game Development / Unity"),
    ("Fiction - Fantasy", "Fiction / Fantasy"),
    ("Maker & DIY", "Maker & DIY"),
    ("RPG / Game Master Resources", "RPG / Game Master Resources"),
])
def test_legacy_labels_are_mapped_to_hierarchy(original, expected):
    assert normalise_category(original) == expected


def test_store_parent_and_subcategory_as_separate_columns():
    assert category_fields("RPG - Pathfinder") == {
        "category": "RPG", "subcategory": "Pathfinder"
    }
    assert full_category_from_record({
        "category": "RPG", "subcategory": "Pathfinder"
    }) == "RPG / Pathfinder"


def test_legacy_db_record_still_works():
    assert full_category_from_record({
        "category": "Software - Vue.js", "subcategory": None
    }) == "Software / Vue.js"


@pytest.mark.parametrize("bad", ["..", "RPG/../../outside", "A\\\\B", "CON", ""])
def test_invalid_names_rejected(bad):
    with pytest.raises(ValueError):
        category_key(bad)


def test_normalise_definition_explicit_parent():
    assert normalise_definition({
        "name": "Godot", "parent": "Game Development",
        "extensions": [".pdf"]
    })["name"] == "Game Development / Godot"
