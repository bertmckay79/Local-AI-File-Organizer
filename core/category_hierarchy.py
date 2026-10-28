"""Two-level category paths and backwards-compatible legacy names."""
from __future__ import annotations

from pathlib import Path

SEPARATOR = " / "
LEGACY_PARENTS = ("RPG", "Software", "Game Development", "Fiction")
REVIEW = "Needs Review"


def split_category(value: str) -> tuple[str, str | None]:
    """Interpret canonical and legacy labels without permitting nested paths."""
    value = str(value or "").strip()
    if SEPARATOR in value:
        parent, leaf = value.split(SEPARATOR, 1)
        return parent.strip(), leaf.strip() or None
    for parent in LEGACY_PARENTS:
        prefix = parent + " - "
        if value.startswith(prefix):
            return parent, value[len(prefix):].strip() or None
    return value, None


def validate_segment(value: str) -> str:
    segment = str(value or "").strip()
    if not segment or segment in (".", ".."):
        raise ValueError("Category or subcategory name is required")
    if len(segment) > 90 or any(ch in segment for ch in '/\\<>:"|?*\x00'):
        raise ValueError("Category names must be short and cannot contain path separators")
    if segment.endswith((".", " ")):
        raise ValueError("Invalid category folder name")
    if segment.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }:
        raise ValueError("Reserved Windows folder name")
    return segment


def category_key(parent: str, subcategory: str | None = None) -> str:
    top = validate_segment(parent)
    if subcategory:
        child = validate_segment(subcategory)
        if top.casefold() == child.casefold():
            raise ValueError("Category and subcategory cannot have the same name")
        return top + SEPARATOR + child
    return top


def normalise_category(value: str) -> str:
    parent, leaf = split_category(value)
    return category_key(parent, leaf)


def category_fields(value: str) -> dict[str, str | None]:
    parent, leaf = split_category(value)
    return {"category": parent, "subcategory": leaf}


def normalise_definition(item: dict) -> dict:
    result = dict(item)
    name = str(result.get("name") or "").strip()
    explicit_parent = str(result.get("parent") or "").strip()
    if explicit_parent:
        leaf = str(result.get("subcategory") or name).strip()
        top = explicit_parent
    else:
        top, leaf = split_category(name)
    result["name"] = category_key(top, leaf)
    result["parent"] = top
    result["subcategory"] = leaf
    return result


def full_category_from_record(record: dict) -> str:
    """Read both new (category/subcategory) and old DB category values."""
    top = str(record.get("category") or "").strip()
    sub = str(record.get("subcategory") or "").strip()
    if sub:
        return category_key(top, sub)
    return normalise_category(top) if top else ""
