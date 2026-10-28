"""ISBN/title fallback uses optional catalogue matching and suggests without moving."""
from unittest.mock import Mock

from core.book_identifier import (
    BookIdentity, clean_title, extract_isbn, identify_and_suggest,
    openlibrary_lookup, suggest_category, valid_isbn,
)

BOOKS = [
    {"name": "Maker & DIY", "extensions": [".pdf", ".epub"],
     "keywords": ["Book of Making", "Raspberry Pi", "HackSpace", "electronics"]},
    {"name": "Software - Security", "extensions": [".pdf"],
     "keywords": ["cryptography", "security"]},
    {"name": "Documents", "extensions": [".pdf"], "keywords": ["book"]},
]


def test_isbn_10_and_13_validation():
    assert valid_isbn("0-306-40615-2") == "0306406152"
    assert valid_isbn("978-0-306-40615-7") == "9780306406157"
    assert valid_isbn("978-0-306-40615-1") is None


def test_only_labelled_isbn_in_evidence():
    text = "Inventory 9780306406157 unrelated. ISBN-13: 978-0-306-40615-7."
    assert extract_isbn({}, text) == "9780306406157"
    assert extract_isbn({}, "Inventory 9780306406157") is None


def test_filename_title_removes_download_suffix():
    title = clean_title("Book_of_Making_2025_Z-Library.pdf", {}, None)
    assert title == "Book of Making 2025"


def test_offline_suggestion_matches_distinctive_title_without_network(monkeypatch):
    import core.book_identifier as module
    monkeypatch.setattr(module.requests, "get",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected HTTP")))
    identity, suggestion = identify_and_suggest(
        "Book_of_Making_2025_Z-Library.pdf", {}, None, BOOKS, ".pdf",
        external_lookup=False
    )
    assert identity.isbn is None
    assert suggestion.category == "Maker & DIY"


def test_ambiguous_title_does_not_invent_suggestion():
    result = suggest_category(BookIdentity("Unknown document"), BOOKS, ".pdf")
    assert result is None


def test_openlibrary_isbn_lookup_validates_record(monkeypatch):
    import core.book_identifier as module
    response = Mock()
    response.json.return_value = {
        "docs": [{"title": "Book of Making", "isbn": ["9780306406157"],
                  "subject": ["Electronics", "Arduino"], "author_name": ["A Writer"]}]
    }
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)
    value = openlibrary_lookup(BookIdentity("Bad file name", isbn="9780306406157"))
    assert value.title == "Book of Making"
    assert value.source == "Open Library ISBN"
    assert "Electronics" in value.subjects


def test_openlibrary_rejects_bad_title_match(monkeypatch):
    import core.book_identifier as module
    response = Mock()
    response.json.return_value = {"docs": [{
        "title": "Totally Different Book", "subject": ["Poetry"]
    }]}
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)
    value = openlibrary_lookup(BookIdentity("Book of Making"))
    assert value.subjects == ()
    assert value.source == "local"


def test_isbn_extraction_stops_at_a_newline_before_year():
    source = "ISBN-13: 978-0-306-40615-7\n2023 First printing"
    assert extract_isbn({}, source) == "9780306406157"


def test_isbn_10_can_be_followed_by_year():
    source = "ISBN-10: 0-306-40615-2 2023"
    assert extract_isbn({}, source) == "0306406152"
