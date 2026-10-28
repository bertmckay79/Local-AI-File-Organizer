"""Calibre integration is opt-in, ebook-only and never auto-moves duplicates."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.book_identifier import BookIdentity
from core.calibre_bridge import (
    CalibreBridge, CalibreResult, EBOOK_EXTENSIONS, _identity_from_opf,
    apply_calibre_result,
)

ISBN = "9780306406157"


def _bridge(monkeypatch, records, identity=None, **kwargs):
    settings = {
        "enabled": True, "library_path": "/fake/calibre-library",
        "import_missing": False, "fetch_metadata": False, **kwargs
    }
    bridge = CalibreBridge(settings)
    monkeypatch.setattr(bridge, "_validate", lambda: None)
    monkeypatch.setattr(bridge, "_identify", lambda path: (
        identity or BookIdentity(title="The Test Book", isbn=ISBN, author="Jane Doe")
    ))
    bridge._records = records
    monkeypatch.setattr(bridge, "_list", lambda refresh=False: bridge._records)
    return bridge


def _book(tmp_path, ext=".epub"):
    path = tmp_path / ("an-ebook" + ext)
    path.write_bytes(b"Fake ebook used only with mocked CLI")
    return path


@pytest.mark.parametrize("extension", [".epub", ".mobi", ".azw3", ".fb2", ".prc"])
def test_ebook_formats_are_supported(extension):
    assert extension in EBOOK_EXTENSIONS


@pytest.mark.parametrize("extension", [".pdf", ".zip", ".cbz", ".txt", ".docx"])
def test_non_ebook_formats_never_contact_calibre(monkeypatch, tmp_path, extension):
    bridge = CalibreBridge({"enabled": True, "library_path": "not-configured"})
    monkeypatch.setattr(bridge, "_validate", lambda: pytest.fail("Unexpected Calibre check"))
    assert bridge.process(_book(tmp_path, extension)).status == "skipped"


def test_disabled_calibre_does_not_call_cli(monkeypatch, tmp_path):
    bridge = CalibreBridge({"enabled": False})
    monkeypatch.setattr(bridge, "_validate", lambda: pytest.fail("Unexpected Calibre check"))
    assert bridge.process(_book(tmp_path)).status == "skipped"


def test_isbn_match_flags_duplicate_at_another_path(monkeypatch, tmp_path):
    bridge = _bridge(monkeypatch, [{
        "id": 42, "title": "Different Edition", "authors": "Another Author",
        "isbn": ISBN, "identifiers": {}, "formats": "EPUB",
    }])
    result = bridge.process(_book(tmp_path))
    assert result.status == "duplicate"
    assert result.book_id == 42
    assert "Already in Calibre" in result.message
    info = {"file_path": "D:/Downloads/my-book.epub", "metadata_json": '{"publisher": "X"}'}
    apply_calibre_result(info, result)
    assert info["is_duplicate"] == 1
    assert info["duplicate_of"] == "calibre:book:42"
    assert json.loads(info["metadata_json"])["calibre"]["status"] == "duplicate"
    assert json.loads(info["metadata_json"])["publisher"] == "X"


def test_exact_title_and_author_matches_alternate_ebook_format(monkeypatch, tmp_path):
    bridge = _bridge(
        monkeypatch, [{
            "id": 78, "title": "The Test Book", "authors": ["Jane Doe"],
            "isbn": "", "formats": "MOBI"
        }],
        identity=BookIdentity("The Test Book", None, "Jane Doe"),
    )
    assert bridge.process(_book(tmp_path, ".azw3")).status == "duplicate"


def test_title_only_match_is_ambiguous_and_not_imported(monkeypatch, tmp_path):
    bridge = _bridge(
        monkeypatch, [{"id": 8, "title": "The Test Book", "authors": "Another Writer"}],
        identity=BookIdentity("The Test Book", None, None),
        import_missing=True,
    )
    monkeypatch.setattr(bridge, "_db", lambda *args: pytest.fail("Import should not be called"))
    response = bridge.process(_book(tmp_path))
    assert response.status == "review"
    assert "Possible matching" in response.message


def test_missing_ebook_imports_and_fetches_metadata(monkeypatch, tmp_path):
    bridge = _bridge(monkeypatch, [], import_missing=True, fetch_metadata=True)
    calls = []
    monkeypatch.setattr(bridge, "_db", lambda *args: (
        calls.append(args) or "Added book ids: 133"
    ))
    fetched = []
    monkeypatch.setattr(bridge, "_fetch_and_set_metadata",
                        lambda book_id, book: (fetched.append((book_id, book.title)) or
                                               "Online metadata downloaded to Calibre"))
    result = bridge.process(_book(tmp_path))
    assert result.status == "imported"
    assert result.book_id == 133
    assert fetched == [(133, "The Test Book")]
    assert calls[0][0] == "add"
    assert "--isbn" in calls[0]
    assert calls[0][-1].endswith(".epub")


def test_import_disabled_leaves_missing_ebook_untouched(monkeypatch, tmp_path):
    bridge = _bridge(monkeypatch, [])
    monkeypatch.setattr(bridge, "_db", lambda *args: pytest.fail("No import requested"))
    result = bridge.process(_book(tmp_path))
    assert result.status == "review"
    assert "disabled" in result.message


def test_import_error_never_reports_success(monkeypatch, tmp_path):
    bridge = _bridge(monkeypatch, [], import_missing=True)
    def failure(*args):
        raise RuntimeError("Calibre library is locked")
    monkeypatch.setattr(bridge, "_db", failure)
    result = bridge.process(_book(tmp_path))
    assert result.status == "error"
    assert "locked" in result.message


def test_read_ebook_metadata_from_calibre_opf():
    opf = ("""<package xmlns:dc="http://purl.org/dc/elements/1.1/">"""
           """<metadata><dc:title>The Test Book</dc:title>"""
           """<dc:creator>Jane Doe</dc:creator>"""
           f"""<dc:identifier>{ISBN}</dc:identifier></metadata></package>""")
    book = _identity_from_opf(opf, "Fallback Title")
    assert book.title == "The Test Book"
    assert book.author == "Jane Doe"
    assert book.isbn == ISBN


def test_calibre_commands_are_list_arguments_not_shell(monkeypatch, tmp_path):
    from core import calibre_bridge as module
    library = tmp_path / "Calibre Library"
    library.mkdir()
    (library / "metadata.db").write_bytes(b"")
    bridge = CalibreBridge({"enabled": True, "library_path": str(library)})
    monkeypatch.setattr(bridge, "_executable", lambda tool: "/fake/" + tool)
    proc = Mock()
    proc.returncode = 0
    proc.stdout = "[]"
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: (
        proc if isinstance(args, list) and kwargs["stdin"] == module.subprocess.DEVNULL
        else pytest.fail("Unsafe subprocess invocation")
    ))
    assert bridge._list() == []


def test_metadata_title_mismatch_does_not_modify_calibre(monkeypatch):
    bridge = CalibreBridge({"enabled": True, "library_path": "/fake"})
    opf = ('<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata>'
           '<dc:title>Unrelated Novel</dc:title></metadata></package>')
    monkeypatch.setattr(bridge, "_executable", lambda name: name)
    monkeypatch.setattr(bridge, "_run", lambda name, *args: opf)
    monkeypatch.setattr(bridge, "_db", lambda *args: pytest.fail("Should not update metadata"))
    outcome = bridge._fetch_and_set_metadata(
        8, BookIdentity(title="The Test Book", isbn=None)
    )
    assert "title mismatch" in outcome


def test_ebook_inside_calibre_library_is_skipped(monkeypatch, tmp_path):
    library = tmp_path / "Calibre"
    library.mkdir()
    path = _book(library)
    bridge = CalibreBridge({"enabled": True, "library_path": str(library)})
    monkeypatch.setattr(bridge, "_validate", lambda: None)
    monkeypatch.setattr(bridge, "_identify", lambda p: pytest.fail("Should not inspect"))
    assert bridge.process(path).status == "review"


def test_fresh_import_is_not_marked_duplicate():
    record = {"metadata_json": None}
    apply_calibre_result(record, CalibreResult(
        "imported", "Added", 25, "A Book", "C:/Calibre Library",
    ))
    assert record["is_duplicate"] == 0
    assert record["duplicate_of"] is None
    assert json.loads(record["metadata_json"])["calibre"]["book_id"] == 25



def test_organizer_refuses_to_move_duplicate_calibre_ebook(tmp_path):
    from database.db_manager import DatabaseManager
    from database.operations import OperationHistory
    from core.organizer import FileOrganizer

    db = DatabaseManager(tmp_path / "index.sqlite")
    organizer = FileOrganizer(
        db, OperationHistory(db),
        {"organize": {"output_base": str(tmp_path / "organised")}}
    )
    ebook = _book(tmp_path)
    file_data = {
        "file_path": str(ebook), "file_name": ebook.name,
        "extension": ".epub", "category": "RPG",
    }
    apply_calibre_result(file_data, CalibreResult(
        "duplicate", "Calibre has the book", 17, "My Book", str(tmp_path / "library")
    ))
    db.upsert_file(file_data)
    success, reason, operation = organizer.move_file(str(ebook), "RPG")
    assert not success
    assert "Calibre duplicate" in reason
    assert ebook.is_file()
    assert operation is None
    db.disconnect()


def test_organizer_refuses_to_move_imported_calibre_ebook(tmp_path):
    from database.db_manager import DatabaseManager
    from database.operations import OperationHistory
    from core.organizer import FileOrganizer

    db = DatabaseManager(tmp_path / "index.sqlite")
    organizer = FileOrganizer(
        db, OperationHistory(db),
        {"organize": {"output_base": str(tmp_path / "organised")}}
    )
    ebook = _book(tmp_path, ".mobi")
    file_data = {
        "file_path": str(ebook), "file_name": ebook.name,
        "extension": ".mobi", "category": "eBooks",
    }
    apply_calibre_result(file_data, CalibreResult(
        "imported", "Book imported", 18, "The Test Book", str(tmp_path / "library")
    ))
    db.upsert_file(file_data)
    success, reason, operation = organizer.move_file(str(ebook), "eBooks")
    assert not success
    assert "Calibre-managed" in reason
    assert ebook.is_file()
    assert operation is None
    db.disconnect()



def test_calibre_library_source_not_moved_even_if_manually_assigned(tmp_path):
    from database.db_manager import DatabaseManager
    from database.operations import OperationHistory
    from core.organizer import FileOrganizer

    library = tmp_path / "Calibre"
    library.mkdir()
    ebook = _book(library)
    db = DatabaseManager(tmp_path / "cache.sqlite")
    info = {
        "file_path": str(ebook), "file_name": ebook.name,
        "extension": ".epub", "category": "eBooks",
    }
    apply_calibre_result(info, CalibreResult(
        "review", "Ebook is inside the managed Calibre library",
        library=str(library)
    ))
    db.upsert_file(info)
    organizer = FileOrganizer(
        db, OperationHistory(db),
        {"organize": {"output_base": str(tmp_path / "out")}}
    )
    ok, message, operation = organizer.move_file(str(ebook), "eBooks")
    assert not ok
    assert "inside Calibre library" in message
    assert ebook.is_file()
    assert operation is None
    db.disconnect()



def test_metadata_isbn_mismatch_and_different_title_not_applied(monkeypatch):
    bridge = CalibreBridge({"enabled": True})
    opf = ('<package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata>'
           '<dc:title>Different Book</dc:title>'
           '<dc:identifier>0306406152</dc:identifier>'
           '</metadata></package>')
    monkeypatch.setattr(bridge, "_executable", lambda program: program)
    monkeypatch.setattr(bridge, "_run", lambda name, *args: opf)
    monkeypatch.setattr(bridge, "_db", lambda *args: pytest.fail("Unsafe metadata update"))
    assert "mismatch" in bridge._fetch_and_set_metadata(
        8, BookIdentity("The Test Book", isbn="9780306406157")
    )
