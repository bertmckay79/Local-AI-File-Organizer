"""Calibre outcomes flow into both Analyze and Quick Organize without moving files."""
import json
from unittest.mock import Mock

from core.calibre_bridge import CalibreBridge, CalibreResult
from database.db_manager import DatabaseManager
from ui.analyze_view import AnalyzeWorker
from ui.quick_organize_view import QuickOrganizeWorker


def _data(tmp_path):
    ebook = tmp_path / "my-ebook.epub"
    ebook.write_bytes(b"This file is only used with a mocked Calibre bridge")
    return {
        "file_path": str(ebook), "file_name": ebook.name,
        "extension": ".epub", "size_bytes": ebook.stat().st_size,
    }


def test_analyze_worker_persists_duplicate_and_omits_move_queue(
    tmp_path, monkeypatch
):
    db = DatabaseManager(tmp_path / "index.sqlite")
    book = _data(tmp_path)
    monkeypatch.setattr(
        CalibreBridge, "process",
        lambda bridge, path: CalibreResult(
            "duplicate", "Already in Calibre (ID 40)", 40, "My Ebook",
            "D:/Calibre Library",
        )
    )
    worker = AnalyzeWorker(
        [book], ["eBooks", "Needs Review"],
        Mock(), Mock(), Mock(), Mock(), db, use_ai=False,
        calibre_settings={"enabled": True, "library_path": "D:/Calibre Library"},
    )
    finished = []
    events = []
    worker.finished_analysis.connect(finished.append)
    worker.file_analyzed.connect(lambda *args: events.append(args))
    worker.run()

    assert finished == [{book["file_path"]: "Needs Review"}]
    assert events[0][4] == "Duplicate"
    saved = db.get_file_by_path(book["file_path"])
    assert saved["is_duplicate"] == 1
    assert saved["duplicate_of"] == "calibre:book:40"
    assert saved["category"] == "Needs Review"
    assert json.loads(saved["metadata_json"])["calibre"]["status"] == "duplicate"
    db.disconnect()


def test_quick_organize_worker_never_queues_calibre_imported_original(
    tmp_path, monkeypatch
):
    db = DatabaseManager(tmp_path / "index.sqlite")
    book = _data(tmp_path)
    monkeypatch.setattr(
        CalibreBridge, "process",
        lambda bridge, path: CalibreResult(
            "imported", "Imported to Calibre (ID 41)", 41, "My Ebook",
            "D:/Calibre Library",
        )
    )
    worker = QuickOrganizeWorker(
        scan_paths=[str(tmp_path)],
        config={"calibre": {"enabled": True}},
        ollama=Mock(), db=db, organizer=Mock(),
        metadata_extractor=Mock(), content_reader=Mock(), ocr=Mock(),
        use_ai=False,
    )
    messages = []
    worker.file_processed.connect(lambda *args: messages.append(args))
    results = {"categories": {}, "analyzed": 0}
    queue = worker._analyze_stage([book], results)

    assert queue == {}
    assert results["analyzed"] == 1
    assert results["categories"]["Needs Review"] == 1
    assert "Calibre imported" in messages[0][2]
    record = db.get_file_by_path(book["file_path"])
    assert record["category"] == "Needs Review"
    assert record["is_duplicate"] == 0
    assert json.loads(record["metadata_json"])["calibre"]["status"] == "imported"
    db.disconnect()
