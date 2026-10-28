"""Regression tests for PDF/EPUB extraction and safe classification."""
import json
import zipfile
from unittest.mock import Mock

import pytest

from core.document_classifier import (
    DocumentClassificationService, NEEDS_REVIEW,
    extract_pdf_evidence, extract_epub_evidence, load_category_definitions
)

CATS = [
    {"name": "Documents", "keywords": []},
    {"name": "eBooks", "keywords": []},
    {"name": "Software - Vue.js", "keywords": ["Pinia", "Composition API"]},
    {"name": "RPG - Dungeons & Dragons", "keywords": ["Player's Handbook"]},
    {"name": "Miscellaneous", "keywords": []},
]


def classifier(response, extension=".pdf", content=None):
    ai = Mock()
    ai.classify_file.return_value = response
    reader = Mock()
    metadata = Mock()
    metadata.extract.return_value = {}
    ocr = Mock()
    ocr.is_available.return_value = False
    service = DocumentClassificationService(ai, reader, metadata, ocr, CATS)
    return service, {"file_path": "missing" + extension, "file_name": "missing" + extension,
                     "extension": extension}


def test_pdf_content_category(monkeypatch):
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence", lambda p: "Vue 3 Pinia")
    service, file = classifier("Software - Vue.js")
    result = service.classify(file)
    assert result.category == "Software - Vue.js" and not result.requires_review
    assert service.ollama.classify_file.call_args.kwargs["category_details"] == CATS


@pytest.mark.parametrize("category", ["Documents", "eBooks", "Miscellaneous", None])
def test_generic_result_needs_review(monkeypatch, category):
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence", lambda p: "Vue 3")
    service, file = classifier(category)
    assert service.classify(file).category == NEEDS_REVIEW


def test_unreadable_pdf_needs_review():
    service, file = classifier("RPG - Dungeons & Dragons")
    assert service.classify(file).category == NEEDS_REVIEW


def test_ollama_exception_needs_review(monkeypatch):
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence", lambda p: "D&D rules")
    service, file = classifier(None)
    service.ollama.classify_file.side_effect = ConnectionError("unavailable")
    assert service.classify(file).category == NEEDS_REVIEW


def test_load_categories_includes_keywords_and_deduplicates(tmp_path):
    config = tmp_path / "categories.json"
    config.write_text(json.dumps({"categories": CATS}), encoding="utf-8")
    db = Mock()
    db.get_custom_categories.return_value = [{"name": "software - vue.js"}, {"name": "Game Design"}]
    result = load_category_definitions(config, db)
    assert len([x for x in result if x["name"].casefold() == "software - vue.js"]) == 1
    assert next(x for x in result if x["name"] == "Software - Vue.js")["keywords"] == ["Pinia", "Composition API"]
    assert any(x["name"] == NEEDS_REVIEW for x in result)


def test_pdf_evidence_samples_later_pages(tmp_path):
    fitz = pytest.importorskip("fitz")
    file = tmp_path / "book.pdf"
    with fitz.open() as doc:
        for i in range(12):
            page = doc.new_page()
            page.insert_text((72, 72), "Pinia Composition API" if i == 6 else f"Page {i}")
        doc.save(str(file))
    assert "Pinia Composition API" in extract_pdf_evidence(file)


def test_epub_spine_order_and_metadata(tmp_path):
    file = tmp_path / "book.epub"
    container = ('<container><rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>')
    opf = ('<package xmlns="http://www.idpf.org/2007/opf">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           '<dc:title>TypeScript Handbook</dc:title></metadata>'
           '<manifest><item id="one" href="one.xhtml"/>'
           '<item id="two" href="two.xhtml"/></manifest>'
           '<spine><itemref idref="two"/><itemref idref="one"/></spine></package>')
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OPS/package.opf", opf)
        archive.writestr("OPS/one.xhtml", "<html><body>Other text</body></html>")
        archive.writestr("OPS/two.xhtml", "<html><body>Type inference with generics</body></html>")
    text = extract_epub_evidence(file)
    assert "TypeScript Handbook" in text
    assert text.index("Type inference") < text.index("Other text")


def test_organizer_refuses_to_move_review_files():
    from core.organizer import FileOrganizer
    organizer = FileOrganizer.__new__(FileOrganizer)
    success, message, operation = organizer.move_file("does-not-exist.pdf", NEEDS_REVIEW)
    assert not success
    assert "not moved" in message
    assert operation is None


def test_ai_prompt_contains_category_keywords(monkeypatch):
    from core.ollama_client import OllamaClient
    client = OllamaClient()
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append((messages, kwargs))
        return json.dumps({"category": "Software - Vue.js"})

    monkeypatch.setattr(client, "chat", fake_chat)
    actual = client.classify_file(
        {"file_name": "vue.pdf", "extension": ".pdf"}, [c["name"] for c in CATS],
        "Vue Pinia", category_details=CATS
    )
    assert actual == "Software - Vue.js"
    assert "Pinia, Composition API" in calls[0][0][1]["content"]
    assert calls[0][1]["num_predict"] >= 512


def test_qwen35_classification_disables_thinking_and_forced_json(monkeypatch):
    from core.ollama_client import OllamaClient
    client = OllamaClient(model="qwen3.5:4b")
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs)
        return 'Answer: {"category": "RPG - Game Master Resources"}'

    monkeypatch.setattr(client, "chat", fake_chat)
    category = client.classify_file(
        {"file_name": "Adventure Design.pdf", "extension": ".pdf"},
        ["RPG - Game Master Resources", "Documents"],
        "How to design roleplaying adventures"
    )
    assert category == "RPG - Game Master Resources"
    assert calls[0]["think"] is False
    assert calls[0]["use_json_format"] is False
    assert calls[0]["num_predict"] >= 768


def test_qwen35_thinking_setting_is_top_level_payload(monkeypatch):
    from core.ollama_client import OllamaClient
    import core.ollama_client as module
    client = OllamaClient(model="qwen3.5:4b")
    post = Mock()
    post.return_value.status_code = 200
    post.return_value.json.return_value = {
        "message": {"content": '{"category": "RPG - Other Systems"}'}
    }
    monkeypatch.setattr(module.requests, "post", post)
    assert client.classify_file(
        {"file_name": "Roleplaying.pdf", "extension": ".pdf"},
        ["RPG - Other Systems"], "Basic Roleplaying rules"
    ) == "RPG - Other Systems"
    payload = post.call_args.kwargs["json"]
    assert payload["think"] is False
    assert "format" not in payload
    assert payload["options"]["num_predict"] >= 768
