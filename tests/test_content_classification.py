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


def test_generic_book_ai_result_gets_subject_only_retry(monkeypatch):
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence", lambda p: "Game master advice")
    service, file = classifier("Documents")
    service.definitions = CATS + [{"name": "RPG - Game Master Resources", "keywords": ["GM"], "extensions": [".pdf", ".epub"]}]
    service.category_names = [item["name"] for item in service.definitions]
    service.ollama.classify_file.side_effect = ["Documents", "RPG - Game Master Resources"]
    result = service.classify(file)
    assert result.category == "RPG - Game Master Resources"
    assert not result.requires_review
    retry_categories = service.ollama.classify_file.call_args.args[1]
    assert "Documents" not in retry_categories


def test_empty_ai_response_gives_actionable_review_reason(monkeypatch):
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence", lambda p: "World building tools")
    service, file = classifier(None)
    result = service.classify(file)
    assert result.category == NEEDS_REVIEW
    assert "Ollama" in result.reason



def test_book_retry_uses_extension_eligible_categories_not_prefixes(monkeypatch):
    """Maker books and future custom topics must be considered during AI retry."""
    from core.document_classifier import book_subject_categories

    categories = [
        {"name": "Documents", "extensions": [".pdf"]},
        {"name": "eBooks", "extensions": [".epub"]},
        {"name": "Maker & DIY", "extensions": [".pdf", ".epub"],
         "keywords": ["HackSpace", "Raspberry Pi Pico", "soldering"]},
        {"name": "Pictures", "extensions": [".png"]},
        {"name": "Needs Review", "extensions": []},
    ]
    assert [c["name"] for c in book_subject_categories(categories, ".pdf")] == ["Maker & DIY"]
    assert [c["name"] for c in book_subject_categories(categories, ".epub")] == ["Maker & DIY"]


def test_maker_book_classifies_on_subject_only_retry(monkeypatch):
    monkeypatch.setattr(
        "core.document_classifier.extract_pdf_evidence",
        lambda path: "Book of Making 2025 HackSpace DIY electronics and Raspberry Pi projects"
    )
    service, file = classifier("eBooks")
    maker = {
        "name": "Maker & DIY", "extensions": [".pdf", ".epub"],
        "keywords": ["HackSpace", "microcontrollers", "making", "3D printing"],
    }
    service.definitions = CATS + [maker]
    service.category_names = [item["name"] for item in service.definitions]
    service.ollama.classify_file.side_effect = ["eBooks", "Maker & DIY"]
    decision = service.classify(file)
    assert decision.category == "Maker & DIY"
    assert decision.requires_review is False
    categories_on_retry = service.ollama.classify_file.call_args.args[1]
    assert "Maker & DIY" in categories_on_retry
    assert "eBooks" not in categories_on_retry


def test_failed_book_gets_title_based_suggestion_without_automatic_category(monkeypatch):
    monkeypatch.setattr(
        "core.document_classifier.extract_pdf_evidence",
        lambda p: "This is an edited collection of hands-on maker projects."
    )
    maker = {
        "name": "Maker & DIY", "extensions": [".pdf", ".epub"],
        "keywords": ["Book of Making", "electronics"],
    }
    service, book = classifier(None)
    service.definitions = CATS + [maker]
    service.category_names = [item["name"] for item in service.definitions]
    book["file_name"] = "Book_of_Making_2025_Z-Library.pdf"
    decision = service.classify(book)
    assert decision.category == NEEDS_REVIEW
    assert decision.requires_review is True
    assert decision.suggested_category == "Maker & DIY"
    assert "Suggested category: Maker & DIY" in decision.reason
    assert decision.book_title == "Book of Making 2025"


def test_external_book_lookup_disabled_by_default(monkeypatch):
    import core.book_identifier as book_identifier
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence", lambda p: "Game rules")
    monkeypatch.setattr(book_identifier, "openlibrary_lookup",
                        lambda identity: (_ for _ in ()).throw(
                            AssertionError("External lookup should require opt-in")
                        ))
    service, file = classifier(None)
    decision = service.classify(file)
    assert decision.category == NEEDS_REVIEW


def test_suggestion_never_becomes_move_category(monkeypatch):
    monkeypatch.setattr("core.document_classifier.extract_pdf_evidence",
                        lambda p: "Practical electronics reference")
    service, file = classifier(None)
    file["file_name"] = "Book_of_Making_2025.pdf"
    service.definitions += [{
        "name": "Maker & DIY", "extensions": [".pdf"], "keywords": ["Book of Making"]
    }]
    service.category_names = [d["name"] for d in service.definitions]
    decision = service.classify(file)
    assert decision.category == NEEDS_REVIEW
    assert decision.suggested_category == "Maker & DIY"
