"""Non-destructive local diagnostic for PDF/EPUB classification and Ollama responses.

Example:
    python scripts/diagnose_book_classification.py --folder "D:\Books\Test" --model qwen3.5:4b
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import ollama_client as ollama_module
from core.content_reader import ContentReader
from core.document_classifier import (
    DocumentClassificationService, extract_pdf_evidence,
    extract_epub_evidence, load_category_definitions,
)
from core.metadata import MetadataExtractor
from core.ocr import OCRProcessor
from core.ollama_client import OllamaClient
from utils.config import ConfigManager


def inspect_response(payload, response=None, error=None, duration=None):
    """Capture diagnosis without storing the prompt or extracted book content."""
    entry = {
        "model": payload.get("model"),
        "thinking": payload.get("think", "model default"),
        "forced_json": payload.get("format") == "json",
        "num_predict": (payload.get("options") or {}).get("num_predict"),
        "prompt_characters": sum(len(str(m.get("content") or ""))
                                 for m in payload.get("messages", [])),
        "seconds": round(duration, 2) if duration is not None else None,
    }
    if error is not None:
        entry["error"] = f"{type(error).__name__}: {error}"
    if response is None:
        return entry
    entry["http_status"] = response.status_code
    if response.status_code != 200:
        entry["error"] = response.text[:500]
        return entry
    try:
        data = response.json()
    except ValueError:
        entry["error"] = "Ollama did not return JSON"
        return entry
    message = data.get("message") or {}
    answer = message.get("content") or ""
    entry.update({
        "done_reason": data.get("done_reason"),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "response_characters": len(answer),
        "thinking_characters": len(message.get("thinking") or ""),
        "response_preview": answer[:350],
    })
    return entry


def install_trace(trace):
    """Observe the real Ollama request made by the existing classification code."""
    original_post = ollama_module.requests.post

    def trace_post(url, **kwargs):
        if not str(url).endswith("/api/chat"):
            return original_post(url, **kwargs)
        start = time.monotonic()
        payload = kwargs.get("json") or {}
        try:
            response = original_post(url, **kwargs)
        except Exception as exc:
            trace.append(inspect_response(payload, error=exc,
                                          duration=time.monotonic()-start))
            raise
        trace.append(inspect_response(payload, response=response,
                                      duration=time.monotonic()-start))
        return response

    ollama_module.requests.post = trace_post
    return original_post


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Individual PDF or EPUB files")
    parser.add_argument("--folder", help="Scan a folder recursively for PDFs and EPUBs")
    parser.add_argument("--model", help="Actual model selected in the Analyze tab")
    parser.add_argument("--url", help="Ollama URL, defaults to app configuration")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offline", action="store_true", help="Only test text extraction")
    parser.add_argument("--output", default="classification_diagnostic.json")
    args = parser.parse_args(argv)

    files = [Path(x).expanduser().resolve() for x in args.files]
    if args.folder:
        folder = Path(args.folder).expanduser().resolve()
        if not folder.is_dir():
            parser.error(f"Not a directory: {folder}")
        files.extend(x for x in folder.rglob("*") if x.is_file())
    files = list(dict.fromkeys(
        x for x in files if x.is_file() and x.suffix.lower() in (".pdf", ".epub")
    ))[:args.limit]
    if not files:
        parser.error("Specify at least one existing PDF/EPUB file, or --folder")

    config = ConfigManager().config
    settings = config.get("ollama", {})
    model = args.model or settings.get("model", "qwen2.5:3b")
    url = args.url or settings.get("server_url", "http://localhost:11434")
    client = OllamaClient(server_url=url, model=model, timeout=args.timeout,
                          max_tokens=settings.get("max_tokens", 100),
                          temperature=settings.get("temperature", 0.1))
    definitions = load_category_definitions(ROOT / "config" / "categories.json")
    ocr_settings = config.get("ocr", {})
    ocr = OCRProcessor(enabled=ocr_settings.get("enabled", False),
                       language=ocr_settings.get("language", "eng"),
                       max_pages=ocr_settings.get("max_pages", 10))
    service = DocumentClassificationService(
        client, ContentReader(), MetadataExtractor(), ocr, definitions
    )
    report = {
        "ollama_url": url, "model": model,
        "ocr_available": ocr.is_available(), "category_count": len(definitions),
        "books": [],
    }
    if not args.offline:
        report["ollama_reachable"] = client.is_available()
        report["available_models"] = client.list_models()
        if not report["ollama_reachable"]:
            print("WARNING: Cannot reach Ollama:", url)
        if model not in report["available_models"]:
            print("WARNING: The model is not listed by Ollama:", model)

    for book in files:
        print("\nChecking", book.name)
        evidence = (extract_pdf_evidence(book) if book.suffix.lower() == ".pdf"
                    else extract_epub_evidence(book))
        entry = {
            "file": str(book), "extension": book.suffix.lower(),
            "evidence_characters": len(evidence or ""),
        }
        if args.offline:
            entry["category"] = "Not tested (--offline)"
        else:
            trace = []
            original_post = install_trace(trace)
            try:
                decision = service.classify({
                    "file_path": str(book), "file_name": book.name,
                    "extension": book.suffix.lower(),
                })
                entry.update({
                    "category": decision.category, "reason": decision.reason,
                    "requires_review": decision.requires_review,
                    "suggested_category": decision.suggested_category,
                    "book_title": decision.book_title,
                    "book_isbn": decision.book_isbn,
                })
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                ollama_module.requests.post = original_post
            entry["ollama_requests"] = trace
        report["books"].append(entry)
        print("  Readable characters:", entry["evidence_characters"])
        print("  Category:", entry.get("category", "ERROR"))
        print("  Reason:", entry.get("reason", entry.get("error", "-")))
        if entry.get("suggested_category"):
            print("  Suggested category:", entry["suggested_category"])
        for call in entry.get("ollama_requests", []):
            print("  Ollama HTTP:", call.get("http_status", "failed"),
                  "output:", call.get("response_characters", 0),
                  "thinking:", call.get("thinking_characters", 0),
                  "done_reason:", call.get("done_reason"),
                  "error:", str(call.get("error") or "")[:120])

    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print("\nReport:", destination)
    print("Review filenames and response previews before sharing the report.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
