"""Shared subject-aware AI classification for scanned files and ebooks."""
from __future__ import annotations

import json
import logging
import posixpath
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

NEEDS_REVIEW = "Needs Review"
BOOK_EXTS = {".pdf", ".epub"}
GENERIC_BOOK_CATEGORIES = {"documents", "ebooks", "miscellaneous", "reference & manuals"}
logger = logging.getLogger(__name__)


def load_category_definitions(path, db=None):
    """Keep keywords with categories; suppress duplicate names from user DB."""
    with open(path, encoding="utf-8") as f:
        definitions = json.load(f)["categories"]
    if db is not None:
        try:
            definitions.extend(db.get_custom_categories())
        except Exception as exc:
            logger.warning("Failed to load custom categories: %s", exc)
    seen, unique = set(), []
    for item in definitions:
        name = str(item.get("name", "")).strip()
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            unique.append(item)
    if NEEDS_REVIEW.casefold() not in seen:
        unique.append({"name": NEEDS_REVIEW, "keywords": [], "extensions": []})
    return unique



def book_subject_categories(definitions, extension):
    """Eligible subject categories for PDF/EPUB retry, including user-defined ones.

    Category eligibility follows configured file extensions, not naming prefixes.
    Generic buckets (eBooks/Documents) and Needs Review are never candidates.
    """
    ext = extension.lower()
    return [
        item for item in definitions
        if item.get("name", "").casefold() not in GENERIC_BOOK_CATEGORIES
        and item.get("name", "") != NEEDS_REVIEW
        and ext in {
            str(suffix).lower() if str(suffix).startswith(".") else "." + str(suffix).lower()
            for suffix in (item.get("extensions") or [])
        }
    ]


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.words = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.words.append(data.strip())


def extract_pdf_evidence(path, limit=7000):
    """Extract title/TOC and samples from different parts of a book."""
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            count = len(doc)
            if not count:
                return None
            sections = []
            toc = doc.get_toc()
            if toc:
                sections.append("Contents: " + " | ".join(t[1] for t in toc[:35])[:1600])
            indices = list(dict.fromkeys(
                i for i in (0, 1, 2, 4, count // 4, count // 2, count * 3 // 4)
                if i < count
            ))
            for index in indices:
                text = " ".join(doc[index].get_text("text").split())
                if text:
                    sections.append(f"[Page {index + 1}] {text[:1200]}")
                if sum(map(len, sections)) >= limit:
                    break
            return "\n".join(sections)[:limit] or None
    except Exception as exc:
        logger.warning("PDF evidence extraction failed for %s: %s", path, exc)
        return None


def extract_epub_evidence(path, limit=7000):
    """Read EPUB metadata and XHTML in package spine order (not ZIP order)."""
    try:
        with zipfile.ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = root.find(".//{*}rootfile")
            if rootfile is None:
                return None
            package_path = rootfile.attrib["full-path"]
            package = ElementTree.fromstring(archive.read(package_path))
            parts = []
            metadata = package.find(".//{*}metadata")
            if metadata is not None:
                dc_ns = "{http://purl.org/dc/elements/1.1/}"
                for tag in ("title", "creator", "subject", "description"):
                    values = [n.text.strip() for n in metadata.findall(dc_ns + tag)
                              if n.text and n.text.strip()]
                    if values:
                        parts.append(tag.title() + ": " + " | ".join(values)[:550])
            manifest = package.find(".//{*}manifest")
            spine = package.find(".//{*}spine")
            if manifest is None or spine is None:
                return "\n".join(parts)[:limit] or None
            items = {i.attrib.get("id"): i for i in manifest.findall("{*}item")}
            ordered = [items[i.attrib["idref"]] for i in spine.findall("{*}itemref")
                       if i.attrib.get("idref") in items]
            nav = [i for i in items.values()
                   if "nav" in i.attrib.get("properties", "").split()]
            choices = nav[:1]
            if ordered:
                indices = list(dict.fromkeys(i for i in
                    (0, 1, 2, len(ordered)//4, len(ordered)//2, len(ordered)*3//4)
                    if i < len(ordered)))
                choices.extend(ordered[i] for i in indices)
            names = set(archive.namelist())
            for item in choices:
                href = unquote(item.attrib.get("href", "").split("#")[0])
                member = posixpath.normpath(posixpath.join(posixpath.dirname(package_path), href))
                if member not in names:
                    continue
                parser = _HTMLText()
                # Limit bytes read from each ZIP member to avoid oversized HTML entries.
                with archive.open(member) as page:
                    parser.feed(page.read(131072).decode("utf-8", errors="replace"))
                snippet = " ".join(parser.words)[:1200]
                if snippet:
                    parts.append(f"[{href}] {snippet}")
                if sum(map(len, parts)) >= limit:
                    break
            return "\n".join(parts)[:limit] or None
    except Exception as exc:
        logger.warning("EPUB evidence extraction failed for %s: %s", path, exc)
        return None


@dataclass(frozen=True)
class ClassificationDecision:
    category: str
    reason: str
    metadata: dict
    content_summary: str | None
    requires_review: bool = False


class DocumentClassificationService:
    """Single classifier shared by Analyze and Quick Organize."""

    def __init__(self, ollama, reader, metadata_extractor, ocr, definitions):
        self.ollama = ollama
        self.reader = reader
        self.metadata_extractor = metadata_extractor
        self.ocr = ocr
        self.definitions = definitions
        self.category_names = [d["name"] for d in definitions]

    def classify(self, file_data):
        path = file_data.get("file_path", "")
        extension = str(file_data.get("extension") or Path(path).suffix).lower()
        metadata = {}
        try:
            metadata = self.metadata_extractor.extract(path) or {}
        except Exception as exc:
            logger.warning("Metadata extraction failed for %s: %s", path, exc)

        try:
            if extension == ".pdf":
                content = extract_pdf_evidence(path)
            elif extension == ".epub":
                content = extract_epub_evidence(path)
            else:
                content = self.reader.read_summary(path, max_length=3000)
        except Exception as exc:
            logger.warning("Content extraction failed for %s: %s", path, exc)
            content = None

        if not content and extension == ".pdf" and self.ocr is not None:
            try:
                if self.ocr.is_available():
                    content = self.ocr.extract_text(path, max_length=7000)
            except Exception as exc:
                logger.warning("OCR failed for %s: %s", path, exc)

        if extension in BOOK_EXTS and not content:
            return ClassificationDecision(
                NEEDS_REVIEW, "No readable document text; check PDF OCR or EPUB file",
                metadata, content, True
            )

        file_info = {
            "file_name": file_data.get("file_name") or Path(path).name,
            "extension": extension,
            "metadata": metadata,
        }
        try:
            category = self.ollama.classify_file(
                file_info, self.category_names, content,
                category_details=self.definitions
            )
        except Exception as exc:
            logger.warning("AI classification failed for %s: %s", path, exc)
            return ClassificationDecision(
                NEEDS_REVIEW, f"AI request failed: {type(exc).__name__}: {str(exc)[:120]}",
                metadata, content, True
            )

        # Retry generic book categories once, using only subject categories.
        # Unlike extension fallback, the model must still use book evidence.
        if extension in BOOK_EXTS and category and category.casefold() in GENERIC_BOOK_CATEGORIES:
            subject_definitions = book_subject_categories(self.definitions, extension)
            if subject_definitions:
                try:
                    names = [item["name"] for item in subject_definitions]
                    retried = self.ollama.classify_file(
                        file_info, names, content, category_details=subject_definitions
                    )
                    if retried in names:
                        logger.info("Reclassified %s from %s to %s on retry",
                                    path, category, retried)
                        return ClassificationDecision(
                            retried, f"AI subject-only retry (initially {category})",
                            metadata, content
                        )
                except Exception as exc:
                    logger.warning("Subject-only retry failed for %s: %s", path, exc)

        if not category:
            reason = "Ollama returned no usable category; check model and application logs"
        elif category not in self.category_names:
            reason = f"AI returned unknown category: {category}"
        elif category in (NEEDS_REVIEW, "Miscellaneous"):
            reason = "AI could not identify a specific category"
        elif extension in BOOK_EXTS and category.casefold() in GENERIC_BOOK_CATEGORIES:
            reason = f"AI returned generic book category: {category}; subject retry failed"
        else:
            return ClassificationDecision(category, "AI subject classification", metadata, content)

        logger.warning("Classification needs review for %s: %s", path, reason)
        return ClassificationDecision(NEEDS_REVIEW, reason, metadata, content, True)
