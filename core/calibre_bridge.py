"""Opt-in local Calibre ebook duplicate detection and importing.

Uses Calibre's supported CLI tools, never writes metadata.db directly.
Only known ebook formats are eligible; PDFs and archives are excluded.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from core.book_identifier import BookIdentity, clean_title, valid_isbn

logger = logging.getLogger(__name__)

EBOOK_EXTENSIONS = frozenset({".epub", ".mobi", ".azw", ".azw3", ".azw4",
                              ".prc", ".fb2", ".kepub"})
DC = "{http://purl.org/dc/elements/1.1/}"
IDENTIFIER_NS = "{http://www.idpf.org/2007/opf}"


@dataclass(frozen=True)
class CalibreResult:
    status: str  # skipped, duplicate, imported, review, error
    message: str
    book_id: int | None = None
    title: str | None = None
    library: str | None = None

    @property
    def requires_review(self) -> bool:
        return self.status in ("duplicate", "review", "error", "imported")

    @property
    def is_duplicate(self) -> bool:
        return self.status == "duplicate"


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _authors(value) -> set[str]:
    if isinstance(value, list):
        text = " & ".join(str(x) for x in value)
    else:
        text = str(value or "")
    return {_norm(x) for x in re.split(r"\s*(?:&|;|\band\b)\s*", text)
            if _norm(x) not in ("", "unknown", "unknown author")}


def _book_isbns(record: dict) -> set[str]:
    numbers = [record.get("isbn")]
    identifiers = record.get("identifiers") or {}
    if isinstance(identifiers, dict):
        numbers.extend(v for k, v in identifiers.items() if str(k).lower() == "isbn")
    return {code for n in numbers if n for code in [valid_isbn(str(n))] if code}


def _identity_from_opf(xml: str, fallback_title: str) -> BookIdentity:
    root = ElementTree.fromstring(xml)
    title_node = root.find(".//" + DC + "title")
    title = (title_node.text or "").strip() if title_node is not None else ""
    authors = [str(node.text).strip() for node in root.findall(".//" + DC + "creator")
               if node.text and node.text.strip()]
    isbn = None
    for node in root.findall(".//" + DC + "identifier"):
        candidate = valid_isbn(node.text or "")
        if candidate:
            isbn = candidate
            break
    return BookIdentity(
        title=title if title and len(title) > 2 else fallback_title,
        isbn=isbn,
        author=" & ".join(authors) or None,
        source="Calibre ebook-meta",
    )


class CalibreBridge:
    """One instance per scan; calibre library listing cached for the scan."""

    def __init__(self, settings: dict):
        self.enabled = bool(settings.get("enabled", False))
        self.import_missing = bool(settings.get("import_missing", False))
        self.fetch_metadata = bool(settings.get("fetch_metadata", False))
        self.library = Path(settings.get("library_path") or "").expanduser()
        self.cli_dir = str(settings.get("cli_path") or "").strip()
        self.timeout = max(15, min(int(settings.get("timeout", 90)), 600))
        self._records: list[dict] | None = None

    def _executable(self, program: str) -> str:
        if self.cli_dir:
            path = Path(self.cli_dir).expanduser()
            if path.is_dir():
                for name in (program + ".exe", program):
                    candidate = path / name
                    if candidate.is_file():
                        return str(candidate)
            elif path.is_file():
                sibling = path.parent / (program + path.suffix)
                if sibling.is_file():
                    return str(sibling)
                if path.stem.casefold() == program.casefold():
                    return str(path)
            raise FileNotFoundError(f"Cannot locate {program} in {self.cli_dir}")
        found = shutil.which(program)
        if not found:
            raise FileNotFoundError(
                f"{program} not found. Configure the Calibre program folder in Settings."
            )
        return found

    def _run(self, name: str, *args: str) -> str:
        command = [self._executable(name), *[str(arg) for arg in args]]
        result = subprocess.run(
            command, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=self.timeout,
            check=False, stdin=subprocess.DEVNULL,
        )
        if result.returncode:
            raise RuntimeError(
                f"{name} exited {result.returncode}: {result.stderr.strip()[:350]}"
            )
        return result.stdout

    def _db(self, *args: str) -> str:
        return self._run("calibredb", "--with-library", str(self.library), *args)

    def _validate(self) -> None:
        if not self.library.is_dir() or not (self.library / "metadata.db").is_file():
            raise ValueError(
                "Choose a Calibre library folder containing metadata.db in Settings."
            )
        self._executable("calibredb")
        self._executable("ebook-meta")

    def _list(self, refresh: bool = False) -> list[dict]:
        if self._records is None or refresh:
            output = self._db(
                "list", "--for-machine",
                "--fields", "title,authors,isbn,identifiers,formats",
            )
            records = json.loads(output)
            if not isinstance(records, list):
                raise ValueError("Calibre list did not return a book list")
            self._records = records
        return self._records

    def _identify(self, path: Path) -> BookIdentity:
        # ebook-meta writes an OPF file but does not alter the source ebook.
        # It reads EPUB, MOBI, AZW3 and other supported ebook formats.
        default = clean_title(path.name, {}, None)
        with tempfile.TemporaryDirectory(prefix="calibre-book-meta-") as tmp:
            opf = Path(tmp) / "ebook.opf"
            self._run("ebook-meta", str(path), "--to-opf", str(opf))
            identity = _identity_from_opf(opf.read_text(encoding="utf-8"), default)
        return identity

    def _match(self, book: BookIdentity) -> tuple[dict | None, bool]:
        """Return exact duplicate, or ambiguous-title flag (never auto-import)."""
        isbn = valid_isbn(book.isbn or "")
        title = _norm(book.title)
        authors = _authors(book.author)
        possible = False
        for existing in self._list():
            if isbn and isbn in _book_isbns(existing):
                return existing, False
            if title and title == _norm(existing.get("title")):
                existing_authors = _authors(existing.get("authors"))
                if authors and existing_authors and authors.intersection(existing_authors):
                    return existing, False
                possible = True
        return None, possible

    def _fetch_and_set_metadata(self, book_id: int, book: BookIdentity) -> str:
        """Use Calibre's own metadata plugins; reject a mismatched online book."""
        args = ["--opf"]
        if book.isbn:
            args.extend(["--isbn", book.isbn])
        elif book.title and len(book.title) >= 5:
            args.extend(["--title", book.title])
            if book.author:
                args.extend(["--authors", book.author])
        else:
            return "Metadata lookup skipped: missing ISBN/title"
        try:
            self._executable("fetch-ebook-metadata")
            opf_xml = self._run("fetch-ebook-metadata", *args)
            if not opf_xml.strip():
                return "No online metadata returned"
            fetched = _identity_from_opf(opf_xml, "")
            # Online search sources can return a different edition or an
            # unrelated book. Never overwrite existing metadata unless the
            # identifier matches OR the result's title matches exactly.
            if book.isbn and fetched.isbn != book.isbn:
                if _norm(book.title) != _norm(fetched.title):
                    return "Online metadata ignored: ISBN/title mismatch"
            if not book.isbn and _norm(book.title) != _norm(fetched.title):
                return "Online metadata ignored: title mismatch"
            with tempfile.TemporaryDirectory(prefix="calibre-fetched-meta-") as tmp:
                opf = Path(tmp) / "metadata.opf"
                opf.write_text(opf_xml, encoding="utf-8")
                self._db("set_metadata", str(book_id), str(opf))
            return "Online metadata downloaded to Calibre"
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            logger.warning("Calibre metadata download failed for book %s: %s", book_id, exc)
            return f"Metadata download failed: {str(exc)[:130]}"

    def process(self, file_path: str | Path) -> CalibreResult:
        path = Path(file_path)
        if path.suffix.lower() not in EBOOK_EXTENSIONS or not self.enabled:
            return CalibreResult("skipped", "Calibre integration not applicable")
        if not path.is_file():
            return CalibreResult("error", f"Source ebook not found: {path.name}")
        try:
            self._validate()
            # Avoid treating books inside the Calibre library as new imports.
            if path.resolve().is_relative_to(self.library.resolve()):
                return CalibreResult(
                    "review", "Ebook is inside the managed Calibre library; never move it",
                    library=str(self.library),
                )
            book = self._identify(path)
            if not valid_isbn(book.isbn or "") and len(_norm(book.title)) < 5:
                return CalibreResult("review", "ISBN/title unavailable; cannot safely compare")
            existing, ambiguous = self._match(book)
            if existing:
                book_id = int(existing["id"])
                return CalibreResult(
                    "duplicate", f"Already in Calibre (book ID {book_id}: {existing.get('title')})",
                    book_id, str(existing.get("title") or book.title), str(self.library),
                )
            if ambiguous:
                return CalibreResult(
                    "review", "Possible matching Calibre title; author/ISBN confirmation required",
                    title=book.title, library=str(self.library),
                )
            if len(_norm(book.title)) < 5:
                return CalibreResult(
                    "review", "ISBN was not found in Calibre, but title is insufficient for import",
                    title=book.title, library=str(self.library),
                )
            if not self.import_missing:
                return CalibreResult(
                    "review", "Not present in Calibre; automatic importing disabled",
                    title=book.title, library=str(self.library),
                )
            # Re-read before adding to guard against a changed library since the
            # cached listing was loaded (e.g. another Calibre window imported it).
            self._list(refresh=True)
            existing, ambiguous = self._match(book)
            if existing:
                return CalibreResult(
                    "duplicate", f"Already in Calibre (book ID {existing['id']})",
                    int(existing["id"]), book.title, str(self.library),
                )
            if ambiguous:
                return CalibreResult(
                    "review", "Possible matching Calibre title; import not attempted",
                    title=book.title, library=str(self.library),
                )
            before_ids = {int(item["id"]) for item in self._records}
            args = ["add"]
            if book.title:
                args.extend(["--title", book.title])
            if book.author:
                args.extend(["--authors", book.author])
            if book.isbn:
                args.extend(["--isbn", book.isbn])
            args.append(str(path))
            self._db(*args)
            after_records = self._list(refresh=True)
            new_records = [
                item for item in after_records if int(item["id"]) not in before_ids
            ]
            # Don't rely on localized CLI success messages to find the ID.
            if len(new_records) == 1:
                book_id = int(new_records[0]["id"])
            elif len(new_records) > 1:
                same = [
                    item for item in new_records
                    if (book.isbn and book.isbn in _book_isbns(item))
                    or (_norm(book.title) == _norm(item.get("title"))
                        and (not book.author
                             or _authors(book.author).intersection(_authors(item.get("authors")))))
                ]
                if len(same) != 1:
                    return CalibreResult(
                        "review", "Multiple concurrent Calibre imports; verify new book manually",
                        title=book.title, library=str(self.library),
                    )
                book_id = int(same[0]["id"])
            else:
                # Calibre can reject duplicates with a zero exit status.
                existing, ambiguous = self._match(book)
                if existing:
                    return CalibreResult(
                        "duplicate", f"Calibre already has this book (ID {existing['id']})",
                        int(existing["id"]), book.title, str(self.library),
                    )
                return CalibreResult(
                    "review", "Calibre did not add a book; check its library",
                    title=book.title, library=str(self.library),
                )
            metadata_message = (
                self._fetch_and_set_metadata(book_id, book)
                if self.fetch_metadata else "Online metadata download disabled"
            )
            return CalibreResult(
                "imported", f"Imported to Calibre (ID {book_id}). {metadata_message}",
                book_id, book.title, str(self.library),
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError,
                subprocess.TimeoutExpired) as exc:
            logger.warning("Calibre check failed for %s: %s", path, exc)
            return CalibreResult("error", f"Calibre integration failed: {str(exc)[:180]}")


def apply_calibre_result(file_data: dict, result: CalibreResult) -> None:
    """Persist duplicate status and book identity separately from AI category."""
    if result.status == "skipped":
        return
    file_data["is_duplicate"] = int(result.is_duplicate)
    file_data["duplicate_of"] = (
        f"calibre:book:{result.book_id}" if result.is_duplicate else None
    )
    # Preserve prior metadata values and attach Calibre diagnostic details.
    try:
        metadata = json.loads(file_data.get("metadata_json") or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
    except (TypeError, ValueError):
        metadata = {}
    metadata["calibre"] = {
        "status": result.status, "book_id": result.book_id,
        "title": result.title, "message": result.message, "library": result.library,
    }
    file_data["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
