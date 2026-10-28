"""Optional book identification and conservative category suggestions.

No network traffic unless external_lookup=True. Never automatically categorizes files.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
ISBN_RE = re.compile(
    r"\bISBN(?:[- ]?1[03])?\s*[:#]?\s*([0-9Xx](?:[ \t-]?[0-9Xx]){9,12})", re.I
)
GENERIC_NAMES = {"documents", "ebooks", "miscellaneous", "reference & manuals", "needs review"}
CLEAN_SUFFIX = re.compile(
    r"(?i)[ _-]*(?:z[-_ ]?library|z[-_ ]?lib|1lib)(?:\.[a-z]{2,4})?.*$"
)
WORD_RE = re.compile(r"[a-z0-9]+", re.I)


@dataclass(frozen=True)
class BookIdentity:
    title: str
    isbn: str | None = None
    author: str | None = None
    subjects: tuple[str, ...] = ()
    source: str = "local"


@dataclass(frozen=True)
class BookSuggestion:
    category: str
    identity: BookIdentity
    explanation: str


def valid_isbn(value: str) -> str | None:
    code = re.sub(r"[^0-9Xx]", "", value).upper()
    if len(code) == 10:
        if any(ch not in "0123456789" for ch in code[:9]):
            return None
        total = sum((10 - i) * int(ch) for i, ch in enumerate(code[:9]))
        total += 10 if code[-1] == "X" else int(code[-1])
        return code if total % 11 == 0 else None
    if len(code) == 13 and code.startswith(("978", "979")) and code.isdigit():
        total = sum(int(ch) * (1 if i % 2 == 0 else 3)
                    for i, ch in enumerate(code[:12]))
        return code if (10 - total % 10) % 10 == int(code[-1]) else None
    return None


def extract_isbn(metadata: dict, evidence: str | None) -> str | None:
    for key in ("isbn", "ISBN", "isbn_13", "isbn_10", "identifier"):
        if metadata.get(key):
            value = metadata[key]
            parts = value if isinstance(value, (list, tuple)) else [value]
            for part in parts:
                cleaned = valid_isbn(str(part))
                if cleaned:
                    return cleaned
    # Only labelled ISBNs: avoid confusing unrelated 10/13-digit IDs with ISBNs.
    for match in ISBN_RE.finditer((evidence or "")[:7000]):
        digits = re.sub(r"[^0-9Xx]", "", match.group(1))
        # Prefer a complete ISBN-13, but also allow ISBN-10 followed by
        # adjacent page/year digits from a noisy PDF text extraction.
        for candidate in (digits[:13], digits[:10]):
            isbn = valid_isbn(candidate)
            if isbn:
                return isbn
    return None


def clean_title(file_name: str, metadata: dict, evidence: str | None) -> str:
    title = str(metadata.get("title") or "").strip()
    if not title or title.casefold() in {"untitled", "document", "microsoft word"} or len(title) < 5:
        title = CLEAN_SUFFIX.sub("", Path(file_name).stem)
        title = title.replace("_", " ")
        title = re.sub(r"\s+", " ", title).strip(" -_.")
    # Don't send long or noisy filenames to the external search service.
    if not title or len(title) < 4:
        lines = [line.strip() for line in (evidence or "").splitlines()[:7]]
        title = next((line for line in lines if 6 <= len(line) <= 100), "")
    return title[:140]


def identify_book(file_name: str, metadata: dict, evidence: str | None) -> BookIdentity:
    return BookIdentity(
        title=clean_title(file_name, metadata, evidence),
        isbn=extract_isbn(metadata, evidence),
        author=str(metadata.get("author") or "").strip() or None,
    )


def _normalize_title(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def openlibrary_lookup(identity: BookIdentity, timeout: float = 6) -> BookIdentity:
    """Search Open Library by ISBN first, then by title, and verify a match.

    This is only called after an explicit opt-in in the application settings.
    """
    if not identity.isbn and len(identity.title) < 5:
        return identity

    queries = []
    if identity.isbn:
        queries.append({"isbn": identity.isbn})
    if len(identity.title) >= 5:
        queries.append({"title": identity.title})
    for query in queries:
        try:
            response = requests.get(
                "https://openlibrary.org/search.json",
                params={
                    **query, "limit": 5,
                    "fields": "key,title,author_name,subject,isbn",
                },
                headers={"User-Agent": "LocalAIFileOrganizer/1.0 (book metadata lookup)"},
                timeout=timeout,
            )
            response.raise_for_status()
            docs = response.json().get("docs") or []
        except (requests.RequestException, ValueError, AttributeError) as exc:
            logger.warning("Open Library lookup failed (%s): %s",
                           "ISBN" if "isbn" in query else "title", exc)
            continue

        for item in docs[:5]:
            title = str(item.get("title") or "")
            remote_isbns = {
                re.sub(r"[^0-9Xx]", "", str(value)).upper()
                for value in (item.get("isbn") or [])
            }
            isbn_match = bool(identity.isbn and identity.isbn in remote_isbns)
            local_title = _normalize_title(identity.title)
            remote_title = _normalize_title(title)
            title_match = (
                len(local_title) >= 5 and len(remote_title) >= 5
                and SequenceMatcher(None, local_title, remote_title).ratio() >= .83
            )
            if not (isbn_match or title_match):
                continue
            return BookIdentity(
                title=title[:140],
                isbn=identity.isbn,
                author=", ".join(str(x) for x in (item.get("author_name") or [])[:2]) or identity.author,
                subjects=tuple(str(x) for x in (item.get("subject") or [])[:15]),
                source="Open Library ISBN" if isbn_match else "Open Library title",
            )
    return identity


def suggest_category(identity: BookIdentity, definitions: list[dict],
                     extension: str) -> BookSuggestion | None:
    """Match specific category keywords against *title and bibliographic subjects*.

    Require a clear signal; ambiguous matches stay Needs Review.
    """
    title = _normalize_title(identity.title)
    subjects = _normalize_title(" ".join(identity.subjects))
    scored = []
    for item in definitions:
        name = str(item.get("name") or "")
        if name.casefold() in GENERIC_NAMES:
            continue
        suffixes = {str(x).casefold() for x in (item.get("extensions") or [])}
        if extension.casefold() not in suffixes:
            continue
        keywords = item.get("keywords") or []
        title_hits, subject_hits = set(), set()
        for keyword in keywords:
            kw = _normalize_title(str(keyword))
            if len(kw) < 4:
                continue
            # Word-boundary matching avoids e.g. 'unity' matching 'community'.
            if re.search(r"\b" + re.escape(kw) + r"\b", title):
                title_hits.add(kw)
            if subjects and re.search(r"\b" + re.escape(kw) + r"\b", subjects):
                subject_hits.add(kw)
        score = 3 * len(title_hits) + 2 * len(subject_hits)
        if score:
            scored.append((score, name, title_hits, subject_hits))
    scored.sort(reverse=True)
    if not scored:
        return None
    score, category, title_hits, subject_hits = scored[0]
    # A single explicit multiword keyword in the title is strong evidence;
    # generic single-word overlap requires corroboration.
    strong_title = any(" " in kw for kw in title_hits)
    if score < 4 and not strong_title:
        return None
    if len(scored) > 1 and scored[1][0] >= score:
        return None
    matches = sorted(title_hits | subject_hits)[:3]
    source = "catalogue subjects and title" if subject_hits else "book title"
    return BookSuggestion(
        category=category,
        identity=identity,
        explanation=f"{source}; matched {', '.join(matches)}",
    )


def identify_and_suggest(filename: str, metadata: dict, evidence: str | None,
                         definitions: list[dict], extension: str,
                         external_lookup: bool = False) -> tuple[BookIdentity, BookSuggestion | None]:
    identity = identify_book(filename, metadata, evidence)
    if external_lookup:
        identity = openlibrary_lookup(identity)
    return identity, suggest_category(identity, definitions, extension)
