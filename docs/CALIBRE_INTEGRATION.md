# Calibre ebook integration

This optional integration uses Calibre's supported command-line tools for checking duplicates, adding missing ebooks, and optionally downloading metadata.

## Supported formats

EPUB, MOBI, AZW, AZW3, AZW4, PRC, FB2, KEPUB.

It deliberately excludes PDFs, ZIP archives, images, and other documents.

## Setup

1. In Calibre, locate the library folder containing metadata.db.
2. In Local AI File Organizer open Settings > Advanced > Calibre ebook integration.
3. Enable checking ebooks against a local Calibre library.
4. Set Calibre library to the folder containing metadata.db.
5. If the programs aren't on PATH, enter your Calibre installation folder as Calibre tools (usually C:\\Program Files\\Calibre2 on Windows).
6. Save settings, and test a few *copied* ebooks in Analyze. The Calibre column shows the result.
7. When duplicate matches look correct, enable Import missing ebooks.
8. Optionally enable After import, download metadata using Calibre's online sources. This can transmit title, author, or ISBN to the installed metadata plugins.

Checking, importing, and metadata downloading are explicit choices. All three default to disabled. They apply to both Analyze and Quick Organize, with or without AI classification.

## How matching works

- The original ebook's title/author/ISBN is read locally via ebook-meta.
- Calibre library records are read via calibredb list --for-machine.
- A valid ISBN match is treated as an existing book, even when the library has another ebook format.
- A normalized title *and full author* match is treated as an existing book.
- Title-only matches are ambiguous: they are held for review; no import occurs.
- Other ebooks can be imported via calibredb add (if importing is enabled).
- After adding, the integration compares Calibre book IDs before/after and verifies the newly created record; it does not trust text in CLI output.
- Metadata retrieval uses fetch-ebook-metadata --opf and calibredb set_metadata, with an ISBN/title sanity check to avoid applying data for a different book.
- Missing executables, locked libraries, unexpected outputs or ambiguous matches are reported for review.

## File safety

- The original file is *never* deleted or moved by this integration.
- A duplicate gets is_duplicate=1 and duplicate_of=calibre:book:<ID> in the application's SQLite index.
- Duplicates, imported originals and uncertain matches remain in Needs Review and are excluded from automatic folder moves.
- Files *inside* the Calibre library are never imported again or moved out.
- This does not place matched ebooks in the application's hash-based Duplicate Finder groups; the Analyze Calibre column and persisted index hold the Calibre match.
- No direct changes to Calibre's metadata.db are made; all actions use Calibre's CLI.

## Local testing

Run the new mocked tests before using your real library:

    .\.venv\Scripts\python.exe -m pytest tests/test_calibre_bridge.py -v
    .\.venv\Scripts\python.exe -m pytest tests/ -v

The tests do not contact your Calibre installation. For an end-to-end verification, back up your Calibre library and use a copy of the library with a few copied ebooks first.

## Calibre documentation

- https://manual.calibre-ebook.com/generated/en/calibredb.html
- https://manual.calibre-ebook.com/generated/en/ebook-meta.html
- https://manual.calibre-ebook.com/generated/en/fetch-ebook-metadata.html
