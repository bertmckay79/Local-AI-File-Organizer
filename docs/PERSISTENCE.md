# Persistent settings, categories and file history

All runtime data now resides in a **per-user directory**, not in the
version-controlled source tree.

| What | Windows default |
| --- | --- |
| Settings (Ollama, OCR, scan rules, online book lookup, UI) | `%LOCALAPPDATA%\LocalAIFileOrganizer\user_config.json` |
| Database (custom categories, file records, operations and index) | `%LOCALAPPDATA%\LocalAIFileOrganizer\file_organizer.db` |
| Logs | `%LOCALAPPDATA%\LocalAIFileOrganizer\logs\organizer.log` |

Linux defaults to `~/.local/share/LocalAIFileOrganizer` (or
`$XDG_DATA_HOME/LocalAIFileOrganizer`). Portable/test installs can
override the data directory with `LOCAL_AI_FILE_ORGANIZER_DATA_DIR`.

The default categories remain in version-controlled `config/categories.json`.
**New custom categories and subcategories are stored in SQLite**, surviving
branches, application upgrades and restart. Folder naming rules are distinct
from the database. The application retains `-AI` on top-level generated folders.

## One-time migration

When the app first starts with the new persistence code, it:

1. Imports the previous source-tree `config/user_config.json` if no
   persisted user config exists yet.
2. Loads defaults with those user overrides.
3. Imports the previous relative `file_organizer.db` if no
   persisted database exists. SQLite's backup API includes WAL transactions.
4. Upgrades the old category strings in existing file records without
   removing the original book metadata, custom categories or operation history.

The old files remain where they were: **no deletion or overwrite**. If an app-data
database already exists, the legacy one is not automatically imported again.
When an explicitly configured absolute DB path is present, it is used as-is.

If an earlier release was run from several working directories, you may have
multiple legacy `file_organizer.db` files. The program cannot safely
merge these automatically. Compare them before deleting any. It chooses
the source-tree copy before a current-working-directory copy on first import.

## What persists

- All settings saved through Settings → Save Settings.
- Last-used scan folder selections in Scan and Quick Organize.
- Last selected tab and window dimensions.
- Indexed file records, manual review decisions and custom categories.
- Previously analysed and still-existing files (first 2,000 records are
  restored to the preview/organize workflow on startup).

Changes to selected scan folders are persisted immediately. All other
configuration is saved on normal exit or when you click Save Settings.
Reopening the app never initiates a scan, AI inference or a file move automatically.

## Troubleshooting

In PowerShell:

```powershell
explorer "$env:LOCALAPPDATA\LocalAIFileOrganizer"
Get-ChildItem "$env:LOCALAPPDATA\LocalAIFileOrganizer" -Force
```

If your categories or indexed files seem missing, **don't delete the old
database**. Check the above location and your cloned repo's
`file_organizer.db`; running the old app from other folders may have
produced more than one database. Back up all copies before deciding which
should become the persistent database.

Back up the user-data folder before switching versions. It can hold cloud AI
API keys in the settings JSON, so keep backups private and never commit this
directory.

Test restart persistence with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_persistence.py -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
