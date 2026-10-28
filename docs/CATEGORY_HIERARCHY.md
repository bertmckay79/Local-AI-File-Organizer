# Category and subcategory support

The organiser supports a maximum of two levels:

- **Category**: RPG, Software, Game Development, Maker & DIY, etc.
- **Subcategory**: Dungeons & Dragons, Game Master Resources, Godot, Unity, Web Development, etc.

For example, a book classified as `Game Development / Godot` is moved to:

```text
<output folder>/Game Development-AI/Godot/<filename>.pdf
```

The `-AI` suffix on the parent is intentional: the scanner recognises
those output folders and avoids processing its own generated content.

## Built-in definitions

`config/categories.json` uses optional `parent` on child categories:

```json
{
  "name": "Godot",
  "parent": "Game Development",
  "extensions": [".pdf", ".epub"],
  "keywords": ["Godot", "GDScript"]
}
```

Existing names, such as `RPG - Dungeons & Dragons`, continue to be read as
`RPG / Dungeons & Dragons`. On startup, the database upgrades existing
*file records* into the already-present `category` and `subcategory` columns.

## Creating and assigning categories

**Settings → Categories** now shows Category and Subcategory separately.
Use **Add Category / Subcategory** to create a top-level category (empty
subcategory) or a child beneath an existing or new parent.

**Analyze** now offers three actions for the selected result:

1. **Accept Suggested Category** — only if a category is proposed for a
   `Needs Review` file.
2. **Assign Existing** — choose any configured category/subcategory.
3. **New Category / Subcategory** — create a user-defined category,
   specify subject keywords and optionally assign the selected file.

A **suggestion never triggers a move**. Once a user approves a category,
the application's database is updated and the Organize preview is refreshed.
The standard organize confirmation still applies.

Custom categories are stored in the existing SQLite `custom_categories`
table, using the canonical `Parent / Child` name. No extra
table or manual schema migration is required.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Use a copy of your sample books to check the *Analyze → review → Organize*
workflow before processing your actual collection. Older output folders are
not moved automatically; a future organize operation uses the new layout.

To manually trigger the GitHub test workflow, open **Actions → Run tests →
Run workflow**, and choose your feature branch.
