"""Deletes any stale ./vie.db before test collection.

`app.core.config.settings` is a module-level singleton read once at first
import, so a per-test `monkeypatch.setenv("DATABASE_URL", ...)` only takes
effect if that import hasn't happened yet — and pytest imports every test
module's top-level statements during collection, before any fixture runs.
In practice several test modules import app.* at module scope, so the
default sqlite:///./vie.db is what actually gets used across a whole test
session. Deleting it here (this file is imported first, before any test
module) guarantees create_all() runs against a clean schema instead of an
old file missing newer columns.
"""

from pathlib import Path

Path("vie.db").unlink(missing_ok=True)
