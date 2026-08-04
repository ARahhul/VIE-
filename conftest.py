"""Test-session setup, run before any test module is imported.

`app.core.config.settings` is a module-level singleton read once at first
import, so a per-test `monkeypatch.setenv(...)` only takes effect if that
import hasn't happened yet — and pytest imports every test module's
top-level statements during collection, before any fixture runs. In
practice several test modules import app.* at module scope, so whatever
this file does first is what the whole session gets.

Two things need forcing here:
1. A stale ./vie.db (missing newer columns from an older schema) would
   otherwise make create_all() a no-op against the existing file.
2. A real .env with live API credentials (video-LLM backend, LangSmith
   tracing) must not leak into the test session — real .env is for running
   the app, not for pytest making live network calls on every run. Setting
   these in os.environ (not .env) takes precedence over .env in
   pydantic-settings, so this reliably overrides it regardless of what's
   configured for actual runs.
"""

import os
from pathlib import Path

os.environ["VIDEO_LLM_BACKEND"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

Path("vie.db").unlink(missing_ok=True)
