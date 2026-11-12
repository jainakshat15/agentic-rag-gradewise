"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/conftest.py: every test starts from the defaults           │
│                                                                      │
│  What: whatever YOUR .env says (MODEL_MODE=free, DATA_SIZE=small…),  │
│        the tests run as MODEL_MODE=cheap + DATA_SIZE=full, so they   │
│        give the same answer on every machine. A test that wants a    │
│        mode sets it itself: monkeypatch.setattr(settings, ...).      │
│  💡 Session-scoped, so module-scoped fixtures see the defaults too.  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import pytest

from gradewise.config import settings


@pytest.fixture(autouse=True, scope="session")
def _default_modes():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "model_mode", "cheap")
        mp.setattr(settings, "data_size", "full")
        yield


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls a real provider API (opt-in: GRADEWISE_LIVE_TESTS=1 + a key)")
