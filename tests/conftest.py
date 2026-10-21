"""Keep normal tests independent of the parent shell and production .env."""

import os

import pytest

from knowledge_mcp import config, documents


def pytest_collection_modifyitems(config, items):
    # Read before the autouse fixture removes inherited KNOWLEDGE_* variables.
    if os.environ.get("KNOWLEDGE_RUN_REAL_MODELS") != "1":
        skip = pytest.mark.skip(reason="real models require KNOWLEDGE_RUN_REAL_MODELS=1")
        for item in items:
            if item.get_closest_marker("real_models"):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_knowledge_env(monkeypatch, request):
    for key in tuple(os.environ):
        if key.startswith("KNOWLEDGE_") and key != "KNOWLEDGE_TEST_QDRANT_URL":
            monkeypatch.delenv(key, raising=False)
    if request.node.get_closest_marker("env_loading") is None:
        monkeypatch.setattr(config, "_load_env_file", lambda *_: None)
        monkeypatch.setattr(documents, "_load_env_file", lambda *_: None)
