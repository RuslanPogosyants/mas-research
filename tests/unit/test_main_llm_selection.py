"""Unit tests: the lifespan selects the real GigaChat adapter only when
credentials are configured, and the in-process fake otherwise.

The real-credentials branch is exercised with a stub adapter so the test never
needs the optional `gigachat` package (absent in CI); the real adapter itself is
validated against the live API separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import src.adapters.gigachat as gigachat_module
from src.adapters.llm import FakeLlmAdapter
from src.config import Settings
from src.main import _build_llm

if TYPE_CHECKING:
    import pytest


def test_build_llm_returns_fake_without_credentials() -> None:
    assert isinstance(_build_llm(Settings(gigachat_credentials="")), FakeLlmAdapter)


def test_build_llm_selects_gigachat_when_credentials_present(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: dict[str, object] = {}

    class _StubAdapter:
        def __init__(self, settings: Settings) -> None:
            constructed["settings"] = settings

        async def complete(self, *, system: str, user: str) -> str:
            return ""

    monkeypatch.setattr(gigachat_module, "GigaChatAdapter", _StubAdapter)
    result = _build_llm(Settings(gigachat_credentials="some-key"))
    assert isinstance(result, _StubAdapter)
    assert constructed  # built from the provided settings
