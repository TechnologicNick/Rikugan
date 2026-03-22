"""Tests for the Codex App Server provider."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.mocks.ida_mock import install_ida_mocks

install_ida_mocks()

from rikugan.providers.codex_app_server import CodexAppServerProvider


class _FakeTransport:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def ensure_started(self) -> None:
        return

    def request(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        self.calls.append((method, params or {}))
        return self.responses[method]

    def shutdown(self) -> None:
        return


class TestCodexAppServerProvider(unittest.TestCase):
    def test_auth_status_chatgpt(self):
        provider = CodexAppServerProvider(
            transport=_FakeTransport(
                {
                    "account/read": {
                        "result": {
                            "account": {"type": "chatgpt", "planType": "team"},
                            "requiresOpenaiAuth": True,
                        }
                    }
                }
            ),
            model="gpt-5.4",
        )
        label, status = provider.auth_status()
        self.assertEqual(label, "ChatGPT (team)")
        self.assertEqual(status, "ok")

    def test_auth_status_requires_login(self):
        provider = CodexAppServerProvider(
            transport=_FakeTransport(
                {
                    "account/read": {
                        "result": {
                            "account": None,
                            "requiresOpenaiAuth": True,
                        }
                    }
                }
            )
        )
        label, status = provider.auth_status()
        self.assertIn("codex login", label)
        self.assertEqual(status, "error")

    def test_list_models_maps_live_response(self):
        provider = CodexAppServerProvider(
            transport=_FakeTransport(
                {
                    "model/list": {
                        "result": {
                            "data": [
                                {
                                    "id": "gpt-5.4",
                                    "displayName": "gpt-5.4",
                                    "inputModalities": ["text", "image"],
                                },
                                {
                                    "id": "gpt-5.4-mini",
                                    "displayName": "GPT-5.4-Mini",
                                    "inputModalities": ["text"],
                                },
                            ]
                        }
                    }
                }
            )
        )
        models = provider.list_models()
        self.assertEqual([m.id for m in models], ["gpt-5.4", "gpt-5.4-mini"])
        self.assertTrue(models[0].supports_vision)
        self.assertFalse(models[1].supports_vision)
