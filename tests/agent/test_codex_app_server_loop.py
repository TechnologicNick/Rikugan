"""Tests for the native Codex App Server loop."""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.mocks.ida_mock import install_ida_mocks

install_ida_mocks()

from rikugan.agent.codex_loop import AppServerAgentLoop
from rikugan.agent.turn import TurnEventType
from rikugan.core.config import RikuganConfig
from rikugan.core.types import ProviderCapabilities
from rikugan.state.session import SessionState
from rikugan.tools.base import tool
from rikugan.tools.registry import ToolRegistry


class _FakeNativeProvider:
    def __init__(self, events: list[dict[str, Any]]):
        self.model = "gpt-5.4-mini"
        self.capabilities = ProviderCapabilities(native_turn_protocol=True)
        self._events = list(events)
        self.replies: list[tuple[int, Any]] = []
        self.turn_calls: list[dict[str, Any]] = []

    def drain_events(self) -> None:
        return

    def ensure_thread(
        self,
        thread_id: str,
        *,
        cwd: str,
        developer_instructions: str,
        dynamic_tools: list[dict[str, Any]],
    ) -> str:
        self.dynamic_tools = dynamic_tools
        self.developer_instructions = developer_instructions
        return "thr_test"

    def start_turn(self, *, thread_id: str, text: str, cwd: str, plan_mode: bool = False) -> str:
        self.turn_calls.append({"thread_id": thread_id, "text": text, "cwd": cwd, "plan_mode": plan_mode})
        return "turn_test"

    def next_event(self, timeout: float = 0.5) -> dict[str, Any] | None:
        if not self._events:
            return None
        return self._events.pop(0)

    def reply_to_request(self, request_id: int, result: Any) -> None:
        self.replies.append((request_id, result))

    def interrupt_turn(self, thread_id: str, turn_id: str) -> None:
        return


@tool(category="test")
def echo_text(value: str) -> str:
    """Echo a string for tests."""
    return f"echo: {value}"


class TestCodexAppServerLoop(unittest.TestCase):
    def _make_loop(self, provider: _FakeNativeProvider) -> AppServerAgentLoop:
        registry = ToolRegistry()
        registry.register_function(echo_text)
        config = RikuganConfig()
        session = SessionState(provider_name="codex_app_server", model_name="gpt-5.4-mini")
        return AppServerAgentLoop(provider, registry, config, session)

    def test_dynamic_tool_call_round_trip(self):
        provider = _FakeNativeProvider(
            [
                {
                    "method": "item/started",
                    "params": {
                        "threadId": "thr_test",
                        "turnId": "turn_test",
                        "item": {
                            "type": "dynamicToolCall",
                            "id": "call_1",
                            "tool": "echo_text",
                            "arguments": {"value": "hello"},
                        },
                    },
                },
                {
                    "method": "item/tool/call",
                    "id": 7,
                    "params": {
                        "threadId": "thr_test",
                        "turnId": "turn_test",
                        "callId": "call_1",
                        "tool": "echo_text",
                        "arguments": {"value": "hello"},
                    },
                },
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thr_test",
                        "turnId": "turn_test",
                        "itemId": "msg_1",
                        "delta": "done",
                    },
                },
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thr_test",
                        "turnId": "turn_test",
                        "tokenUsage": {
                            "last": {
                                "inputTokens": 10,
                                "outputTokens": 5,
                                "totalTokens": 15,
                                "cachedInputTokens": 2,
                            }
                        },
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thr_test",
                        "turn": {"id": "turn_test", "status": "completed", "error": None},
                    },
                },
            ]
        )
        loop = self._make_loop(provider)

        events = list(loop.run("test request"))
        event_types = [event.type for event in events]

        self.assertIn(TurnEventType.TOOL_CALL_START, event_types)
        self.assertIn(TurnEventType.TOOL_CALL_DONE, event_types)
        self.assertIn(TurnEventType.TOOL_RESULT, event_types)
        self.assertIn(TurnEventType.TEXT_DONE, event_types)
        self.assertEqual(provider.replies[0][0], 7)
        self.assertTrue(provider.replies[0][1]["success"])
        self.assertIn("echo: hello", provider.replies[0][1]["contentItems"][0]["text"])

        self.assertEqual(len(loop.session.messages), 3)
        self.assertEqual(loop.session.messages[1].tool_calls[0].name, "echo_text")
        self.assertIn("echo: hello", loop.session.messages[2].tool_results[0].content)

    def test_plan_mode_sets_plan_flag(self):
        provider = _FakeNativeProvider(
            [
                {
                    "method": "item/plan/delta",
                    "params": {
                        "threadId": "thr_test",
                        "turnId": "turn_test",
                        "itemId": "plan_1",
                        "delta": "1. Inspect code",
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thr_test",
                        "turnId": "turn_test",
                        "item": {
                            "type": "plan",
                            "id": "plan_1",
                            "text": "1. Inspect code\n2. Add tests",
                        },
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thr_test",
                        "turn": {"id": "turn_test", "status": "completed", "error": None},
                    },
                },
            ]
        )
        loop = self._make_loop(provider)

        events = list(loop.run("/plan add a provider"))
        self.assertTrue(provider.turn_calls[0]["plan_mode"])
        text_done = [event.text for event in events if event.type == TurnEventType.TEXT_DONE]
        self.assertEqual(text_done[-1], "1. Inspect code\n2. Add tests")
