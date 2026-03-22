"""Tests for rikugan.ui.chat_view — pure logic helpers."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

from tests.qt_stubs import ensure_pyside6_stubs
ensure_pyside6_stubs()

# Force import of the real chat_view module under Qt stubs.
sys.modules.pop("rikugan.ui.chat_view", None)

from rikugan.ui import chat_view as chat_view_mod  # noqa: E402
from rikugan.ui.chat_view import ChatView, _is_hidden_system_user_message, _TOOL_GROUP_MIN_CALLS  # noqa: E402


# ---------------------------------------------------------------------------
# _is_hidden_system_user_message
# ---------------------------------------------------------------------------

class TestIsHiddenSystemUserMessage(unittest.TestCase):
    def test_empty_string_returns_false(self):
        self.assertFalse(_is_hidden_system_user_message(""))

    def test_none_equivalent_empty_returns_false(self):
        self.assertFalse(_is_hidden_system_user_message(""))

    def test_system_prefix_returns_true(self):
        self.assertTrue(_is_hidden_system_user_message("[SYSTEM] some hint"))

    def test_system_prefix_with_leading_whitespace(self):
        self.assertTrue(_is_hidden_system_user_message("   [SYSTEM] some hint"))

    def test_regular_message_returns_false(self):
        self.assertFalse(_is_hidden_system_user_message("Hello world"))

    def test_lowercase_system_returns_false(self):
        self.assertFalse(_is_hidden_system_user_message("[system] hint"))

    def test_partial_system_keyword_returns_false(self):
        self.assertFalse(_is_hidden_system_user_message("SYSTEM"))

    def test_system_in_middle_returns_false(self):
        self.assertFalse(_is_hidden_system_user_message("not [SYSTEM] hint"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestChatViewConstants(unittest.TestCase):
    def test_tool_group_min_calls_is_positive(self):
        self.assertGreater(_TOOL_GROUP_MIN_CALLS, 0)

    def test_tool_group_min_calls_value(self):
        self.assertEqual(_TOOL_GROUP_MIN_CALLS, 2)


class TestChatViewToolBoundaries(unittest.TestCase):
    def test_tool_call_start_clears_current_assistant(self):
        view = object.__new__(ChatView)
        view._current_assistant = MagicMock()
        view._tool_widgets = {}
        view._register_tool_widget = MagicMock()
        view._scroll_to_bottom = MagicMock()
        view._hide_thinking = MagicMock()

        tool_widget = MagicMock()
        original = chat_view_mod.ToolCallWidget
        chat_view_mod.ToolCallWidget = MagicMock(return_value=tool_widget)
        try:
            event = types.SimpleNamespace(
                type=chat_view_mod.TurnEventType.TOOL_CALL_START,
                tool_name="echo_text",
                tool_call_id="call_1",
            )
            ChatView._handle_tool_event(view, event)
        finally:
            chat_view_mod.ToolCallWidget = original

        self.assertIsNone(view._current_assistant)
        self.assertIs(view._tool_widgets["call_1"], tool_widget)
        view._register_tool_widget.assert_called_once_with("echo_text", "call_1", tool_widget)


if __name__ == "__main__":
    unittest.main()
