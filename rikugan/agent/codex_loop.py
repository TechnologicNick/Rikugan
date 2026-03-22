"""Native App Server agent loop for Codex-backed sessions."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from typing import Any

from ..core.errors import CancellationError, ProviderError
from ..core.logging import log_error
from ..core.types import Message, Role, TokenUsage, ToolCall
from .loop import AgentLoop, _parse_user_command
from .turn import TurnEvent


class AppServerAgentLoop(AgentLoop):
    """Agent loop that drives Codex through `codex app-server`."""

    _AUTO_DECLINE_METHODS = frozenset(
        {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "apply_patch/requestApproval",
            "command/exec/requestApproval",
            "permissions/requestApproval",
        }
    )

    def _working_dir(self) -> str:
        if self.session.idb_path:
            return os.path.dirname(self.session.idb_path)
        return os.getcwd()

    @staticmethod
    def _to_dynamic_tools(tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dynamic_tools: list[dict[str, Any]] = []
        for tool in tools_schema:
            function = tool.get("function", {})
            name = function.get("name")
            if not name or name == "spawn_subagent":
                continue
            dynamic_tools.append(
                {
                    "name": name,
                    "description": function.get("description", ""),
                    "inputSchema": function.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return dynamic_tools

    @staticmethod
    def _tool_result_response(text: str) -> dict[str, Any]:
        return {
            "success": True,
            "contentItems": [
                {
                    "type": "inputText",
                    "text": text,
                }
            ],
        }

    @staticmethod
    def _map_usage(payload: dict[str, Any]) -> TokenUsage | None:
        token_usage = payload.get("tokenUsage") or {}
        usage = token_usage.get("last") or token_usage.get("total")
        if not usage:
            return None
        return TokenUsage(
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            total_tokens=usage.get("totalTokens", 0),
            cache_read_tokens=usage.get("cachedInputTokens", 0),
        )

    def _execute_dynamic_tool(
        self,
        request: dict[str, Any],
        tool_calls: list[ToolCall],
        tool_results: list[Any],
    ) -> Generator[TurnEvent, None, None]:
        params = request.get("params", {})
        tool_name = params.get("tool", "")
        call_id = params.get("callId", "")
        arguments = params.get("arguments", {}) or {}
        tool_call = ToolCall(id=call_id, name=tool_name, arguments=arguments)
        tool_calls.append(tool_call)
        yield TurnEvent.tool_call_done(call_id, tool_name, json.dumps(arguments, indent=2, default=str))

        try:
            results = yield from self._execute_tool_calls([tool_call])
            tool_result = results[0]
            tool_results.append(tool_result)
            self.provider.reply_to_request(request["id"], self._tool_result_response(tool_result.content))
        except Exception as exc:
            error_text = f"Tool bridge error: {exc}"
            log_error(error_text)
            self.provider.reply_to_request(request["id"], self._tool_result_response(error_text))
            yield TurnEvent.error_event(error_text)

    def run(self, user_message: str) -> Generator[TurnEvent, None, None]:
        self._cancelled.clear()
        self._running = True
        self.session.is_running = True

        try:
            cmd = _parse_user_command(user_message)
            if cmd.direct_command == "/memory":
                yield from self._handle_memory_command()
                return
            if cmd.direct_command == "/undo":
                yield from self._handle_undo_command(cmd.direct_arg)
                return
            if cmd.direct_command == "/mcp":
                yield from self._handle_mcp_command()
                return
            if cmd.direct_command == "/doctor":
                yield from self._handle_doctor_command()
                return

            user_message = cmd.message
            user_message, active_skill = self._resolve_skill(user_message)
            plan_mode = cmd.use_plan_mode or (active_skill is not None and active_skill.mode == "plan")

            if cmd.use_exploration_mode:
                user_message = (
                    "Investigate this binary using Rikugan's host-native tools only. "
                    "Stay read-only and focus on concrete evidence.\n\n"
                    f"User request: {user_message}"
                )
            elif cmd.use_research_mode:
                user_message = (
                    "Investigate this target thoroughly using Rikugan's host-native tools only. "
                    "Summarize concrete findings clearly.\n\n"
                    f"User request: {user_message}"
                )

            self.session.add_message(Message(role=Role.USER, content=user_message))
            system_prompt = self._build_system_prompt()
            tools_schema = self._build_tools_schema(active_skill, False, False)
            dynamic_tools = self._to_dynamic_tools(tools_schema)
            cwd = self._working_dir()

            self.provider.drain_events()
            thread_id = self.provider.ensure_thread(
                self.session.metadata.get("codex_thread_id", ""),
                cwd=cwd,
                developer_instructions=system_prompt,
                dynamic_tools=dynamic_tools,
            )
            self.session.metadata["codex_thread_id"] = thread_id
            self.session.metadata["codex_auth_mode"] = "external"

            turn_id = self.provider.start_turn(
                thread_id=thread_id,
                text=user_message,
                cwd=cwd,
                plan_mode=plan_mode,
            )

            self.session.current_turn += 1
            yield TurnEvent.turn_start(self.session.current_turn)

            assistant_text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            tool_results: list[Any] = []
            last_usage: TokenUsage | None = None

            while True:
                try:
                    self._check_cancelled()
                except CancellationError:
                    self.provider.interrupt_turn(thread_id, turn_id)
                    raise

                message = self.provider.next_event(timeout=0.5)
                if message is None:
                    continue

                method = message.get("method", "")
                params = message.get("params", {})
                event_thread_id = params.get("threadId")
                event_turn_id = params.get("turnId")
                if event_thread_id and event_thread_id != thread_id:
                    continue
                if event_turn_id and event_turn_id != turn_id:
                    continue

                if method == "item/agentMessage/delta":
                    delta = params.get("delta", "")
                    if delta:
                        assistant_text_parts.append(delta)
                        yield TurnEvent.text_delta(delta)
                    continue

                if method == "item/plan/delta":
                    delta = params.get("delta", "")
                    if delta:
                        assistant_text_parts.append(delta)
                        yield TurnEvent.text_delta(delta)
                    continue

                if method == "item/started":
                    item = params.get("item", {})
                    if item.get("type") == "dynamicToolCall":
                        yield TurnEvent.tool_call_start(item.get("id", ""), item.get("tool", ""))
                    continue

                if method == "item/completed":
                    item = params.get("item", {})
                    if item.get("type") == "plan" and item.get("text"):
                        assistant_text_parts = [item["text"]]
                    continue

                if method == "thread/tokenUsage/updated":
                    usage = self._map_usage(params)
                    if usage is not None:
                        last_usage = usage
                    continue

                if method == "item/tool/call":
                    yield from self._execute_dynamic_tool(message, tool_calls, tool_results)
                    continue

                if method in self._AUTO_DECLINE_METHODS:
                    self.provider.reply_to_request(message["id"], "decline")
                    yield TurnEvent.error_event(
                        "Codex requested shell or file-change approval. Rikugan declined it and kept the turn in host-tool mode."
                    )
                    continue

                if method == "item/tool/requestUserInput":
                    self.provider.reply_to_request(message["id"], {"answers": {}})
                    yield TurnEvent.error_event("Codex requested unsupported external app input.")
                    continue

                if method == "turn/completed":
                    turn = params.get("turn", {})
                    status = turn.get("status")
                    if status == "failed":
                        error = (turn.get("error") or {}).get("message", "Codex turn failed")
                        yield TurnEvent.error_event(error)
                        return
                    if status == "interrupted":
                        raise CancellationError("Agent run cancelled")
                    break

            assistant_text = "".join(assistant_text_parts)
            if assistant_text:
                yield TurnEvent.text_done(assistant_text)

            self.session.add_message(
                Message(
                    role=Role.ASSISTANT,
                    content=assistant_text,
                    tool_calls=tool_calls,
                    token_usage=last_usage,
                )
            )
            if tool_results:
                self.session.add_message(Message(role=Role.TOOL, tool_results=tool_results))
            if last_usage is not None:
                yield TurnEvent.usage_update(last_usage)

            yield TurnEvent.turn_end(self.session.current_turn)

        except CancellationError:
            yield TurnEvent.cancelled_event()
        except ProviderError as exc:
            yield TurnEvent.error_event(str(exc))
        except Exception as exc:
            log_error(f"Codex App Server loop error: {exc}")
            yield TurnEvent.error_event(str(exc))
        finally:
            self._running = False
            self.session.is_running = False
