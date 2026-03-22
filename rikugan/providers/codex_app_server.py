"""Codex App Server provider backed by the local `codex app-server` CLI."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from collections.abc import Generator
from typing import Any, NoReturn

from ..core.errors import AuthenticationError, ProviderError
from ..core.logging import log_debug
from ..core.types import Message, ModelInfo, ProviderCapabilities, StreamChunk
from .base import LLMProvider

_DEFAULT_MODEL = "gpt-5.4"
_DEFAULT_TIMEOUT = 30.0


class CodexAppServerTransport:
    """Persistent JSON-RPC stdio transport for `codex app-server`."""

    def __init__(self, command: str = "") -> None:
        self._command = command or shutil.which("codex.cmd") or shutil.which("codex") or "codex"
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._incoming: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._initialized = False

    def ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        try:
            self._proc = subprocess.Popen(
                [self._command, "app-server", "--session-source", "rikugan"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            raise ProviderError(
                "Failed to start `codex app-server`. Ensure Codex CLI is installed and on PATH.",
                provider="codex_app_server",
            ) from exc

        self._initialized = False
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True, name="codex-app-server-stdout")
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True, name="codex-app-server-stderr")
        self._stderr_thread.start()
        self._initialize()

    def _initialize(self) -> None:
        if self._initialized:
            return
        response = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "rikugan",
                    "title": "Rikugan",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                },
            },
            timeout=10.0,
        )
        if "error" in response:
            raise ProviderError(
                response["error"].get("message", "Failed to initialize `codex app-server`"),
                provider="codex_app_server",
            )
        self.notify("initialized", {})
        self._initialized = True

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log_debug(f"codex app-server emitted malformed JSON: {line[:200]}")
                continue

            msg_id = message.get("id")
            with self._lock:
                pending = self._pending.get(msg_id) if msg_id is not None else None
            if pending is not None:
                pending.put(message)
            else:
                self._incoming.put(message)

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for raw_line in self._proc.stderr:
            line = raw_line.strip()
            if line:
                log_debug(f"codex app-server stderr: {line}")

    def _send(self, payload: dict[str, Any]) -> None:
        self.ensure_started()
        assert self._proc is not None and self._proc.stdin is not None
        message = json.dumps(payload, separators=(",", ":"))
        with self._write_lock:
            self._proc.stdin.write(message + "\n")
            self._proc.stdin.flush()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        self.ensure_started()
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            waiter: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[request_id] = waiter
        self._send({"method": method, "id": request_id, "params": params or {}})
        try:
            return waiter.get(timeout=timeout)
        except queue.Empty as exc:
            raise ProviderError(
                f"`codex app-server` timed out waiting for {method}",
                provider="codex_app_server",
            ) from exc
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"method": method, "params": params or {}})

    def reply(self, request_id: int, result: Any) -> None:
        self._send({"id": request_id, "result": result})

    def incoming_event(self, timeout: float = 0.5) -> dict[str, Any] | None:
        try:
            return self._incoming.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self) -> None:
        while True:
            try:
                self._incoming.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        proc = self._proc
        self._proc = None
        self._initialized = False
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


class CodexAppServerProvider(LLMProvider):
    """Provider adapter that talks to the local Codex App Server."""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        model: str = _DEFAULT_MODEL,
        transport: CodexAppServerTransport | None = None,
        codex_command: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key="", api_base="", model=model or _DEFAULT_MODEL)
        self._transport = transport
        self._codex_command = codex_command
        self._last_account: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "codex_app_server"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tool_use=True,
            vision=True,
            max_context_window=258400,
            max_output_tokens=32768,
            supports_system_prompt=True,
            native_turn_protocol=True,
        )

    def close(self) -> None:
        if self._transport is not None:
            self._transport.shutdown()

    def _get_client(self) -> CodexAppServerTransport:
        if self._transport is None:
            self._transport = CodexAppServerTransport(command=self._codex_command)
        self._transport.ensure_started()
        return self._transport

    def _fetch_account(self) -> dict[str, Any]:
        client = self._get_client()
        response = client.request("account/read", {"refreshToken": False})
        if "error" in response:
            raise ProviderError(
                response["error"].get("message", "Failed to read Codex account status"),
                provider=self.name,
            )
        result = response.get("result", {})
        self._last_account = result.get("account")
        return result

    def auth_status(self) -> tuple[str, str]:
        try:
            result = self._fetch_account()
        except Exception as exc:
            log_debug(f"codex app-server auth status failed: {exc}")
            return "Run `codex login`", "error"

        account = result.get("account")
        if account:
            account_type = account.get("type")
            if account_type == "chatgpt":
                plan = account.get("planType")
                return (f"ChatGPT ({plan})" if plan else "ChatGPT", "ok")
            if account_type == "apiKey":
                return "Codex API Key", "ok"
            return str(account_type or "Codex"), "ok"
        if result.get("requiresOpenaiAuth"):
            return "Run `codex login`", "error"
        return "Ready", "ok"

    def validate_key(self) -> bool:
        _label, status = self.auth_status()
        return status == "ok"

    def _fetch_models_live(self) -> list[ModelInfo]:
        client = self._get_client()
        response = client.request("model/list", {"limit": 100, "includeHidden": False})
        if "error" in response:
            raise ProviderError(
                response["error"].get("message", "Failed to list Codex models"),
                provider=self.name,
            )

        models = []
        for entry in response.get("result", {}).get("data", []):
            model_id = entry.get("id") or entry.get("model")
            if not model_id:
                continue
            modalities = set(entry.get("inputModalities", []))
            models.append(
                ModelInfo(
                    id=model_id,
                    name=entry.get("displayName") or model_id,
                    provider=self.name,
                    context_window=self.capabilities.max_context_window,
                    max_output_tokens=self.capabilities.max_output_tokens,
                    supports_tools=True,
                    supports_vision="image" in modalities,
                )
            )
        return models or self._builtin_models()

    @staticmethod
    def _builtin_models() -> list[ModelInfo]:
        return [
            ModelInfo("gpt-5.4", "gpt-5.4", "codex_app_server", 258400, 32768, True, True),
            ModelInfo("gpt-5.4-mini", "GPT-5.4-Mini", "codex_app_server", 258400, 32768, True, True),
        ]

    def ensure_thread(
        self,
        thread_id: str,
        *,
        cwd: str,
        developer_instructions: str,
        dynamic_tools: list[dict[str, Any]],
    ) -> str:
        client = self._get_client()
        if thread_id:
            response = client.request(
                "thread/resume",
                {
                    "threadId": thread_id,
                    "model": self.model,
                    "cwd": cwd,
                },
            )
            if "error" not in response:
                return thread_id

        response = client.request(
            "thread/start",
            {
                "model": self.model,
                "cwd": cwd,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "developerInstructions": developer_instructions,
                "dynamicTools": dynamic_tools,
            },
        )
        if "error" in response:
            self._raise_rpc_error(response["error"])
        return response["result"]["thread"]["id"]

    def fork_thread(self, thread_id: str) -> str:
        response = self._get_client().request("thread/fork", {"threadId": thread_id})
        if "error" in response:
            self._raise_rpc_error(response["error"])
        return response["result"]["thread"]["id"]

    def start_turn(
        self,
        *,
        thread_id: str,
        text: str,
        cwd: str,
        plan_mode: bool = False,
    ) -> str:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": text}],
            "cwd": cwd,
            "model": self.model,
            "approvalPolicy": "never",
            "sandboxPolicy": {
                "type": "readOnly",
                "access": {"type": "fullAccess"},
            },
        }
        if plan_mode:
            params["collaborationMode"] = {
                "mode": "plan",
                "settings": {
                    "model": self.model,
                    "developer_instructions": None,
                },
            }

        response = self._get_client().request("turn/start", params)
        if "error" in response:
            self._raise_rpc_error(response["error"])
        return response["result"]["turn"]["id"]

    def interrupt_turn(self, thread_id: str, turn_id: str) -> None:
        response = self._get_client().request(
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
            timeout=5.0,
        )
        if "error" in response:
            log_debug(f"turn/interrupt failed: {response['error']}")

    def next_event(self, timeout: float = 0.5) -> dict[str, Any] | None:
        return self._get_client().incoming_event(timeout=timeout)

    def drain_events(self) -> None:
        self._get_client().drain_events()

    def reply_to_request(self, request_id: int, result: Any) -> None:
        self._get_client().reply(request_id, result)

    @staticmethod
    def _raise_rpc_error(error: dict[str, Any]) -> NoReturn:
        message = error.get("message", "Codex App Server error")
        if error.get("code") == 401:
            raise AuthenticationError("Run `codex login` to authenticate Codex.", provider="codex_app_server")
        raise ProviderError(message, provider="codex_app_server")

    # Native-turn providers do not use the chat/completions pipeline.
    def _format_messages(self, messages: list[Message]) -> Any:
        raise ProviderError("codex_app_server uses the native turn protocol", provider=self.name)

    def _build_request_kwargs(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        system: str,
    ) -> dict[str, Any]:
        raise ProviderError("codex_app_server uses the native turn protocol", provider=self.name)

    def _call_api(self, client: Any, kwargs: dict[str, Any]) -> Any:
        raise ProviderError("codex_app_server uses the native turn protocol", provider=self.name)

    def _normalize_response(self, raw: Any) -> Message:
        raise ProviderError("codex_app_server uses the native turn protocol", provider=self.name)

    def _handle_api_error(self, e: Exception) -> NoReturn:
        raise ProviderError(str(e), provider=self.name) from e

    def _stream_chunks(
        self,
        client: Any,
        kwargs: dict[str, Any],
    ) -> Generator[StreamChunk, None, None]:
        raise ProviderError("codex_app_server uses the native turn protocol", provider=self.name)
