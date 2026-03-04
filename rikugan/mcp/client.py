"""MCP client: manages a single MCP server subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import queue
from typing import Any, Dict, List, Optional

from ..constants import MCP_DEFAULT_TIMEOUT
from ..core.errors import MCPConnectionError, MCPError, MCPTimeoutError
from ..core.logging import log_debug, log_error, log_info, log_trace
from .config import MCPServerConfig
from .protocol import (
    MCPToolSchema,
    encode_jsonrpc_request,
    decode_jsonrpc_response,
    parse_content_length_frame,
)


_HEARTBEAT_INTERVAL = 30.0  # seconds between heartbeat pings


class MCPClient:
    """Client for a single MCP server process.

    Communicates via stdio with JSON-RPC 2.0 + Content-Length framing.
    No asyncio — pure threading + subprocess + queue.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.name = config.name
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._pending: Dict[int, queue.Queue] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._tools: List[MCPToolSchema] = []
        self._running = False
        self._started = False  # True only after successful handshake
        self._healthy = True  # False if heartbeat fails

    @property
    def is_running(self) -> bool:
        return (
            self._started
            and self._running
            and self._process is not None
            and self._process.poll() is None
        )

    @property
    def is_healthy(self) -> bool:
        """True if the server is running and the last heartbeat succeeded."""
        return self.is_running and self._healthy

    def start(self, timeout: float = 10.0) -> None:
        """Spawn the MCP server process, perform initialize handshake, and list tools."""
        log_info(f"MCP[{self.name}]: starting server: {self.config.command} {self.config.args}")

        env = os.environ.copy()
        env.update(self.config.env)

        try:
            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,  # isolate from parent signals
            )
        except (OSError, FileNotFoundError) as e:
            raise MCPConnectionError(f"MCP[{self.name}]: failed to start: {e}")

        self._running = True

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"mcp-reader-{self.name}",
        )
        self._reader_thread.start()

        # Initialize handshake
        try:
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rikugan", "version": "0.1.0"},
            }, timeout=timeout)
            log_debug(f"MCP[{self.name}]: initialize response: {json.dumps(result)[:200]}")

            # Send initialized notification (no id, no response expected)
            self._send_notification("notifications/initialized")

        except Exception as e:
            self.stop()
            raise MCPConnectionError(f"MCP[{self.name}]: handshake failed: {e}")

        # Discover tools
        try:
            tools_result = self._send_request("tools/list", {}, timeout=timeout)
            raw_tools = tools_result.get("tools", []) if isinstance(tools_result, dict) else []
            self._tools = []
            for t in raw_tools:
                self._tools.append(MCPToolSchema(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                ))
            log_info(f"MCP[{self.name}]: discovered {len(self._tools)} tools")
        except Exception as e:
            log_error(f"MCP[{self.name}]: tools/list failed: {e}")
            # Non-fatal — server may have no tools yet

        self._started = True

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True,
            name=f"mcp-heartbeat-{self.name}",
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Shut down the MCP server process."""
        log_debug(f"MCP[{self.name}]: stopping")
        self._running = False

        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except OSError:
                log_debug(f"MCP[{self.name}]: stdin already closed")
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log_debug(f"MCP[{self.name}]: terminate timed out, killing")
                self._process.kill()
                self._process.wait(timeout=2)
            except OSError as e:
                log_debug(f"MCP[{self.name}]: stop error: {e}")
            self._process = None

        # Unblock any pending requests
        with self._lock:
            for q in self._pending.values():
                q.put({"error": {"code": -1, "message": "Server stopped"}})
            self._pending.clear()

    def get_tools(self) -> List[MCPToolSchema]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: Dict[str, Any], timeout: float = MCP_DEFAULT_TIMEOUT) -> str:
        """Call an MCP tool and return the result as a string."""
        log_debug(f"MCP[{self.name}]: calling tool {name}")
        result = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        }, timeout=timeout)

        if isinstance(result, dict) and "error" in result:
            raise MCPError(f"MCP tool {name} error: {result['error']}")

        # Extract text content from MCP tool result
        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        else:
                            parts.append(json.dumps(item, default=str))
                    else:
                        parts.append(str(item))
                return "\n".join(parts) if parts else json.dumps(result, default=str)
            return json.dumps(result, default=str)

        return str(result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send_request(self, method: str, params: Dict[str, Any], timeout: float = MCP_DEFAULT_TIMEOUT) -> Any:
        """Send a JSON-RPC request and block for the response."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError(f"MCP[{self.name}]: not connected")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            resp_queue: queue.Queue = queue.Queue()
            self._pending[req_id] = resp_queue

        data = encode_jsonrpc_request(method, params, id=req_id)
        log_trace(f"MCP[{self.name}]: sending {method} id={req_id}")

        try:
            self._process.stdin.write(data)
            self._process.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            with self._lock:
                self._pending.pop(req_id, None)
            raise MCPConnectionError(f"MCP[{self.name}]: write failed: {e}")

        # Wait for response
        try:
            response = resp_queue.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(req_id, None)
            raise MCPTimeoutError(f"MCP[{self.name}]: {method} timed out after {timeout}s")

        if "error" in response:
            err = response["error"]
            raise MCPError(f"MCP[{self.name}]: {method} error: {err.get('message', err)}")

        return response.get("result", response)

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no id, no response)."""
        if not self._process or not self._process.stdin:
            return

        body: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            body["params"] = params

        payload = json.dumps(body)
        frame = f"Content-Length: {len(payload)}\r\n\r\n{payload}"

        try:
            self._process.stdin.write(frame.encode("utf-8"))
            self._process.stdin.flush()
        except OSError as e:
            log_debug(f"MCP[{self.name}]: notification write failed: {e}")

    def _heartbeat_loop(self) -> None:
        """Background thread: periodically ping the server to detect dead processes."""
        while self._running:
            time.sleep(_HEARTBEAT_INTERVAL)
            if not self._running:
                break
            try:
                # Use a short timeout for the heartbeat ping
                self._send_request("ping", {}, timeout=5.0)
                if not self._healthy:
                    log_info(f"MCP[{self.name}]: heartbeat recovered")
                self._healthy = True
            except MCPTimeoutError:
                if self._healthy:
                    log_error(f"MCP[{self.name}]: heartbeat timed out — server may be unresponsive")
                self._healthy = False
            except (MCPConnectionError, MCPError):
                if self._healthy:
                    log_error(f"MCP[{self.name}]: heartbeat failed — server may be dead")
                self._healthy = False
            except Exception:
                # Don't crash the heartbeat thread on unexpected errors
                self._healthy = False

    def _read_loop(self) -> None:
        """Background thread: read JSON-RPC responses from stdout."""
        log_trace(f"MCP[{self.name}]: reader started")
        try:
            while self._running and self._process and self._process.poll() is None:
                body = parse_content_length_frame(self._process.stdout)
                if body is None:
                    break

                response = decode_jsonrpc_response(body)
                resp_id = response.get("id")

                if resp_id is not None:
                    with self._lock:
                        q = self._pending.pop(resp_id, None)
                    if q:
                        q.put(response)
                    else:
                        log_trace(f"MCP[{self.name}]: no pending for id={resp_id}")
                else:
                    # Notification from server — log and ignore
                    log_trace(f"MCP[{self.name}]: server notification: {body[:200]}")

        except Exception as e:
            if self._running:
                log_error(f"MCP[{self.name}]: reader error: {e}")
        finally:
            log_trace(f"MCP[{self.name}]: reader stopped")
