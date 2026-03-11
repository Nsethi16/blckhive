"""
MCP (Model Context Protocol) server management.

MCPManager handles:
  - Persisting server configurations to ~/.blckhive/mcp_servers.json
  - Connecting to servers via stdio at the start of each agent run
  - Exposing server tools into the ToolRegistry under a namespaced prefix
  - Installing npm/pip-based MCP servers on demand

When the user tells the agent "add the filesystem MCP server", the agent
calls the built-in `add_mcp_server` tool, which MCPManager persists.
On every future run those servers are automatically connected.

MCP servers communicate over stdin/stdout using JSON-RPC 2.0.
We implement a minimal client here to avoid hard dependencies, but if the
`mcp` package is installed we use it instead.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolRegistry
    from .config import AgentConfig


@dataclass
class MCPServerConfig:
    """Persisted configuration for one MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True


class MCPManager:
    """
    Manages MCP server lifecycle and tool injection.

    Usage:
        mcp = MCPManager(config)
        mcp.attach_to_registry(registry)   # called by run_agent
        ...run...
        mcp.detach_from_registry(registry) # cleanup
    """

    _NAMESPACE_PREFIX = "mcp__"

    def __init__(self, config: "AgentConfig") -> None:
        self._config = config
        self._servers: dict[str, MCPServerConfig] = {}
        self._injected_names: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load server configs from disk."""
        path = self._config.mcp_config_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for name, cfg in data.items():
                self._servers[name] = MCPServerConfig(**{**cfg, "name": name})
        except Exception as exc:  # noqa: BLE001
            print(f"[mcp] Warning: failed to load {path}: {exc}")

    def _save(self) -> None:
        """Persist server configs to disk."""
        path = self._config.mcp_config_path
        data = {
            name: {k: v for k, v in asdict(srv).items() if k != "name"}
            for name, srv in self._servers.items()
        }
        path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Server management API (used by the add_mcp_server built-in tool)
    # ------------------------------------------------------------------

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        description: str = "",
    ) -> str:
        """Add a new MCP server and persist it."""
        self._servers[name] = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            description=description,
            enabled=True,
        )
        self._save()
        return f"MCP server '{name}' saved. It will be available in the next agent run."

    def remove_server(self, name: str) -> str:
        """Remove an MCP server."""
        if name not in self._servers:
            return f"MCP server '{name}' not found."
        del self._servers[name]
        self._save()
        return f"MCP server '{name}' removed."

    def enable_server(self, name: str, enabled: bool = True) -> str:
        if name not in self._servers:
            return f"MCP server '{name}' not found."
        self._servers[name].enabled = enabled
        self._save()
        state = "enabled" if enabled else "disabled"
        return f"MCP server '{name}' {state}."

    def list_servers(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "command": f"{s.command} {' '.join(s.args)}".strip(),
                "description": s.description,
                "enabled": s.enabled,
            }
            for s in self._servers.values()
        ]

    # ------------------------------------------------------------------
    # Tool injection into ToolRegistry
    # ------------------------------------------------------------------

    def attach_to_registry(self, registry: "ToolRegistry") -> None:
        """
        Connect to each enabled MCP server and inject its tools.
        Called at the start of every run_agent() call.
        """
        from .tools import ToolDef

        self._injected_names = []

        for srv in self._servers.values():
            if not srv.enabled:
                continue
            try:
                tools = _fetch_mcp_tools(srv)
                for tool_spec in tools:
                    tool_name = f"{self._NAMESPACE_PREFIX}{srv.name}__{tool_spec['name']}"
                    mcp_tool_name = tool_spec["name"]
                    srv_ref = srv  # capture for closure

                    def _make_executor(s: MCPServerConfig, tn: str):
                        def executor(**kwargs: object) -> str:
                            return _call_mcp_tool(s, tn, kwargs)

                        return executor

                    registry.register_tool(
                        ToolDef(
                            name=tool_name,
                            description=(
                                f"[MCP:{srv.name}] "
                                + tool_spec.get("description", "")
                            ),
                            parameters=tool_spec.get(
                                "inputSchema",
                                {"type": "object", "properties": {}, "required": []},
                            ),
                            func=_make_executor(srv_ref, mcp_tool_name),
                        )
                    )
                    self._injected_names.append(tool_name)

                if tools:
                    print(
                        f"[mcp] Connected '{srv.name}': "
                        f"{len(tools)} tool(s) injected."
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[mcp] Warning: could not connect to '{srv.name}': {exc}")

        # Inject management tools
        self._inject_management_tools(registry)

    def detach_from_registry(self, registry: "ToolRegistry") -> None:
        """Remove injected MCP tools from the registry."""
        for name in self._injected_names:
            registry.unregister(name)
        self._injected_names = []

    def _inject_management_tools(self, registry: "ToolRegistry") -> None:
        """Inject tools that let the agent manage MCP servers at runtime."""
        from .tools import ToolDef

        mcp_ref = self

        def _add_mcp_server(
            name: str,
            command: str,
            args: str = "",
            env_json: str = "{}",
            description: str = "",
        ) -> str:
            args_list = [a for a in args.split() if a]
            try:
                env_dict = json.loads(env_json)
            except json.JSONDecodeError:
                env_dict = {}
            return mcp_ref.add_server(
                name=name,
                command=command,
                args=args_list,
                env=env_dict,
                description=description,
            )

        registry.register_tool(
            ToolDef(
                name="add_mcp_server",
                description=(
                    "Add a new MCP server so it is available in future agent runs. "
                    "For npm-based servers use command='npx' and args like "
                    "'-y @modelcontextprotocol/server-filesystem /tmp'. "
                    "For pip-based servers use the installed entry-point command."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Unique name for this MCP server.",
                        },
                        "command": {
                            "type": "string",
                            "description": "Command to launch the server (e.g. 'npx', 'uvx', 'python').",
                        },
                        "args": {
                            "type": "string",
                            "description": "Space-separated arguments for the command.",
                        },
                        "env_json": {
                            "type": "string",
                            "description": 'JSON object of environment variables, e.g. {"KEY":"value"}.',
                        },
                        "description": {
                            "type": "string",
                            "description": "Human-readable description of what this server provides.",
                        },
                    },
                    "required": ["name", "command"],
                },
                func=_add_mcp_server,
            )
        )
        self._injected_names.append("add_mcp_server")

        def _remove_mcp_server(name: str) -> str:
            return mcp_ref.remove_server(name)

        registry.register_tool(
            ToolDef(
                name="remove_mcp_server",
                description="Remove a previously added MCP server.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Server name to remove."},
                    },
                    "required": ["name"],
                },
                func=_remove_mcp_server,
            )
        )
        self._injected_names.append("remove_mcp_server")

        def _list_mcp_servers() -> str:
            servers = mcp_ref.list_servers()
            if not servers:
                return "No MCP servers configured."
            lines = []
            for s in servers:
                state = "enabled" if s["enabled"] else "disabled"
                lines.append(
                    f"  [{state}] {s['name']}: {s['command']}"
                    + (f" — {s['description']}" if s["description"] else "")
                )
            return "Configured MCP servers:\n" + "\n".join(lines)

        registry.register_tool(
            ToolDef(
                name="list_mcp_servers",
                description="List all configured MCP servers and their status.",
                parameters={"type": "object", "properties": {}, "required": []},
                func=_list_mcp_servers,
            )
        )
        self._injected_names.append("list_mcp_servers")


# ---------------------------------------------------------------------------
# Minimal MCP stdio client
# ---------------------------------------------------------------------------

def _fetch_mcp_tools(srv: MCPServerConfig, timeout: float = 15.0) -> list[dict]:
    """
    Launch an MCP server, initialize it, list its tools, and shut it down.
    Returns a list of tool specs (name, description, inputSchema).

    Uses the `mcp` package if available, otherwise falls back to raw JSON-RPC.
    """
    # Try the official mcp package first
    try:
        return _fetch_tools_via_mcp_package(srv, timeout)
    except ImportError:
        pass

    # Fallback: raw JSON-RPC over stdio
    return _fetch_tools_raw(srv, timeout)


def _fetch_tools_via_mcp_package(
    srv: MCPServerConfig, timeout: float
) -> list[dict]:
    """Use the mcp Python package to list tools."""
    import asyncio
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
    from mcp.client.stdio import stdio_client  # type: ignore[import]

    params = StdioServerParameters(
        command=srv.command,
        args=srv.args,
        env={**{}, **srv.env} if srv.env else None,
    )

    async def _list() -> list[dict]:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema.model_dump()
                        if hasattr(t.inputSchema, "model_dump")
                        else (t.inputSchema if isinstance(t.inputSchema, dict) else {}),
                    }
                    for t in result.tools
                ]

    return asyncio.run(_list())


def _call_mcp_tool_via_package(
    srv: MCPServerConfig, tool_name: str, arguments: dict
) -> str:
    """Call an MCP tool using the mcp package."""
    import asyncio
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
    from mcp.client.stdio import stdio_client  # type: ignore[import]

    params = StdioServerParameters(
        command=srv.command,
        args=srv.args,
        env={**{}, **srv.env} if srv.env else None,
    )

    async def _call() -> str:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        parts.append(content.text)
                    else:
                        parts.append(str(content))
                return "\n".join(parts)

    return asyncio.run(_call())


def _fetch_tools_raw(srv: MCPServerConfig, timeout: float) -> list[dict]:
    """
    Minimal JSON-RPC 2.0 MCP client over stdio.
    Launches the server, sends initialize + tools/list, returns tool specs.
    """
    import os

    env = {**os.environ, **srv.env}
    proc = subprocess.Popen(
        [srv.command, *srv.args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    def send(msg: dict) -> None:
        line = json.dumps(msg) + "\n"
        proc.stdin.write(line)  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    def recv(deadline: float) -> dict | None:
        while time.monotonic() < deadline:
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    deadline = time.monotonic() + timeout

    # Initialize
    send(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "blckhive-agent", "version": "1.0"},
            },
        }
    )
    init_resp = recv(deadline)
    if not init_resp or "error" in init_resp:
        proc.terminate()
        raise RuntimeError(f"MCP init failed: {init_resp}")

    # initialized notification
    send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # List tools
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools_resp = recv(deadline)
    proc.terminate()

    if not tools_resp or "error" in tools_resp:
        raise RuntimeError(f"MCP tools/list failed: {tools_resp}")

    return tools_resp.get("result", {}).get("tools", [])


def _call_mcp_tool(srv: MCPServerConfig, tool_name: str, arguments: dict) -> str:
    """Call a tool on an MCP server."""
    try:
        return _call_mcp_tool_via_package(srv, tool_name, arguments)
    except ImportError:
        return _call_mcp_tool_raw(srv, tool_name, arguments)


def _call_mcp_tool_raw(
    srv: MCPServerConfig, tool_name: str, arguments: dict, timeout: float = 30.0
) -> str:
    """Call an MCP tool using raw JSON-RPC."""
    import os

    env = {**os.environ, **srv.env}
    proc = subprocess.Popen(
        [srv.command, *srv.args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    def send(msg: dict) -> None:
        proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    def recv(deadline: float) -> dict | None:
        while time.monotonic() < deadline:
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    deadline = time.monotonic() + timeout

    send(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "blckhive-agent", "version": "1.0"},
            },
        }
    )
    recv(deadline)
    send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    send(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
    )
    result = recv(deadline)
    proc.terminate()

    if not result or "error" in result:
        raise RuntimeError(f"MCP tool call failed: {result}")

    contents = result.get("result", {}).get("content", [])
    parts = []
    for item in contents:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(str(item))
    return "\n".join(parts)
