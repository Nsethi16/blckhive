"""
Tool registry and built-in tools.

A "tool" is a Python callable that the agent can invoke via the Claude API
function-calling interface.  Tools are registered with a name, description,
JSON-Schema parameters, and an optional phase hint.

Custom tools can be registered at runtime:

    @registry.register(
        name="my_tool",
        description="Does something useful",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    )
    def my_tool(x: str) -> str:
        return f"got {x}"
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .core import Phase


@dataclass
class ToolDef:
    """Definition of a registered tool."""

    name: str
    description: str
    parameters: dict  # JSON Schema object
    func: Callable[..., str]
    phase_hint: Phase | None = None

    def to_anthropic_tool(self) -> dict:
        """Convert to the Anthropic API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    """
    Central registry for all available tools.

    Tools registered here are made available to the Claude API.
    The registry is passed into the agentic loop and updated by MCPManager.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._working_dir: Path = Path.cwd()

    def set_working_dir(self, path: Path) -> None:
        self._working_dir = Path(path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        phase_hint: Phase | None = None,
    ) -> Callable:
        """Decorator to register a function as a tool."""

        def decorator(func: Callable[..., str]) -> Callable:
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                phase_hint=phase_hint,
            )
            return func

        return decorator

    def register_tool(self, tool: ToolDef) -> None:
        """Register a pre-built ToolDef directly."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup / listing
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def to_anthropic_tools(self) -> list[dict]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: dict) -> tuple[str, bool]:
        """
        Execute a tool by name.  Returns (output_string, success).
        Never raises — errors are returned as error strings.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: '{name}'. Available: {self.list_tools()}", False
        try:
            result = tool.func(**arguments)
            return str(result), True
        except Exception as exc:  # noqa: BLE001
            return f"Error in {name}: {type(exc).__name__}: {exc}", False


# ---------------------------------------------------------------------------
# Built-in tools (registered on a shared default registry instance)
# ---------------------------------------------------------------------------

def _make_builtin_registry(working_dir: Path) -> ToolRegistry:
    """Build and return a ToolRegistry pre-loaded with the built-in tools."""

    reg = ToolRegistry()
    reg.set_working_dir(working_dir)

    # ---- GATHER phase -------------------------------------------------

    @reg.register(
        name="read_file",
        description=(
            "Read the full contents of a file. "
            "Returns text content, truncated at 30 000 chars."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the working directory.",
                },
            },
            "required": ["path"],
        },
        phase_hint=Phase.GATHER,
    )
    def read_file(path: str) -> str:
        target = reg._working_dir / path
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        content = target.read_text(encoding="utf-8", errors="replace")
        max_chars = 30_000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated — {len(content)} chars total]"
        return content

    @reg.register(
        name="list_directory",
        description=(
            "List files and subdirectories at a path. "
            "Returns names annotated with [dir] or [file]."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the working dir. Defaults to '.'.",
                },
            },
            "required": [],
        },
        phase_hint=Phase.GATHER,
    )
    def list_directory(path: str = ".") -> str:
        target = reg._working_dir / path
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")
        entries = sorted(target.iterdir())
        lines = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            kind = "[dir] " if entry.is_dir() else "[file]"
            lines.append(f"{kind} {entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"

    @reg.register(
        name="search_files",
        description=(
            "Search for a text pattern (regex) across files. "
            "Returns matching lines with file paths and line numbers. "
            "Limited to 50 matches."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for.",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob to limit search (e.g. '*.py'). Default: all files.",
                },
            },
            "required": ["pattern"],
        },
        phase_hint=Phase.GATHER,
    )
    def search_files(pattern: str, glob: str = "*") -> str:
        import re

        results = []
        for filepath in reg._working_dir.rglob(glob):
            if not filepath.is_file() or ".git" in filepath.parts:
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_num, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    rel = filepath.relative_to(reg._working_dir)
                    results.append(f"{rel}:{line_num}: {line.rstrip()}")
                    if len(results) >= 50:
                        results.append("... (truncated at 50 matches)")
                        return "\n".join(results)
        return "\n".join(results) if results else f"No matches for: {pattern}"

    # ---- ACTION phase -------------------------------------------------

    @reg.register(
        name="write_file",
        description=(
            "Write content to a file. "
            "Creates the file (and parent dirs) if it doesn't exist; overwrites if it does."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the working directory.",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write.",
                },
            },
            "required": ["path", "content"],
        },
        phase_hint=Phase.ACT,
    )
    def write_file(path: str, content: str) -> str:
        target = reg._working_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    @reg.register(
        name="edit_file",
        description=(
            "Replace an exact string in a file with new content. "
            "The old_string must appear exactly once in the file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the working directory.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        phase_hint=Phase.ACT,
    )
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = reg._working_dir / path
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"old_string not found in {path}")
        if count > 1:
            raise ValueError(f"old_string appears {count} times in {path} — must be unique")
        target.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path}: replaced 1 occurrence"

    @reg.register(
        name="run_command",
        description=(
            "Run a shell command and return stdout + stderr. "
            "Timeout: 120 seconds. Use for builds, tests, git, package installs, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120).",
                },
            },
            "required": ["command"],
        },
        phase_hint=Phase.ACT,
    )
    def run_command(command: str, timeout: int = 120) -> str:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(reg._working_dir),
            timeout=timeout,
        )
        parts = []
        if result.stdout.strip():
            parts.append(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr.strip():
            parts.append(f"STDERR:\n{result.stderr.strip()}")
        parts.append(f"EXIT CODE: {result.returncode}")
        return "\n".join(parts)

    # ---- VERIFY phase -------------------------------------------------

    @reg.register(
        name="task_complete",
        description=(
            "Signal that the task is finished. "
            "Provide a clear summary of what was accomplished (or why it couldn't be done)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of the outcome.",
                },
            },
            "required": ["summary"],
        },
        phase_hint=Phase.VERIFY,
    )
    def task_complete(summary: str) -> str:  # noqa: F811
        return f"DONE: {summary}"

    return reg
