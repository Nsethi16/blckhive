"""
Core agent data structures and the main run_agent() entrypoint.

The agent works through three blended phases:
  1. GATHER CONTEXT  - read files, list dirs, search, ask clarifying questions
  2. TAKE ACTION     - write/edit files, run commands, call APIs
  3. VERIFY RESULTS  - check output, run tests, confirm task completion

These phases are not sequential checkpoints — they blend together naturally
as Claude decides what it needs to do next.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolRegistry
    from .skills import SkillRegistry
    from .mcp import MCPManager
    from .config import AgentConfig


class Phase(str, Enum):
    """The three agent phases (used as hints, not hard gates)."""

    GATHER = "gather_context"
    ACT = "take_action"
    VERIFY = "verify_results"


@dataclass
class ToolResult:
    """The outcome of a single tool invocation."""

    tool_name: str
    arguments: dict
    output: str
    success: bool
    phase: Phase


@dataclass
class AgentState:
    """Tracks the full lifecycle of one agent run."""

    task: str
    phase: Phase = Phase.GATHER
    history: list[ToolResult] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    done: bool = False
    final_answer: str = ""
    error: str = ""

    @property
    def summary(self) -> str:
        lines = [
            f"Task   : {self.task}",
            f"Phase  : {self.phase.value}",
            f"Steps  : {len(self.history)}",
            f"Done   : {self.done}",
        ]
        for i, r in enumerate(self.history, 1):
            status = "OK  " if r.success else "FAIL"
            short_out = r.output[:80].replace("\n", " ")
            lines.append(
                f"  {i:02d}. [{status}] [{r.phase.value}] "
                f"{r.tool_name}({_fmt_args(r.arguments)}) -> {short_out}"
            )
        if self.final_answer:
            lines.append(f"\nResult : {self.final_answer[:300]}")
        if self.error:
            lines.append(f"\nError  : {self.error}")
        return "\n".join(lines)


def _fmt_args(args: dict) -> str:
    """Format tool arguments compactly for display."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous software agent.  You complete tasks by calling tools.
    You work through three blended phases — they are not rigid checkpoints:

    1. GATHER CONTEXT  — Read files, list directories, search code, understand the environment.
    2. TAKE ACTION     — Write files, edit code, run commands, call external APIs.
    3. VERIFY RESULTS  — Check outputs, run tests, confirm the task is truly done.

    Guidelines:
    - Always gather enough context before acting — never guess file contents.
    - Prefer targeted edits over full rewrites unless a full rewrite is clearly better.
    - After taking action, verify the result (run tests, read the file you just wrote, etc.).
    - Call `task_complete` with a clear summary when you are finished.
    - If you cannot complete the task, call `task_complete` with an explanation of what blocked you.
    - Think step-by-step before acting. Use tools in the order that makes the most sense.

    Available phase hints: gather_context, take_action, verify_results.
    The current phase is tracked automatically based on your tool calls.
""")


def run_agent(
    task: str,
    registry: "ToolRegistry",
    skills: "SkillRegistry",
    mcp: "MCPManager",
    config: "AgentConfig",
) -> AgentState:
    """
    Run the agent on a task.  Returns the final AgentState.

    This is the main entrypoint.  It:
      1. Attaches MCP server tools to the registry for this run.
      2. Builds the initial message list.
      3. Runs the streaming agentic loop via LLMClient.
      4. Returns the final state with history and answer.
    """
    from .llm import LLMClient

    state = AgentState(task=task)

    # Connect MCP servers and inject their tools into registry for this session
    mcp.attach_to_registry(registry)

    client = LLMClient(config=config)
    try:
        client.run(state=state, registry=registry, skills=skills, mcp=mcp)
    except Exception as exc:  # noqa: BLE001
        state.error = str(exc)
        state.done = True

    # Detach MCP tools so the registry stays clean between runs
    mcp.detach_from_registry(registry)

    return state
