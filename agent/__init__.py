"""
blckhive agent — a simple, extensible agentic application powered by Claude.

Quick start:
    from agent import create_agent, run

    agent = create_agent()
    state = run(agent, "List the Python files in this project and summarise each one.")
    print(state.summary)

The agent works through three blended phases:
    1. Gather context  (read files, search, list dirs)
    2. Take action     (write/edit files, run commands)
    3. Verify results  (check outputs, run tests, confirm done)

Extending the agent
-------------------
Custom tools::

    @agent.tools.register(
        name="http_get",
        description="Perform an HTTP GET request and return the response body.",
        parameters={"type":"object","properties":{"url":{"type":"string"}},"required":["url"]},
    )
    def http_get(url: str) -> str:
        import urllib.request
        return urllib.request.urlopen(url).read().decode()

Custom skills (multi-step recipes)::

    @agent.skills.register(
        name="summarise_dir",
        description="List and summarise every .py file in a directory.",
        parameters={"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
    )
    def summarise_dir(tool_registry, path: str) -> str:
        listing, _ = tool_registry.execute("list_directory", {"path": path})
        return f"Directory: {path}\\n{listing}"

Adding MCP servers (persisted for future runs)::

    agent.mcp.add_server(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        description="Read/write access to /tmp via MCP",
    )
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

from .config import AgentConfig
from .core import AgentState, Phase, ToolResult, run_agent
from .tools import ToolRegistry, ToolDef, _make_builtin_registry
from .skills import SkillRegistry, SkillDef
from .mcp import MCPManager

__all__ = [
    # Main API
    "create_agent",
    "run",
    # Classes (for type annotations and extension)
    "Agent",
    "AgentConfig",
    "AgentState",
    "Phase",
    "ToolResult",
    "ToolRegistry",
    "ToolDef",
    "SkillRegistry",
    "SkillDef",
    "MCPManager",
]


@dataclass
class Agent:
    """
    Container that holds all the components of an agent session.

    Attributes:
        config:  Runtime settings (model, working dir, max turns, …).
        tools:   ToolRegistry with built-in tools + any custom tools you add.
        skills:  SkillRegistry with any custom skills you add.
        mcp:     MCPManager — add/remove MCP servers that persist across runs.
    """

    config: AgentConfig
    tools: ToolRegistry
    skills: SkillRegistry
    mcp: MCPManager


def create_agent(
    *,
    model: str = "claude-opus-4-6",
    working_dir: str | Path | None = None,
    max_turns: int = 30,
    max_tokens: int = 8192,
    stream_output: bool = True,
    state_dir: str | Path | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Agent:
    """
    Create and return a fully initialised Agent.

    Parameters
    ----------
    model:
        Anthropic model ID to use.  Defaults to claude-opus-4-6.
    working_dir:
        Root directory for file operations.  Defaults to cwd.
    max_turns:
        Hard cap on the number of agentic loop iterations.
    max_tokens:
        Maximum tokens per Claude response.
    stream_output:
        If True, print tokens and tool calls to stdout in real time.
    state_dir:
        Directory for persisted state (MCP configs, skill cache).
        Defaults to ~/.blckhive.
    extra_kwargs:
        Additional kwargs forwarded to AgentConfig.

    Returns
    -------
    Agent
        Ready-to-use agent.  Call ``run(agent, task)`` to execute a task.

    Examples
    --------
    Basic usage::

        agent = create_agent(working_dir="/my/project")
        state = run(agent, "Fix the failing tests in tests/test_api.py")
        print(state.final_answer)

    Load custom skills from a file::

        agent = create_agent()
        agent.skills.load_from_file(Path("my_skills.py"))
        state = run(agent, "use my_skill to …")
    """
    kwargs = extra_kwargs or {}

    config = AgentConfig(
        model=model,
        working_dir=Path(working_dir) if working_dir else Path.cwd(),
        max_turns=max_turns,
        max_tokens=max_tokens,
        stream_output=stream_output,
        **({"state_dir": Path(state_dir)} if state_dir else {}),
        **kwargs,
    )

    tools = _make_builtin_registry(config.working_dir)
    skills = SkillRegistry()
    mcp = MCPManager(config)

    # Auto-load user skills from the state dir
    user_skills_dir = config.skills_dir
    if user_skills_dir.is_dir():
        loaded = skills.load_from_directory(user_skills_dir)
        if loaded:
            print(f"[agent] Loaded user skills: {', '.join(loaded)}")

    return Agent(config=config, tools=tools, skills=skills, mcp=mcp)


def run(agent: Agent, task: str) -> AgentState:
    """
    Execute a task with the given agent.

    This is a thin wrapper around ``run_agent`` that picks up the
    components from the Agent dataclass.

    Parameters
    ----------
    agent:
        Agent created by ``create_agent()``.
    task:
        Natural-language task description.

    Returns
    -------
    AgentState
        Final state with .history, .final_answer, .summary, and .done.
    """
    if agent.config.stream_output:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

    return run_agent(
        task=task,
        registry=agent.tools,
        skills=agent.skills,
        mcp=agent.mcp,
        config=agent.config,
    )
