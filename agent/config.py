"""
Configuration for the blckhive agent.
All settings can be overridden via environment variables or constructor arguments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Runtime configuration for an agent session."""

    # Claude model to use
    model: str = field(
        default_factory=lambda: os.environ.get("AGENT_MODEL", "claude-opus-4-6")
    )

    # Working directory for file operations
    working_dir: Path = field(default_factory=lambda: Path.cwd())

    # Maximum agentic loop turns (safety cap)
    max_turns: int = field(
        default_factory=lambda: int(os.environ.get("AGENT_MAX_TURNS", "30"))
    )

    # Max tokens per Claude response
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("AGENT_MAX_TOKENS", "8192"))
    )

    # Whether to stream output to stdout as it arrives
    stream_output: bool = True

    # Directory for persisted agent state (MCP configs, skill cache, etc.)
    state_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AGENT_STATE_DIR", str(Path.home() / ".blckhive"))
        )
    )

    def __post_init__(self) -> None:
        self.working_dir = Path(self.working_dir)
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def mcp_config_path(self) -> Path:
        return self.state_dir / "mcp_servers.json"

    @property
    def skills_dir(self) -> Path:
        return self.state_dir / "skills"
