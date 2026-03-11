"""
Example skill: git_workflow
Provides compound git operations as single skill calls.

To enable: copy this file to ~/.blckhive/skills/git_workflow.py
"""
from __future__ import annotations
from agent.skills import SkillRegistry


def register(skills: SkillRegistry) -> None:

    @skills.register(
        name="git_status_summary",
        description=(
            "Return a human-readable summary of the current git status: "
            "branch, staged files, unstaged files, and recent commits."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
    def git_status_summary(tool_registry) -> str:
        parts = []

        branch, _ = tool_registry.execute(
            "run_command", {"command": "git rev-parse --abbrev-ref HEAD"}
        )
        parts.append(f"Branch: {branch.strip()}")

        status, _ = tool_registry.execute("run_command", {"command": "git status --short"})
        parts.append(f"Status:\n{status.strip() or '  (clean)'}")

        log, _ = tool_registry.execute(
            "run_command",
            {"command": "git log --oneline -5"},
        )
        parts.append(f"Recent commits:\n{log.strip()}")

        return "\n\n".join(parts)

    @skills.register(
        name="git_commit_all",
        description=(
            "Stage all modified files and create a commit with the given message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message.",
                },
            },
            "required": ["message"],
        },
    )
    def git_commit_all(tool_registry, message: str) -> str:
        add_out, ok = tool_registry.execute("run_command", {"command": "git add -A"})
        if not ok:
            return f"git add failed: {add_out}"
        commit_out, ok = tool_registry.execute(
            "run_command", {"command": f'git commit -m {message!r}'}
        )
        return commit_out
