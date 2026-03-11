"""
Example skill: code_review
Reads a file and produces a focused code-review checklist.

To enable: copy this file to ~/.blckhive/skills/code_review.py
"""
from __future__ import annotations
from agent.skills import SkillRegistry


def register(skills: SkillRegistry) -> None:

    @skills.register(
        name="code_review",
        description=(
            "Review a source file for common issues: "
            "style, security, performance, and correctness. "
            "Returns a structured checklist."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to review (relative to working dir).",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional focus area: 'security', 'performance', "
                        "'style', or 'all' (default)."
                    ),
                },
            },
            "required": ["path"],
        },
    )
    def code_review(tool_registry, path: str, focus: str = "all") -> str:
        content, ok = tool_registry.execute("read_file", {"path": path})
        if not ok:
            return f"Could not read {path}: {content}"

        lines = content.splitlines()
        issues: list[str] = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Security checks
            if focus in ("all", "security"):
                for pattern in ("eval(", "exec(", "pickle.loads", "shell=True",
                                "password", "secret", "api_key", "token"):
                    if pattern.lower() in stripped.lower():
                        issues.append(f"  [SECURITY] Line {i}: possible sensitive pattern '{pattern}'")

            # Style checks
            if focus in ("all", "style"):
                if len(line.rstrip()) > 120:
                    issues.append(f"  [STYLE] Line {i}: line exceeds 120 chars ({len(line.rstrip())})")
                if "\t" in line:
                    issues.append(f"  [STYLE] Line {i}: tab character (use spaces)")

            # Performance checks
            if focus in ("all", "performance"):
                for pattern in ("time.sleep(", "while True:", "SELECT *"):
                    if pattern in stripped:
                        issues.append(f"  [PERF] Line {i}: consider reviewing '{pattern}'")

        summary = [
            f"Code review of {path} ({len(lines)} lines)",
            f"Focus: {focus}",
            f"Issues found: {len(issues)}",
            "",
        ]
        if issues:
            summary.extend(issues)
        else:
            summary.append("  No issues found.")

        return "\n".join(summary)
