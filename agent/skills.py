"""
Skill registry and built-in skills.

A "skill" is a higher-level, multi-step recipe that composes tools and/or
calls Claude to accomplish a well-defined sub-task.  Skills appear to the
agent as tools but are implemented in Python, giving you a way to encode
domain knowledge without prompt engineering.

Skills can be:
  1. Registered inline with the @skills.register() decorator.
  2. Loaded from .py files in the skills directory (each file must define
     a module-level `register(registry: SkillRegistry)` function).

Example skill file (~/.blckhive/skills/my_skill.py):

    from agent.skills import SkillRegistry

    def register(skills: SkillRegistry) -> None:
        @skills.register(
            name="summarise_file",
            description="Read a file and return a one-paragraph summary.",
            parameters={"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
        )
        def summarise_file(tool_registry, path: str) -> str:
            content, _ = tool_registry.execute("read_file", {"path": path})
            # Could call Claude here, or just use the content directly
            return content[:500] + "..."
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolRegistry


@dataclass
class SkillDef:
    """
    A skill is a multi-step recipe that the agent can call as a tool.

    The skill function receives the ToolRegistry as its first argument so it
    can invoke other tools directly, followed by keyword arguments matching
    the declared parameters.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable  # (tool_registry: ToolRegistry, **kwargs) -> str

    def to_anthropic_tool(self) -> dict:
        return {
            "name": self.name,
            "description": f"[SKILL] {self.description}",
            "input_schema": self.parameters,
        }


class SkillRegistry:
    """Registry of named skills."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
    ) -> Callable:
        """Decorator to register a function as a skill."""

        def decorator(func: Callable) -> Callable:
            self._skills[name] = SkillDef(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
            )
            return func

        return decorator

    def register_skill(self, skill: SkillDef) -> None:
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup / listing
    # ------------------------------------------------------------------

    def get(self, name: str) -> SkillDef | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def to_anthropic_tools(self) -> list[dict]:
        return [s.to_anthropic_tool() for s in self._skills.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self, name: str, arguments: dict, tool_registry: "ToolRegistry"
    ) -> tuple[str, bool]:
        """Execute a skill.  Returns (output_string, success)."""
        skill = self._skills.get(name)
        if not skill:
            return f"Unknown skill: '{name}'. Available: {self.list_skills()}", False
        try:
            result = skill.func(tool_registry, **arguments)
            return str(result), True
        except Exception as exc:  # noqa: BLE001
            return f"Error in skill {name}: {type(exc).__name__}: {exc}", False

    # ------------------------------------------------------------------
    # File-based loading
    # ------------------------------------------------------------------

    def load_from_directory(self, skills_dir: Path) -> list[str]:
        """
        Load skills from all .py files in skills_dir.

        Each file must define a top-level function:
            def register(skills: SkillRegistry) -> None: ...

        Returns list of loaded skill names.
        """
        if not skills_dir.is_dir():
            return []

        loaded: list[str] = []
        for skill_file in sorted(skills_dir.glob("*.py")):
            if skill_file.name.startswith("_"):
                continue
            try:
                before = set(self._skills.keys())
                _load_skill_module(skill_file, self)
                after = set(self._skills.keys())
                loaded.extend(after - before)
            except Exception as exc:  # noqa: BLE001
                print(f"[skills] Warning: failed to load {skill_file.name}: {exc}")

        return loaded

    def load_from_file(self, skill_file: Path) -> list[str]:
        """Load skills from a single .py file."""
        before = set(self._skills.keys())
        _load_skill_module(skill_file, self)
        after = set(self._skills.keys())
        return list(after - before)


def _load_skill_module(path: Path, registry: SkillRegistry) -> None:
    """Import a skill file and call its register() function."""
    module_name = f"_skill_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "register"):
        raise AttributeError(f"{path} must define a top-level register(skills) function")
    module.register(registry)
